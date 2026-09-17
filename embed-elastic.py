#!/usr/bin/env python3
"""
embed-elastic.py: High-speed bulk indexer for forum embeddings into Elasticsearch.
Uses elasticsearch.helpers.async_bulk for 50-100x faster ingestion.
"""

import os
import asyncio
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from xf_embed.config import settings
from xf_embed.db import get_db_pool, close_db_pool

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INDEX_NAME = os.getenv("ELASTICSEARCH_EMBEDDINGS_INDEX", "wf_embeddings")
BATCH_SIZE = 1000  # High-throughput batch size for bulk indexing
EXPECTED_EMBEDDING_LENGTH = settings.OPENAI_DIMENSIONS


def get_es_client() -> AsyncElasticsearch:
    auth = None
    if settings.ELASTICSEARCH_USER and settings.ELASTICSEARCH_PASSWORD:
        auth = (settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD)
    return AsyncElasticsearch(
        hosts=[settings.ELASTICSEARCH_HOST],
        basic_auth=auth,
        request_timeout=30.0,
        max_retries=3,
        retry_on_timeout=True,
    )


async def create_index(es: AsyncElasticsearch, overwrite: bool = False):
    if overwrite:
        if await es.indices.exists(index=INDEX_NAME):
            await es.indices.delete(index=INDEX_NAME)
            logging.info(f"Index '{INDEX_NAME}' deleted for overwrite.")

    mappings = {
        "mappings": {
            "properties": {
                "post_id": {"type": "integer"},
                "thread_id": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": EXPECTED_EMBEDDING_LENGTH},
                "embedding_length": {"type": "integer"},
            }
        }
    }

    if not await es.indices.exists(index=INDEX_NAME):
        await es.indices.create(index=INDEX_NAME, body=mappings)
        logging.info(f"Index '{INDEX_NAME}' created.")
    else:
        logging.info(f"Index '{INDEX_NAME}' already exists.")


async def fetch_data_from_mysql(pool, last_id: int):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT id, post_id, thread_id, embedding, embedding_length
                FROM openai_embeddings
                WHERE id > %s AND embedding IS NOT NULL
                ORDER BY id ASC
                LIMIT %s
            """,
                (last_id, BATCH_SIZE),
            )
            rows = await cursor.fetchall()
    return rows


async def process_data(pool, es: AsyncElasticsearch):
    last_id = 0
    total_indexed = 0
    total_processed = 0

    # Get total count for progress estimation
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM openai_embeddings WHERE embedding IS NOT NULL")
            total_docs = (await cursor.fetchone())[0]

    with tqdm(total=total_docs, desc="Bulk indexing documents into ES") as pbar:
        while True:
            data = await fetch_data_from_mysql(pool, last_id)
            if not data:
                break

            actions = []
            for rec_id, post_id, thread_id, embedding_blob, stored_length in data:
                total_processed += 1
                if not embedding_blob:
                    continue

                if isinstance(embedding_blob, (bytes, bytearray)):
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                elif isinstance(embedding_blob, str):
                    try:
                        embedding = np.array(json.loads(embedding_blob), dtype=np.float32)
                    except Exception:
                        continue
                else:
                    continue

                if len(embedding) != EXPECTED_EMBEDDING_LENGTH or not np.isfinite(embedding).all():
                    continue

                doc_id = f"post-{post_id}" if post_id else f"thread-{thread_id}"
                actions.append(
                    {
                        "_index": INDEX_NAME,
                        "_id": doc_id,
                        "_source": {
                            "post_id": post_id,
                            "thread_id": thread_id,
                            "embedding": embedding.tolist(),
                            "embedding_length": stored_length or len(embedding),
                        },
                    }
                )

            if actions:
                success, errors = await async_bulk(
                    client=es,
                    actions=actions,
                    chunk_size=len(actions),
                    raise_on_error=False,
                )
                total_indexed += success
                pbar.update(len(data))

            last_id = data[-1][0]

    logging.info(f"Total documents processed: {total_processed}")
    logging.info(f"Total documents successfully indexed: {total_indexed}")


async def main(overwrite: bool = False):
    pool = await get_db_pool()
    es = get_es_client()
    try:
        await create_index(es, overwrite=overwrite)
        await process_data(pool, es)
    finally:
        await close_db_pool()
        await es.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-speed bulk embedding indexer for Elasticsearch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing index")
    args = parser.parse_args()

    try:
        asyncio.run(main(overwrite=args.overwrite))
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
