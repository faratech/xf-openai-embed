import os
import asyncio
import aiomysql
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import numpy as np
import logging
import argparse
from tqdm import tqdm

# Load environment variables
load_dotenv('/web/.env')

# Initialize Elasticsearch client
es = AsyncElasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_HOST", "http://127.0.0.1:9200")],
    basic_auth=(os.getenv("ELASTICSEARCH_USER", "wf_wf"), os.getenv("ELASTICSEARCH_PASSWORD", ""))
)

# Define the Elasticsearch index name
INDEX_NAME = 'wf_embeddings'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BATCH_SIZE = 100  # Batch size for fetching from the database
EXPECTED_EMBEDDING_LENGTH = 1536  # Expected length of each embedding

async def create_index(overwrite=False):
    if overwrite:
        # Delete the existing index if it exists and overwrite is set to True
        if await es.indices.exists(index=INDEX_NAME):
            await es.indices.delete(index=INDEX_NAME)
            logging.info(f"Index {INDEX_NAME} deleted for overwrite.")

    # Define index settings and mappings
    mappings = {
        "mappings": {
            "properties": {
                "post_id": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": EXPECTED_EMBEDDING_LENGTH},
                "embedding_length": {"type": "integer"}
            }
        }
    }

    # Create the index if it doesn't exist
    if not await es.indices.exists(index=INDEX_NAME):
        await es.indices.create(index=INDEX_NAME, body=mappings)
        logging.info(f"Index {INDEX_NAME} created.")
    else:
        logging.info(f"Index {INDEX_NAME} already exists.")

async def get_db_pool():
    return await aiomysql.create_pool(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "wf_wf"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        db=os.getenv("MYSQL_DATABASE", "your_database_name"),
        autocommit=True,
        pool_recycle=3600,
        maxsize=20
    )

async def fetch_data_from_mysql(pool, last_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            # Fetch data from MySQL
            await cursor.execute('''
                SELECT post_id, embedding, embedding_length
                FROM openai_embeddings
                WHERE post_id > %s
                ORDER BY post_id ASC
                LIMIT %s
            ''', (last_id, BATCH_SIZE))
            rows = await cursor.fetchall()
    return rows

def validate_embedding(embedding):
    """ Validate the embedding to ensure no NaN, Inf, or extreme values exist. """
    if len(embedding) != EXPECTED_EMBEDDING_LENGTH:
        logging.error(f"Unexpected embedding length: {len(embedding)}, expected {EXPECTED_EMBEDDING_LENGTH}")
        return False
    
    if not np.isfinite(embedding).all():
        logging.error(f"Embedding contains NaN or Infinity values")
        return False
    
    if np.max(np.abs(embedding)) > 100:  # Arbitrary threshold, adjust if needed
        logging.error(f"Embedding contains extreme values: max abs value = {np.max(np.abs(embedding))}")
        return False
    
    return True

async def index_document(post_id, embedding_blob, stored_length):
    try:
        # Convert embedding to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        # Validate the embedding
        if not validate_embedding(embedding):
            logging.error(f"Invalid embedding detected for document {post_id}")
            return False

        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()

        # Index document in Elasticsearch
        await es.index(
            index=INDEX_NAME,
            id=post_id,
            document={
                "post_id": post_id,
                "embedding": embedding_list,
                "embedding_length": stored_length
            }
        )
        logging.info(f"Successfully indexed document {post_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to index document {post_id}: {e}")
        return False

async def process_data(pool):
    last_id = 0
    total_indexed = 0
    total_processed = 0

    with tqdm(desc="Indexing documents") as pbar:
        while True:
            # Fetch the next batch of data
            data = await fetch_data_from_mysql(pool, last_id)

            if not data:
                break  # No more data to index

            for post_id, embedding_blob, stored_length in data:
                total_processed += 1
                success = await index_document(post_id, embedding_blob, stored_length)
                if success:
                    total_indexed += 1
                pbar.update(1)

            # Update last_id to the highest post_id in the current batch
            last_id = data[-1][0]

    logging.info(f"Total documents processed: {total_processed}")
    logging.info(f"Total documents successfully indexed: {total_indexed}")

async def main(overwrite=False):
    pool = await get_db_pool()
    try:
        await create_index(overwrite=overwrite)
        await process_data(pool)
    finally:
        pool.close()
        await pool.wait_closed()
        await es.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding indexing script for Elasticsearch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing index")
    args = parser.parse_args()

    try:
        asyncio.run(main(overwrite=args.overwrite))
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
