#!/usr/bin/env python3
"""
embed-vector-search.py: Vector similarity search using MariaDB native VECTOR functions.
Uses VEC_DISTANCE_COSINE() for in-database cosine similarity search.
"""

import sys
import os
import asyncio
from xf_embed.config import settings
from xf_embed.embeddings import generate_embedding
from xf_embed.db import get_db_pool, close_db_pool, search_mariadb_native_vector


async def run_search(query: str, top_n: int = 10):
    print(f"Generating embedding for query: '{query}'...")
    pool = await get_db_pool()
    try:
        vec = await generate_embedding(query)
        print(f"Searching MariaDB native vector database (top {top_n})...")
        results = await search_mariadb_native_vector(pool, vec, top_k=top_n)
        return results
    finally:
        await close_db_pool()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 embed-vector-search.py <query> [top_n]")
        print("Example: python3 embed-vector-search.py 'blue screen error' 10")
        sys.exit(1)

    query = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = asyncio.run(run_search(query, top_n))

    print(f"\nTop {len(results)} similar posts/threads:\n")
    print("-" * 80)

    for r in results:
        title = r.get("thread_title") or "N/A"
        msg = r.get("message") or "N/A"
        title_preview = (title[:60] + "...") if len(title) > 60 else title
        msg_preview = (msg[:100] + "...").replace("\n", " ") if len(msg) > 100 else msg.replace("\n", " ")

        print(f"Type: {r.get('type')} | Post ID: {r.get('post_id')} | Thread ID: {r.get('thread_id')} | Similarity: {r.get('score', 0.0):.4f}")
        print(f"Title: {title_preview}")
        print(f"Message: {msg_preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()
