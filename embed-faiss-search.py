#!/usr/bin/env python3
"""
embed-faiss-search.py: Standalone FAISS vector similarity search script.
"""

import sys
import asyncio
from xf_embed.config import settings
from xf_embed.embeddings import generate_embedding
from xf_embed.vector_engine import search_faiss, get_faiss_index, load_faiss_index_from_disk
from xf_embed.db import close_db_pool


async def run_search(query: str, top_k: int = 10):
    idx = await get_faiss_index()
    if not idx:
        print(f"Error: FAISS index file '{settings.FAISS_INDEX_PATH}' not found.")
        print("Build it first using: xf-search index-faiss")
        return []

    print(f"Generating query embedding with {settings.OPENAI_EMBEDDING_MODEL}...")
    vec = await generate_embedding(query)
    print(f"Searching FAISS index ({idx.ntotal} vectors)...")
    results = await search_faiss(vec, top_k=top_k)
    return results


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your search query: ")

    if not query.strip():
        print("Empty query.")
        return

    try:
        results = asyncio.run(run_search(query, top_k=10))
    finally:
        asyncio.run(close_db_pool())

    print(f"\nTop {len(results)} most similar results:\n" + "-" * 60)
    for r in results:
        title = r.get("thread_title") or "N/A"
        msg = (r.get("message") or "")[:120].replace("\n", " ")
        print(f"[{r.get('type', 'post').upper()}] Post ID: {r.get('post_id')} | Thread ID: {r.get('thread_id')}")
        print(f"Similarity Score: {r.get('score', 0.0):.4f}")
        print(f"Title: {title}")
        print(f"Message: {msg}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
