#!/usr/bin/env python3
"""
embed-faiss-search-with-embed.py: Standalone Hybrid search (FAISS + Elasticsearch).
"""

import sys
import asyncio
from xf_embed.config import settings
from xf_embed.hybrid_search import perform_hybrid_search
from xf_embed.db import close_db_pool
from xf_embed.elastic_engine import close_es_client


async def run_hybrid(query: str, max_results: int = 10):
    print(f"Executing hybrid search for: '{query}' (fusion: RRF k={settings.RRF_K})...")
    fused, vec_hits, es_hits = await perform_hybrid_search(query, max_results=max_results)
    return fused, vec_hits, es_hits


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your search query: ")

    if not query.strip():
        print("Empty query.")
        return

    try:
        fused, vec_hits, es_hits = asyncio.run(run_hybrid(query, max_results=10))
    finally:
        asyncio.run(close_db_pool())
        asyncio.run(close_es_client())

    print(f"\nTop {len(fused)} Combined Results (Ranked by Reciprocal Rank Fusion):\n" + "=" * 70)
    for idx, r in enumerate(fused, start=1):
        title = r.get("thread_title") or "N/A"
        msg = (r.get("message") or "")[:120].replace("\n", " ")
        ranks = r.get("ranks", {})
        rank_info = f"vector rank: {ranks.get('vector', 'N/A')}, elastic rank: {ranks.get('elastic', 'N/A')}"
        print(f"#{idx} [{r.get('type', 'post').upper()}] Post ID: {r.get('post_id')} | Thread ID: {r.get('thread_id')}")
        print(f"RRF Score: {r.get('score', 0.0):.6f} ({rank_info})")
        print(f"Title: {title}")
        print(f"Message: {msg}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
