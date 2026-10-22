import asyncio
import time

from xf_embed.config import settings
from xf_embed.elastic_engine import search_elasticsearch as _search_elasticsearch
from xf_embed.cli import _close_clients


def build_query(query, max_results=10):
    # Convert "now" to an appropriate timestamp
    now_timestamp = int(time.time())

    # Building the query DSL
    query_dsl = {
        "simple_query_string": {
            "query": query,
            "fields": ["title^3", "message"],
            "default_operator": "and"
        }
    }

    # Applying function score for recency with decay
    function_score_dsl = {
        "function_score": {
            "query": query_dsl,
            "functions": [
                {
                    "exp": {
                        "date": {
                            "origin": now_timestamp,
                            "scale": "10d",  # Adjust decay scale as needed
                            "decay": 0.5
                        }
                    }
                }
            ],
            "boost_mode": "sum"
        }
    }

    # Assembling the final query DSL
    search_dsl = {
        "query": function_score_dsl,
        "size": max_results,
        "sort": [
            {"_score": "desc"},
            {"date": "desc"}
        ],
        "_source": ["title", "message", "date", "user", "discussion_id", "node"]
    }

    return search_dsl

def search_elasticsearch(query, max_results=10):
    """Keep legacy source aliases while using the authorized common engine."""
    async def run():
        try:
            results = await _search_elasticsearch(
                query, max_results=max_results, index_name=settings.ELASTICSEARCH_INDEX,
            )
            return [{
                "title": item.get("thread_title", ""),
                "message": item.get("message", ""),
                "date": item.get("post_date", 0),
                "user": item.get("user_id"),
                "discussion_id": item.get("thread_id"),
                "node": item.get("node_id"),
            } for item in results]
        finally:
            await _close_clients()

    return asyncio.run(run())


def main():
    query = input("Enter your search query: ")
    top_es_results = search_elasticsearch(query)

    # Output the results
    if top_es_results:
        for result in top_es_results:
            print(result)
    else:
        print("No results found or an error occurred.")

if __name__ == "__main__":
    main()
