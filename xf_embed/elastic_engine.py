import logging
import time
from typing import Any, Dict, List, Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from xf_embed.config import settings

logger = logging.getLogger(__name__)

_es_client: Optional[AsyncElasticsearch] = None


def get_es_client() -> AsyncElasticsearch:
    """Returns singleton AsyncElasticsearch client."""
    global _es_client
    if _es_client is None:
        auth = None
        if settings.ELASTICSEARCH_USER and settings.ELASTICSEARCH_PASSWORD:
            auth = (settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD)

        _es_client = AsyncElasticsearch(
            hosts=[settings.ELASTICSEARCH_HOST],
            basic_auth=auth,
            request_timeout=15.0,
            max_retries=3,
            retry_on_timeout=True,
        )
    return _es_client


async def close_es_client() -> None:
    """Closes singleton AsyncElasticsearch client."""
    global _es_client
    if _es_client is not None:
        logger.info("Closing Elasticsearch client...")
        await _es_client.close()
        _es_client = None
        logger.info("Elasticsearch client closed.")


def build_bm25_query_dsl(
    query: str,
    max_results: int = 10,
    apply_recency_decay: bool = True,
) -> Dict[str, Any]:
    """
    Builds Elasticsearch BM25 query DSL with field weights and optional recency decay.
    """
    now_timestamp = int(time.time())

    query_dsl: Dict[str, Any] = {
        "simple_query_string": {
            "query": query,
            "fields": ["title^3", "message"],
            "default_operator": "and",
        }
    }

    if apply_recency_decay:
        final_query: Dict[str, Any] = {
            "function_score": {
                "query": query_dsl,
                "functions": [
                    {
                        "exp": {
                            "date": {
                                "origin": now_timestamp,
                                "scale": "30d",
                                "decay": 0.5,
                            }
                        }
                    }
                ],
                "boost_mode": "sum",
            }
        }
    else:
        final_query = query_dsl

    search_dsl = {
        "query": final_query,
        "size": max_results,
        "sort": [
            {"_score": "desc"},
            {"date": "desc"},
        ],
        "_source": [
            "title",
            "message",
            "date",
            "user",
            "discussion_id",
            "node",
            "post_id",
            "type",
            "thread",
        ],
    }

    return search_dsl


async def search_elasticsearch(
    query: str,
    max_results: int = 10,
    index_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes keyword BM25 search in Elasticsearch.
    """
    es = get_es_client()
    target_index = index_name or settings.ELASTICSEARCH_INDEX
    dsl = build_bm25_query_dsl(query, max_results=max_results)

    try:
        response = await es.search(index=target_index, body=dsl)
        results = []
        hits = response.get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
            score = float(hit.get("_score", 0.0))

            # Derive post_id or thread_id from id or source fields
            hit_id = str(hit.get("_id", ""))
            post_id = source.get("post_id")
            thread_id = source.get("discussion_id") or source.get("thread")
            item_type = source.get("type", "post")

            if not post_id and hit_id.startswith("post-"):
                try:
                    post_id = int(hit_id.replace("post-", ""))
                except ValueError:
                    pass

            results.append({
                "type": item_type,
                "post_id": post_id,
                "thread_id": thread_id,
                "thread_title": source.get("title", ""),
                "message": source.get("message", ""),
                "post_date": source.get("date", 0),
                "score": score,
                "es_score": score,
            })
        return results
    except Exception as e:
        logger.error(f"Elasticsearch search failed on index '{target_index}': {e}")
        return []


async def bulk_index_documents(
    documents: List[Dict[str, Any]],
    index_name: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Bulk index documents using async_bulk for ultra-fast batch ingestion.
    Returns (success_count, error_count).
    """
    es = get_es_client()
    target_index = index_name or settings.ELASTICSEARCH_INDEX

    actions = [
        {
            "_index": target_index,
            "_id": doc.get("_id") or f"post-{doc.get('post_id')}",
            "_source": doc.get("source", doc),
        }
        for doc in documents
    ]

    success, errors = await async_bulk(
        client=es,
        actions=actions,
        chunk_size=len(actions),
        raise_on_error=False,
    )
    return success, len(errors) if isinstance(errors, list) else (1 if errors else 0)
