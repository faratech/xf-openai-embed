import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from xf_embed.config import settings
from xf_embed.db import fetch_details_for_posts_and_threads, get_db_pool
from xf_embed.visibility import guest_visible_node_ids

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
    visible_node_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """
    Builds Elasticsearch BM25 query DSL with field weights and optional recency decay.
    """
    now_timestamp = int(time.time())

    filters: List[Dict[str, Any]] = [{"terms": {"type": ["post", "thread"]}}]
    if visible_node_ids is not None:
        filters.append({"terms": {"node": sorted(visible_node_ids)}})

    query_dsl: Dict[str, Any] = {
        "bool": {
            "must": [{
                "simple_query_string": {
                    "query": query,
                    "fields": ["title^3", "message"],
                    "default_operator": "and",
                }
            }],
            "filter": filters,
            "must_not": [{"term": {"hidden": True}}],
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
            "discussion_id",
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
    target_index = index_name or settings.ELASTICSEARCH_INDEX

    try:
        pool = await get_db_pool()
        visible_nodes = await guest_visible_node_ids(pool)
        if not visible_nodes:
            return []
        es = get_es_client()
        dsl = build_bm25_query_dsl(query, max_results=max_results, visible_node_ids=visible_nodes)
        response = await es.search(index=target_index, body=dsl)
        results = []
        hits = response.get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
            score = float(hit.get("_score") or 0.0)

            # Derive post_id or thread_id from id or source fields
            hit_id = str(hit.get("_id", ""))
            hit_type, _, hit_content_id = hit_id.rpartition("-")
            item_type = source.get("type") or hit_type
            if item_type not in ("post", "thread"):
                continue
            post_id = source.get("post_id") if item_type == "post" else None
            thread_id = source.get("discussion_id") or source.get("thread")

            try:
                if item_type == "post":
                    post_id = int(post_id or (hit_content_id if hit_type == "post" else 0))
                    if post_id <= 0:
                        continue
                else:
                    thread_id = int(thread_id or (hit_content_id if hit_type == "thread" else 0))
                    if thread_id <= 0:
                        continue
                if thread_id is not None:
                    thread_id = int(thread_id)
            except (ValueError, TypeError):
                continue

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
        # ES may lag deletion, moderation, permission or content edits. Refresh
        # and authorize candidates in one batch before any caller can cache them.
        return await fetch_details_for_posts_and_threads(pool, results, visible_node_ids=visible_nodes)
    except Exception as e:
        logger.error(f"Elasticsearch search failed on index '{target_index}': {e}")
        # A backend outage is not a valid empty result: callers must not cache
        # it as a successful search for SEARCH_RESULT_CACHE_TTL.
        raise


def _bulk_document_id(document: Dict[str, Any]) -> str:
    explicit_id = document.get("_id")
    if explicit_id is not None and str(explicit_id):
        return str(explicit_id)
    source = document.get("source", document)
    fields = ("thread_id", "post_id") if source.get("type") == "thread" else ("post_id", "thread_id")
    for field in fields:
        content_id = source.get(field, document.get(field))
        if content_id is not None:
            return f"{field[:-3]}-{content_id}"
    raise ValueError("Bulk document has no identity: provide _id, post_id or thread_id")


async def bulk_index_documents(
    documents: List[Dict[str, Any]],
    index_name: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Bulk index documents using async_bulk for ultra-fast batch ingestion.
    Returns (success_count, error_count).
    """
    if not documents:
        return 0, 0

    # Reject deterministic identity errors before any earlier chunk is written.
    for document in documents:
        _bulk_document_id(document)

    es = get_es_client()
    target_index = index_name or settings.ELASTICSEARCH_INDEX

    actions = (
        {
            "_index": target_index,
            "_id": _bulk_document_id(doc),
            "_source": doc.get("source", doc),
        }
        for doc in documents
    )

    success, errors = await async_bulk(
        client=es,
        actions=actions,
        chunk_size=500,
        raise_on_error=False,
    )
    return success, len(errors) if isinstance(errors, list) else (1 if errors else 0)
