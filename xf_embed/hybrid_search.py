import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from xf_embed.config import settings
from xf_embed.db import get_db_pool, search_mariadb_native_vector
from xf_embed.elastic_engine import search_elasticsearch
from xf_embed.embeddings import generate_embedding
from xf_embed.vector_engine import search_faiss

logger = logging.getLogger(__name__)


def compute_item_key(item: Dict[str, Any]) -> str:
    """Generates a stable unique deduplication key for a search item."""
    if item.get("post_id"):
        return f"post:{item['post_id']}"
    if item.get("thread_id"):
        return f"thread:{item['thread_id']}"
    # Fallback to record_id or title
    return f"rec:{item.get('record_id', item.get('thread_title', ''))}"


def _merge_missing_fields(existing: Dict[str, Any], item: Dict[str, Any]) -> None:
    for field in ("thread_title", "message", "username", "post_date", "thread_id"):
        if not existing.get(field) and item.get(field):
            existing[field] = item[field]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Computes Reciprocal Rank Fusion (RRF) across multiple ranked result sets.
    RRF Score = sum(1 / (k + rank)) for each rank list where the item appears.
    """
    scores: Dict[str, float] = {}
    item_lookup: Dict[str, Dict[str, Any]] = {}
    source_ranks: Dict[str, Dict[str, int]] = {}

    list_names = ["vector", "elastic"] if len(ranked_lists) == 2 else [f"list_{i}" for i in range(len(ranked_lists))]

    for list_idx, result_list in enumerate(ranked_lists):
        source_name = list_names[list_idx] if list_idx < len(list_names) else f"source_{list_idx}"
        for rank, item in enumerate(result_list, start=1):
            key = compute_item_key(item)
            if key not in item_lookup:
                item_lookup[key] = dict(item)
                scores[key] = 0.0
                source_ranks[key] = {}
            else:
                # Merge fields if missing (e.g. title or message from one source)
                _merge_missing_fields(item_lookup[key], item)

            # Multiple embedding records can resolve to the same forum item.
            # A source contributes only its first (best) rank for that item.
            if source_name in source_ranks[key]:
                continue

            rrf_val = 1.0 / (k + rank)
            scores[key] += rrf_val
            source_ranks[key][source_name] = rank

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for key in sorted_keys:
        item = item_lookup[key]
        item["rrf_score"] = round(scores[key], 6)
        item["score"] = item["rrf_score"]
        item["ranks"] = source_ranks.get(key, {})
        fused_results.append(item)

    return fused_results


def weighted_score_fusion(
    vector_results: List[Dict[str, Any]],
    elastic_results: List[Dict[str, Any]],
    alpha: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Normalizes vector and BM25 scores to [0, 1] and fuses them linearly:
    Score = alpha * norm_vector + (1 - alpha) * norm_bm25
    """
    def min_max_normalize(items: List[Dict[str, Any]], score_key: str = "score") -> Dict[str, float]:
        if not items:
            return {}
        scores: Dict[str, float] = {}
        for it in items:
            key = compute_item_key(it)
            score = float(it.get(score_key, 0.0))
            scores[key] = max(scores.get(key, score), score)
        min_s = min(scores.values())
        max_s = max(scores.values())
        if max_s == min_s:
            # A singleton or tied source still supplies retrieval evidence.
            return dict.fromkeys(scores, 1.0)
        span = max_s - min_s
        return {key: (score - min_s) / span for key, score in scores.items()}

    v_norm = min_max_normalize(vector_results, "score")
    e_norm = min_max_normalize(elastic_results, "es_score")

    combined_keys = list(v_norm)
    combined_keys.extend(key for key in e_norm if key not in v_norm)
    item_lookup: Dict[str, Dict[str, Any]] = {}
    for it in vector_results + elastic_results:
        k = compute_item_key(it)
        if k not in item_lookup:
            item_lookup[k] = dict(it)
        else:
            _merge_missing_fields(item_lookup[k], it)

    fused = []
    for key in combined_keys:
        vs = v_norm.get(key, 0.0)
        es = e_norm.get(key, 0.0)
        combined_score = round(alpha * vs + (1.0 - alpha) * es, 6)
        item = item_lookup[key]
        item["score"] = combined_score
        item["vector_norm_score"] = round(vs, 4)
        item["elastic_norm_score"] = round(es, 4)
        fused.append(item)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


async def perform_vector_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Generates query embedding and queries vector backend (FAISS or MariaDB native)."""
    query_vector = await generate_embedding(query)

    if settings.VECTOR_BACKEND == "mariadb":
        pool = await get_db_pool()
        return await search_mariadb_native_vector(pool, query_vector, top_k=max_results)
    else:
        return await search_faiss(query_vector, top_k=max_results)


async def perform_hybrid_search(
    query: str,
    max_results: int = 10,
    algorithm: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes concurrent vector and Elasticsearch searches using asyncio.gather,
    then combines them using RRF or weighted score fusion.
    Returns (fused_results, vector_results, elastic_results).
    """
    chosen_algo = settings.HYBRID_ALGORITHM if algorithm is None else algorithm
    if chosen_algo not in ("rrf", "weighted"):
        raise ValueError(f"Unsupported hybrid algorithm: {chosen_algo!r}")

    fetch_k = max(max_results * 2, 20)  # Fetch slightly more candidates for fusion

    # Run both vector search and Elasticsearch in parallel non-blocking async
    vector_task = asyncio.create_task(perform_vector_search(query, max_results=fetch_k))
    elastic_task = asyncio.create_task(search_elasticsearch(query, max_results=fetch_k))

    tasks = (vector_task, elastic_task)
    try:
        vector_results, elastic_results = await asyncio.gather(*tasks)
    except BaseException:
        # gather propagates one backend failure without stopping the other.
        # Finish cancellation before the request returns or releases clients.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if chosen_algo == "weighted":
        fused = weighted_score_fusion(vector_results, elastic_results, alpha=settings.HYBRID_ALPHA)
    else:
        fused = reciprocal_rank_fusion([vector_results, elastic_results], k=settings.RRF_K)

    return fused[:max_results], vector_results[:max_results], elastic_results[:max_results]
