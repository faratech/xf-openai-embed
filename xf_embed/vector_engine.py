import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from xf_embed.config import settings
from xf_embed.db import fetch_details_for_posts_and_threads, get_db_pool
from xf_embed.embeddings import normalize_vector

logger = logging.getLogger(__name__)

_faiss_index: Optional[faiss.Index] = None
_index_lock = asyncio.Lock()


def create_base_faiss_index(dimension: int) -> faiss.Index:
    """Create a new FAISS index with IndexIDMap2 for custom 64-bit ID tracking."""
    if settings.FAISS_METRIC == "cosine":
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        metric = faiss.METRIC_L2

    if settings.FAISS_INDEX_TYPE == "hnsw":
        logger.info(f"Creating HNSW index (M={settings.FAISS_HNSW_M}, metric={metric})...")
        sub_index = faiss.IndexHNSWFlat(dimension, settings.FAISS_HNSW_M, metric)
        sub_index.hnsw.efSearch = settings.FAISS_HNSW_EF_SEARCH
        sub_index.hnsw.efConstruction = settings.FAISS_HNSW_EF_CONSTRUCTION
    else:
        logger.info(f"Creating Flat index (metric={metric})...")
        if metric == faiss.METRIC_INNER_PRODUCT:
            sub_index = faiss.IndexFlatIP(dimension)
        else:
            sub_index = faiss.IndexFlatL2(dimension)

    index = faiss.IndexIDMap2(sub_index)
    return index


def save_faiss_index(index: faiss.Index, path: Optional[str] = None) -> None:
    filepath = path or settings.FAISS_INDEX_PATH
    logger.info(f"Writing FAISS index to {filepath} (total vectors: {index.ntotal})...")
    faiss.write_index(index, filepath)
    logger.info(f"FAISS index successfully saved to {filepath}")


def load_faiss_index_from_disk(path: Optional[str] = None) -> Optional[faiss.Index]:
    filepath = path or settings.FAISS_INDEX_PATH
    if os.path.exists(filepath):
        logger.info(f"Loading FAISS index from {filepath}...")
        index = faiss.read_index(filepath)
        logger.info(f"FAISS index loaded ({index.ntotal} vectors).")
        return index
    return None


async def get_faiss_index() -> Optional[faiss.Index]:
    """Retrieves the active FAISS index, loading it from disk if not yet loaded."""
    global _faiss_index
    if _faiss_index is None:
        async with _index_lock:
            if _faiss_index is None:
                _faiss_index = load_faiss_index_from_disk()
    return _faiss_index


async def build_faiss_index(
    batch_size: int = 10000,
    save_path: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> faiss.Index:
    """
    Builds a complete FAISS index by streaming vectors from MySQL in batches.
    Memory-efficient: avoids loading all 680k+ embeddings into Python all at once.
    """
    global _faiss_index
    pool = await get_db_pool()
    dimension = settings.OPENAI_DIMENSIONS
    index = create_base_faiss_index(dimension)

    last_id = 0
    total_added = 0
    logger.info("Starting FAISS index build from database...")

    from xf_embed.db import fetch_embeddings_batch

    while True:
        record_ids, _, _, vectors, next_last_id = await fetch_embeddings_batch(
            pool=pool,
            last_id=last_id,
            batch_size=batch_size,
        )

        if len(record_ids) == 0:
            break

        if settings.FAISS_METRIC == "cosine":
            vectors = normalize_vector(vectors)

        ids_array = np.array(record_ids, dtype=np.int64)
        index.add_with_ids(vectors, ids_array)
        total_added += len(record_ids)
        last_id = next_last_id

        if progress_callback:
            progress_callback(len(record_ids), total_added)
        else:
            logger.info(f"Indexed {total_added} vectors (latest DB id: {last_id})...")

    save_faiss_index(index, save_path)
    async with _index_lock:
        _faiss_index = index

    logger.info(f"FAISS index built successfully with {total_added} total vectors.")
    return index


async def search_faiss(
    query_vector: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index with query vector, resolves the record IDs to forum
    posts/threads, and returns enriched result objects.
    """
    index = await get_faiss_index()
    if index is None or index.ntotal == 0:
        logger.warning("FAISS index is not initialized or empty.")
        return []

    # Ensure query vector is unit normalized for cosine similarity
    if settings.FAISS_METRIC == "cosine":
        query_vector = normalize_vector(query_vector)

    query_mat = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

    # Perform FAISS search (C++ thread-safe execution)
    distances, indices = index.search(query_mat, top_k)

    scores = distances[0]
    record_ids = indices[0]

    # Filter out invalid indices (-1 in FAISS denotes empty results)
    valid_results = [(int(rec_id), float(score)) for rec_id, score in zip(record_ids, scores) if rec_id != -1]
    if not valid_results:
        return []

    # Map record_ids back to post_id/thread_id from DB in a single unified query
    valid_ids = [r[0] for r in valid_results]
    id_to_score = {r[0]: r[1] for r in valid_results}

    pool = await get_db_pool()
    from xf_embed.db import fetch_details_by_record_ids
    details_map = await fetch_details_by_record_ids(pool, valid_ids)

    enriched = []
    for rec_id in valid_ids:
        if rec_id in details_map:
            item = dict(details_map[rec_id])
            score = id_to_score.get(rec_id, 0.0)
            similarity = float(score) if settings.FAISS_METRIC == "cosine" else max(0.0, 1.0 / (1.0 + float(score)))
            item["score"] = similarity
            item["distance"] = float(score)
            enriched.append(item)

    # Sort enriched items according to score descending
    enriched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return enriched
