import asyncio
import logging
import os
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from xf_embed.config import settings
from xf_embed.db import get_db_pool
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
    filepath = Path(path or settings.FAISS_INDEX_PATH)
    logger.info(f"Writing FAISS index to {filepath} (total vectors: {index.ntotal})...")
    # Keep the last usable index intact if writing fails or the process exits.
    with tempfile.TemporaryDirectory(prefix=f".{filepath.name}.", dir=filepath.parent) as temporary_dir:
        temporary_path = str(Path(temporary_dir) / filepath.name)
        faiss.write_index(index, temporary_path)
        if filepath.exists():
            previous = filepath.stat()
            os.chmod(temporary_path, previous.st_mode & 0o777)
            current = os.stat(temporary_path)
            if (current.st_uid, current.st_gid) != (previous.st_uid, previous.st_gid):
                os.chown(temporary_path, previous.st_uid, previous.st_gid)
        os.replace(temporary_path, filepath)
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
                _faiss_index = await asyncio.to_thread(load_faiss_index_from_disk)
    return _faiss_index


def _add_embeddings(index: faiss.Index, vectors: np.ndarray, record_ids: List[int]) -> None:
    if settings.FAISS_METRIC == "cosine":
        vectors = normalize_vector(vectors)
    index.add_with_ids(vectors, np.asarray(record_ids, dtype=np.int64))


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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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

        if next_last_id <= last_id:
            break
        last_id = next_last_id
        if not record_ids:
            continue

        await asyncio.to_thread(_add_embeddings, index, vectors, record_ids)
        total_added += len(record_ids)

        if progress_callback:
            progress_callback(len(record_ids), total_added)
        else:
            logger.info(f"Indexed {total_added} vectors (latest DB id: {last_id})...")

    await asyncio.to_thread(save_faiss_index, index, save_path)
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
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    index = await get_faiss_index()
    if index is None or index.ntotal == 0:
        logger.warning("FAISS index is not initialized or empty.")
        return []

    # Ensure query vector is unit normalized for cosine similarity
    if settings.FAISS_METRIC == "cosine":
        query_vector = normalize_vector(query_vector)

    query_mat = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

    from xf_embed.db import fetch_details_by_record_ids
    total_vectors = int(index.ntotal)
    candidate_count = min(top_k, total_vectors)
    pool = None
    hydrated_ids = set()
    details_map = {}

    while True:
        # Authorization and section deduplication can remove the nearest rows.
        # Expand until enough forum items remain or every vector was considered.
        distances, indices = await asyncio.to_thread(index.search, query_mat, candidate_count)
        enriched = {}
        for offset in range(0, len(indices[0]), 1000):
            batch = [
                (int(rec_id), float(score)) for rec_id, score in zip(
                    indices[0][offset:offset + 1000], distances[0][offset:offset + 1000],
                ) if rec_id != -1
            ]
            new_ids = list(dict.fromkeys(rec_id for rec_id, _ in batch if rec_id not in hydrated_ids))
            if new_ids:
                if pool is None:
                    pool = await get_db_pool()
                details_map.update(await fetch_details_by_record_ids(pool, new_ids))
                hydrated_ids.update(new_ids)
            else:
                # Large previously checked prefixes should not monopolize the
                # event loop while the candidate window expands.
                await asyncio.sleep(0)

            for rec_id, score in batch:
                if rec_id not in details_map:
                    continue
                item = dict(details_map[rec_id])
                if item.get("post_id"):
                    key = ("post", item["post_id"])
                elif item.get("thread_id"):
                    key = ("thread", item["thread_id"])
                else:
                    key = ("record", rec_id)
                similarity = score if settings.FAISS_METRIC == "cosine" else max(0.0, 1.0 / (1.0 + score))
                if key in enriched and enriched[key]["score"] >= similarity:
                    continue
                item["score"] = similarity
                item["distance"] = score
                enriched[key] = item

            # FAISS returns candidates in metric order. Later batches cannot
            # improve these best unique hits, so avoid hydrating the whole window.
            if len(enriched) >= top_k:
                break

        ranked = sorted(enriched.values(), key=lambda item: item["score"], reverse=True)
        if len(ranked) >= top_k or candidate_count >= total_vectors:
            return ranked[:top_k]
        candidate_count = min(candidate_count * 2, total_vectors)
