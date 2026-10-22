import asyncio
import json
import logging
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple
import asyncmy.cursors
import numpy as np

from xf_embed.config import settings
from xf_embed.visibility import guest_visible_node_ids

logger = logging.getLogger(__name__)

_pool: Optional[asyncmy.Pool] = None
_pool_lock = asyncio.Lock()


async def get_db_pool() -> asyncmy.Pool:
    """Get or create the shared asyncmy connection pool."""
    global _pool
    async with _pool_lock:
        if _pool is None:
            logger.info("Initializing asyncmy database connection pool...")
            _pool = await asyncmy.create_pool(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                db=settings.MYSQL_DATABASE,
                charset=settings.MYSQL_CHARSET,
                minsize=1,
                maxsize=settings.MYSQL_POOL_SIZE,
                autocommit=True,
                pool_recycle=3600,
            )
            logger.info("Database connection pool initialized.")
        return _pool


async def close_db_pool() -> None:
    """Close the shared asyncmy connection pool."""
    global _pool
    async with _pool_lock:
        if _pool is not None:
            logger.info("Closing asyncmy connection pool...")
            _pool.close()
            await _pool.wait_closed()
            _pool = None
            logger.info("Database connection pool closed.")


def _decode_embeddings_batch(
    rows: List[Tuple[Any, ...]], last_id: int, dimensions: int,
) -> Tuple[List[int], List[Optional[int]], List[Optional[int]], np.ndarray, int]:
    record_ids: List[int] = []
    post_ids: List[Optional[int]] = []
    thread_ids: List[Optional[int]] = []
    vectors: List[np.ndarray] = []
    next_last_id = last_id

    for rec_id, p_id, t_id, emb_blob in rows:
        # Pagination must advance even when every vector on this page is invalid.
        next_last_id = rec_id
        try:
            if isinstance(emb_blob, (bytes, bytearray)):
                vec = np.frombuffer(emb_blob, dtype=np.float32)
            elif isinstance(emb_blob, str):
                vec = np.asarray(json.loads(emb_blob), dtype=np.float32)
            else:
                continue
        except (TypeError, ValueError, OverflowError):
            continue

        if vec.shape != (dimensions,) or not np.isfinite(vec).all() or not np.any(vec):
            continue
        record_ids.append(rec_id)
        post_ids.append(p_id)
        thread_ids.append(t_id)
        vectors.append(vec)

    vectors_arr = np.vstack(vectors) if vectors else np.empty((0, dimensions), dtype=np.float32)
    return record_ids, post_ids, thread_ids, vectors_arr, next_last_id


async def fetch_embeddings_batch(
    pool: asyncmy.Pool,
    last_id: int = 0,
    batch_size: int = 10000,
) -> Tuple[List[int], List[Optional[int]], List[Optional[int]], np.ndarray, int]:
    """
    Fetch a batch of embeddings from openai_embeddings using keyset pagination (id > last_id).
    Returns (record_ids, post_ids, thread_ids, embeddings_array, next_last_id).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    query = """
        SELECT id, post_id, thread_id, embedding
        FROM openai_embeddings
        WHERE id > %s AND embedding IS NOT NULL
        ORDER BY id ASC
        LIMIT %s
    """

    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (last_id, batch_size))
            rows = await cursor.fetchall()

    # Release the database connection before CPU-heavy decoding/stacking.
    return await asyncio.to_thread(_decode_embeddings_batch, rows, last_id, settings.OPENAI_DIMENSIONS)


async def fetch_details_for_posts_and_threads(
    pool: asyncmy.Pool,
    items: List[Dict[str, Any]],
    visible_node_ids: Optional[frozenset[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Enriches search results (list of dicts containing 'post_id' or 'thread_id')
    with full metadata from xf_post and xf_thread using parameterized queries.
    """
    if not items:
        return []
    visible_nodes = await guest_visible_node_ids(pool) if visible_node_ids is None else visible_node_ids
    if not visible_nodes:
        return []
    node_ids = sorted(visible_nodes)
    node_placeholders = ",".join(["%s"] * len(node_ids))

    post_ids = list(dict.fromkeys(item["post_id"] for item in items if item.get("post_id")))
    thread_ids = list(dict.fromkeys(
        item["thread_id"] for item in items if item.get("thread_id") and not item.get("post_id")
    ))

    post_map: Dict[int, Dict[str, Any]] = {}
    thread_map: Dict[int, Dict[str, Any]] = {}

    async with pool.acquire() as conn:
        async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
            if post_ids:
                format_strings = ",".join(["%s"] * len(post_ids))
                post_query = f"""
                    SELECT
                        p.post_id,
                        p.thread_id,
                        p.message,
                        p.post_date,
                        p.user_id,
                        p.username,
                        t.node_id,
                        t.title AS thread_title
                    FROM xf_post p
                    INNER JOIN xf_thread t ON p.thread_id = t.thread_id
                    WHERE p.post_id IN ({format_strings})
                        AND p.message_state = 'visible'
                        AND t.discussion_state = 'visible'
                        AND t.node_id IN ({node_placeholders})
                """
                await cursor.execute(post_query, post_ids + node_ids)
                rows = await cursor.fetchall()
                for r in rows:
                    post_map[r["post_id"]] = r

            if thread_ids:
                format_strings = ",".join(["%s"] * len(thread_ids))
                thread_query = f"""
                    SELECT
                        t.thread_id,
                        t.title AS thread_title,
                        t.post_date,
                        t.last_post_date,
                        t.user_id,
                        t.username,
                        t.node_id,
                        COALESCE(first_post.message, '') AS message
                    FROM xf_thread t
                    LEFT JOIN xf_post first_post ON first_post.post_id = t.first_post_id
                        AND first_post.thread_id = t.thread_id AND first_post.message_state = 'visible'
                    WHERE t.thread_id IN ({format_strings})
                        AND t.discussion_state = 'visible'
                        AND t.node_id IN ({node_placeholders})
                """
                await cursor.execute(thread_query, thread_ids + node_ids)
                rows = await cursor.fetchall()
                for r in rows:
                    thread_map[r["thread_id"]] = r

    enriched: List[Dict[str, Any]] = []
    for item in items:
        p_id = item.get("post_id")
        t_id = item.get("thread_id")
        base = dict(item)

        if p_id and p_id in post_map:
            p_data = post_map[p_id]
            base["type"] = "post"
            base["thread_id"] = p_data.get("thread_id")
            base["thread_title"] = p_data.get("thread_title") or ""
            base["message"] = p_data.get("message") or ""
            base["post_date"] = p_data.get("post_date") or 0
            base["username"] = p_data.get("username") or ""
            base["user_id"] = p_data.get("user_id") or 0
            base["node_id"] = p_data.get("node_id")
            enriched.append(base)
        elif not p_id and t_id and t_id in thread_map:
            t_data = thread_map[t_id]
            base["type"] = "thread"
            base["thread_id"] = t_id
            base["thread_title"] = t_data.get("thread_title") or ""
            base["message"] = t_data.get("message") or ""
            base["post_date"] = t_data.get("last_post_date") or t_data.get("post_date") or 0
            base["username"] = t_data.get("username") or ""
            base["user_id"] = t_data.get("user_id") or 0
            base["node_id"] = t_data.get("node_id")
            enriched.append(base)

    return enriched


async def fetch_details_by_record_ids(
    pool: asyncmy.Pool,
    record_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Fetches post/thread metadata for FAISS record IDs in a single unified SQL query with JOINs.
    Eliminates multiple sequential database round trips.
    """
    if not record_ids:
        return {}
    visible_nodes = sorted(await guest_visible_node_ids(pool))
    if not visible_nodes:
        return {}

    format_strings = ",".join(["%s"] * len(record_ids))
    node_placeholders = ",".join(["%s"] * len(visible_nodes))
    query = f"""
        SELECT
            oe.id AS record_id,
            oe.post_id,
            COALESCE(p.thread_id, oe.thread_id) AS thread_id,
            COALESCE(p.message, '') AS message,
            COALESCE(t.title, xt.title, '') AS thread_title,
            COALESCE(p.post_date, xt.last_post_date, 0) AS post_date,
            COALESCE(p.username, xt.username, '') AS username,
            COALESCE(p.user_id, xt.user_id, 0) AS user_id,
            COALESCE(t.node_id, xt.node_id) AS node_id
        FROM openai_embeddings oe
        LEFT JOIN xf_post p ON oe.post_id = p.post_id
        LEFT JOIN xf_thread t ON p.thread_id = t.thread_id
        LEFT JOIN xf_thread xt ON oe.thread_id = xt.thread_id
        WHERE oe.id IN ({format_strings})
            AND (
                (oe.post_id IS NOT NULL AND p.message_state = 'visible'
                    AND t.discussion_state = 'visible' AND t.node_id IN ({node_placeholders}))
                OR (oe.post_id IS NULL AND xt.discussion_state = 'visible'
                    AND xt.node_id IN ({node_placeholders}))
            )
    """

    results = {}
    async with pool.acquire() as conn:
        async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
            await cursor.execute(query, record_ids + visible_nodes + visible_nodes)
            rows = await cursor.fetchall()
            for r in rows:
                r["type"] = "post" if r.get("post_id") else "thread"
                results[r["record_id"]] = r

    return results


async def search_mariadb_native_vector(
    pool: asyncmy.Pool,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Performs cosine similarity search directly inside MariaDB 12 using VEC_DISTANCE_COSINE.
    Distance = 1 - cosine_similarity. Similarity = 1 - distance.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    visible_nodes = sorted(await guest_visible_node_ids(pool))
    if not visible_nodes:
        return []
    node_placeholders = ",".join(["%s"] * len(visible_nodes))
    vector_json = "[" + ",".join(f"{x:.6f}" for x in query_embedding.tolist()) + "]"
    zero_vector = np.zeros(settings.OPENAI_DIMENSIONS, dtype=np.float32).tobytes()

    query = f"""
        SELECT
            candidates.post_id,
            candidates.thread_id,
            COALESCE(p.message, '') AS message,
            COALESCE(t.title, '') AS thread_title,
            COALESCE(p.post_date, t.last_post_date, 0) AS post_date,
            COALESCE(p.username, t.username, '') AS username,
            COALESCE(p.user_id, t.user_id, 0) AS user_id,
            t.node_id,
            candidates.distance
        FROM (
            SELECT
                e.post_id,
                COALESCE(p.thread_id, e.thread_id) AS thread_id,
                MIN(VEC_DISTANCE_COSINE(e.embedding, VEC_FromText(%s))) AS distance
            FROM openai_embeddings e
            LEFT JOIN xf_post p ON e.post_id = p.post_id
            LEFT JOIN xf_thread t ON p.thread_id = t.thread_id
            LEFT JOIN xf_thread xt ON e.post_id IS NULL AND e.thread_id = xt.thread_id
            WHERE e.embedding IS NOT NULL
                AND OCTET_LENGTH(e.embedding) = %s
                AND VEC_DISTANCE_EUCLIDEAN(e.embedding, %s) > 0
                AND (
                    (e.post_id IS NOT NULL AND p.message_state = 'visible'
                        AND t.discussion_state = 'visible' AND t.node_id IN ({node_placeholders}))
                    OR (e.post_id IS NULL AND xt.discussion_state = 'visible'
                        AND xt.node_id IN ({node_placeholders}))
                )
            GROUP BY e.post_id, COALESCE(p.thread_id, e.thread_id)
            HAVING distance IS NOT NULL
            ORDER BY distance ASC, e.post_id ASC, COALESCE(p.thread_id, e.thread_id) ASC
            LIMIT %s
        ) candidates
        LEFT JOIN xf_post p ON candidates.post_id = p.post_id
        LEFT JOIN xf_thread t ON candidates.thread_id = t.thread_id
        ORDER BY candidates.distance ASC, candidates.post_id ASC, candidates.thread_id ASC
    """

    async with pool.acquire() as conn:
        async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
            await cursor.execute(
                query, [vector_json, settings.OPENAI_DIMENSIONS * 4, zero_vector]
                + visible_nodes + visible_nodes + [top_k],
            )
            rows = await cursor.fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        distance = float(r["distance"])
        similarity = max(0.0, 1.0 - distance)
        item_type = "post" if r.get("post_id") else "thread"
        results.append({
            "type": item_type,
            "post_id": r.get("post_id"),
            "thread_id": r.get("thread_id"),
            "thread_title": r.get("thread_title"),
            "message": r.get("message"),
            "post_date": r.get("post_date"),
            "username": r.get("username"),
            "user_id": r.get("user_id"),
            "node_id": r.get("node_id"),
            "score": similarity,
            "distance": distance,
        })
    return results
