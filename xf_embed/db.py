import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncmy.cursors
import numpy as np

from xf_embed.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncmy.Pool] = None


async def get_db_pool() -> asyncmy.Pool:
    """Get or create the global aiomysql connection pool."""
    global _pool
    if _pool is None:
        logger.info("Initializing aiomysql database connection pool...")
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
    """Close the aiomysql connection pool."""
    global _pool
    if _pool is not None:
        logger.info("Closing aiomysql connection pool...")
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("Database connection pool closed.")


async def fetch_embeddings_batch(
    pool: asyncmy.Pool,
    last_id: int = 0,
    batch_size: int = 10000,
) -> Tuple[List[int], List[Optional[int]], List[Optional[int]], np.ndarray, int]:
    """
    Fetch a batch of embeddings from openai_embeddings using keyset pagination (id > last_id).
    Returns (record_ids, post_ids, thread_ids, embeddings_array, next_last_id).
    """
    record_ids: List[int] = []
    post_ids: List[Optional[int]] = []
    thread_ids: List[Optional[int]] = []
    vectors: List[np.ndarray] = []
    next_last_id = last_id

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

            for row in rows:
                rec_id, p_id, t_id, emb_blob = row
                next_last_id = rec_id
                if not emb_blob:
                    continue

                if isinstance(emb_blob, (bytes, bytearray)):
                    vec = np.frombuffer(emb_blob, dtype=np.float32)
                elif isinstance(emb_blob, str):
                    try:
                        vec = np.array(json.loads(emb_blob), dtype=np.float32)
                    except Exception:
                        continue
                else:
                    continue

                if len(vec) == settings.OPENAI_DIMENSIONS:
                    record_ids.append(rec_id)
                    post_ids.append(p_id)
                    thread_ids.append(t_id)
                    vectors.append(vec)

    if vectors:
        vectors_arr = np.vstack(vectors)
    else:
        vectors_arr = np.empty((0, settings.OPENAI_DIMENSIONS), dtype=np.float32)

    return record_ids, post_ids, thread_ids, vectors_arr, next_last_id


async def fetch_details_for_posts_and_threads(
    pool: asyncmy.Pool,
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enriches search results (list of dicts containing 'post_id' or 'thread_id')
    with full metadata from xf_post and xf_thread using parameterized queries.
    """
    if not items:
        return []

    post_ids = [item["post_id"] for item in items if item.get("post_id")]
    thread_ids = [item["thread_id"] for item in items if item.get("thread_id") and not item.get("post_id")]

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
                        t.title AS thread_title
                    FROM xf_post p
                    LEFT JOIN xf_thread t ON p.thread_id = t.thread_id
                    WHERE p.post_id IN ({format_strings})
                """
                await cursor.execute(post_query, post_ids)
                rows = await cursor.fetchall()
                for r in rows:
                    post_map[r["post_id"]] = r

            if thread_ids:
                format_strings = ",".join(["%s"] * len(thread_ids))
                thread_query = f"""
                    SELECT
                        thread_id,
                        title AS thread_title,
                        post_date,
                        last_post_date,
                        user_id,
                        username
                    FROM xf_thread
                    WHERE thread_id IN ({format_strings})
                """
                await cursor.execute(thread_query, thread_ids)
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
            enriched.append(base)
        elif t_id and t_id in thread_map:
            t_data = thread_map[t_id]
            base["type"] = "thread"
            base["thread_id"] = t_id
            base["thread_title"] = t_data.get("thread_title") or ""
            base["message"] = ""
            base["post_date"] = t_data.get("last_post_date") or t_data.get("post_date") or 0
            base["username"] = t_data.get("username") or ""
            enriched.append(base)
        else:
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

    format_strings = ",".join(["%s"] * len(record_ids))
    query = f"""
        SELECT
            oe.id AS record_id,
            oe.post_id,
            oe.thread_id,
            COALESCE(p.message, '') AS message,
            COALESCE(t.title, xt.title, '') AS thread_title,
            COALESCE(p.post_date, xt.last_post_date, 0) AS post_date,
            COALESCE(p.username, xt.username, '') AS username
        FROM openai_embeddings oe
        LEFT JOIN xf_post p ON oe.post_id = p.post_id
        LEFT JOIN xf_thread t ON p.thread_id = t.thread_id
        LEFT JOIN xf_thread xt ON oe.thread_id = xt.thread_id
        WHERE oe.id IN ({format_strings})
    """

    results = {}
    async with pool.acquire() as conn:
        async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
            await cursor.execute(query, record_ids)
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
    vector_json = "[" + ",".join(f"{x:.6f}" for x in query_embedding.tolist()) + "]"

    query = """
        SELECT
            e.id,
            e.post_id,
            e.thread_id,
            COALESCE(p.message, '') AS message,
            COALESCE(t.title, '') AS thread_title,
            COALESCE(p.post_date, xt.last_post_date, 0) AS post_date,
            VEC_DISTANCE_COSINE(e.embedding, VEC_FromText(%s)) AS distance
        FROM openai_embeddings e
        LEFT JOIN xf_post p ON e.post_id = p.post_id
        LEFT JOIN xf_thread t ON p.thread_id = t.thread_id
        LEFT JOIN xf_thread xt ON e.thread_id = xt.thread_id
        WHERE e.embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT %s
    """

    async with pool.acquire() as conn:
        async with conn.cursor(asyncmy.cursors.DictCursor) as cursor:
            await cursor.execute(query, (vector_json, top_k))
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
            "score": similarity,
            "distance": distance,
        })
    return results
