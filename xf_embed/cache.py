import asyncio
from collections import OrderedDict
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, Optional

import numpy as np
import redis.asyncio as aioredis

from xf_embed.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None
_redis_lock = asyncio.Lock()
_redis_retry_at = 0.0
_REDIS_RETRY_DELAY = 5.0
_local_memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_MEMORY_CACHE_MAX_SIZE = 2048


def compute_query_hash(query: str) -> str:
    """Shared XenForo query hash; preserve its normalization contract."""
    return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()


def _vector_cache_keys(query: str):
    h = compute_query_hash(query)
    qualified = f"v2:{settings.OPENAI_EMBEDDING_MODEL}:{settings.OPENAI_DIMENSIONS}:{h}"
    # The historical XenForo namespace has no model metadata. Share only its
    # established default embedding space; all other models are isolated.
    keys = [f"xf_embed:qemb:{qualified}"]
    if settings.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small" and settings.OPENAI_DIMENSIONS == 1536:
        keys.extend((f"xfai:qemb:{h}", f"xf:xfai:qemb:{h}", f"xf_embed:qemb:{h}"))
    return f"qemb:{qualified}", keys


def compute_search_cache_key(query: str, mode: str, max_results: int, algorithm: Optional[str] = None) -> str:
    # Include configuration that changes ranking and embedding spaces. Versioning
    # also prevents serving results produced before metadata/fusion fixes.
    key = ["v3", query.strip().lower(), mode, max_results, algorithm or settings.HYBRID_ALGORITHM,
           settings.HYBRID_ALPHA, settings.RRF_K, settings.VECTOR_BACKEND,
           settings.FAISS_METRIC, settings.FAISS_INDEX_PATH, settings.ELASTICSEARCH_INDEX,
           settings.OPENAI_EMBEDDING_MODEL, settings.OPENAI_DIMENSIONS]
    return hashlib.md5(json.dumps(key).encode("utf-8")).hexdigest()


def _memory_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _local_memory_cache.get(key)
    if entry is None:
        return None
    if entry["expires_at"] <= time.monotonic():
        del _local_memory_cache[key]
        return None
    _local_memory_cache.move_to_end(key)
    return entry


def _memory_put(key: str, value: Dict[str, Any], ttl: float) -> None:
    if ttl <= 0 or _MEMORY_CACHE_MAX_SIZE <= 0:
        _local_memory_cache.pop(key, None)
        return
    _local_memory_cache.pop(key, None)
    while len(_local_memory_cache) >= _MEMORY_CACHE_MAX_SIZE:
        _local_memory_cache.popitem(last=False)
    _local_memory_cache[key] = {**value, "expires_at": time.monotonic() + ttl}


def _promotion_ttl(milliseconds: int, cap: float) -> float:
    if milliseconds == -1:  # Source intentionally has no expiry.
        return cap
    return min(cap, max(0.0, milliseconds / 1000.0))


async def _redis_failed(client: aioredis.Redis) -> None:
    global _redis_client, _redis_retry_at
    if _redis_client is client or _redis_client is None:
        _redis_client = None
        _redis_retry_at = time.monotonic() + _REDIS_RETRY_DELAY
    try:
        await client.aclose()
    except Exception:
        pass


async def get_redis_client() -> Optional[aioredis.Redis]:
    """Share a pool and briefly back off after failure; memory stays available."""
    global _redis_client
    if not settings.CACHE_ENABLED or time.monotonic() < _redis_retry_at:
        return None
    if _redis_client is None:
        async with _redis_lock:
            if _redis_client is not None:
                return _redis_client
            if time.monotonic() < _redis_retry_at:
                return None
            client = aioredis.Redis(
                host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                db=settings.REDIS_DB, password=settings.REDIS_PASSWORD,
                decode_responses=False, socket_timeout=1.5,
                socket_connect_timeout=1.5,
            )
            try:
                await client.ping()
            except asyncio.CancelledError:
                await client.aclose()
                raise
            except Exception as exc:
                await _redis_failed(client)
                logger.warning("Could not connect to Redis; using memory cache: %s", exc)
                return None
            _redis_client = client
    return _redis_client


async def close_cache() -> None:
    global _redis_client, _redis_retry_at
    async with _redis_lock:
        client, _redis_client = _redis_client, None
        _redis_retry_at = 0.0
        _local_memory_cache.clear()
        if client is not None:
            await client.aclose()


def _decode_vector(raw: bytes) -> Optional[np.ndarray]:
    try:
        try:
            values = json.loads(raw)
        except (ValueError, UnicodeError):
            import phpserialize
            values = phpserialize.loads(raw)
            if isinstance(values, dict):
                values = [value for _, value in sorted(values.items())]
        vector = np.asarray(values, dtype=np.float32)
        if (vector.shape == (settings.OPENAI_DIMENSIONS,) and np.isfinite(vector).all()
                and np.any(vector)):
            return vector
    except (ValueError, TypeError, ImportError, OverflowError):
        pass
    return None


async def get_cached_query_vector(query: str) -> Optional[np.ndarray]:
    if not settings.CACHE_ENABLED:
        return None
    mem_key, keys = _vector_cache_keys(query)
    entry = _memory_get(mem_key)
    if entry is not None:
        return entry["vector"]
    client = await get_redis_client()
    if client is not None:
        try:
            # Read value and remaining lifetime together in one round trip.
            async with client.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.get(key)
                    pipe.pttl(key)
                values = await pipe.execute()
            for raw, remaining in zip(values[::2], values[1::2]):
                ttl = _promotion_ttl(remaining, 600)
                if raw and ttl > 0:
                    vector = _decode_vector(raw)
                    if vector is not None:
                        _memory_put(mem_key, {"vector": vector}, ttl)
                        return vector
        except Exception as exc:
            await _redis_failed(client)
            logger.warning("Redis vector read failed: %s", exc)
    return None


async def cache_query_vector(query: str, vector: np.ndarray, ttl: Optional[int] = None) -> None:
    if not settings.CACHE_ENABLED:
        return
    target_ttl = settings.QUERY_VECTOR_CACHE_TTL if ttl is None else ttl
    if target_ttl <= 0:
        return
    mem_key, keys = _vector_cache_keys(query)
    _memory_put(mem_key, {"vector": vector}, min(target_ttl, 600))
    client = await get_redis_client()
    if client is not None:
        try:
            json_bytes = json.dumps(vector.tolist()).encode("utf-8")
            async with client.pipeline(transaction=False) as pipe:
                for key in keys:
                    if key.startswith("xf:xfai:"):
                        try:
                            import phpserialize
                            pipe.set(key, phpserialize.dumps(vector.tolist()), ex=target_ttl)
                        except ImportError:
                            pass
                    else:
                        pipe.set(key, json_bytes, ex=target_ttl)
                await pipe.execute()
        except Exception as exc:
            await _redis_failed(client)
            logger.warning("Redis vector write failed: %s", exc)


def _valid_search_results(data: Any) -> bool:
    """Reject damaged cache records before promotion or permission lookups."""
    if not isinstance(data, dict):
        return False
    fields = ("faiss_results", "combined_results", "vector_results", "elasticsearch_results")
    primary = next((field for field in ("combined_results", "faiss_results", "elasticsearch_results")
                    if field in data), None)
    if "algorithm" in data and data["algorithm"] not in ("rrf", "weighted"):
        return False
    if "query" in data and not isinstance(data["query"], str):
        return False
    if "cached" in data and not isinstance(data["cached"], bool):
        return False
    if primary is not None and "count" not in data:
        return False
    if "count" in data and (type(data["count"]) is not int or data["count"] < 0):
        return False

    def finite_number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    if data.get("latency_ms") is not None and not finite_number(data["latency_ms"]):
        return False
    for field in fields:
        if field not in data:
            continue
        items = data[field]
        if items is None and field != primary:
            continue
        if not isinstance(items, list):
            return False
        for item in items:
            if not isinstance(item, dict):
                return False
            identities = [item.get("post_id"), item.get("thread_id")]
            if (not any(identities)
                    or any(value is not None and (type(value) is not int or value <= 0)
                           for value in identities)):
                return False
            if "type" in item and not isinstance(item["type"], str):
                return False
            if any(item.get(key) is not None and not isinstance(item[key], str)
                   for key in ("thread_title", "message", "username")):
                return False
            if item.get("post_date") is not None and type(item["post_date"]) is not int:
                return False
            if "score" in item and not finite_number(item["score"]):
                return False
            if any(item.get(key) is not None and not finite_number(item[key])
                   for key in ("rrf_score", "distance")):
                return False
            ranks = item.get("ranks")
            if ranks is not None and (not isinstance(ranks, dict)
                                      or any(not isinstance(key, str) or type(value) is not int
                                             for key, value in ranks.items())):
                return False
    return primary is None or data["count"] == len(data[primary])


async def get_cached_search_results(cache_key: str) -> Optional[Dict[str, Any]]:
    if not settings.CACHE_ENABLED:
        return None
    mem_key = f"res:{cache_key}"
    entry = _memory_get(mem_key)
    if entry is not None:
        if not _valid_search_results(entry["data"]):
            _local_memory_cache.pop(mem_key, None)
            return None
        return await _validate_search_results(entry["data"])
    client = await get_redis_client()
    if client is not None:
        try:
            key = f"xf_embed:search:{cache_key}"
            async with client.pipeline(transaction=False) as pipe:
                pipe.get(key)
                pipe.pttl(key)
                raw, remaining = await pipe.execute()
            ttl = _promotion_ttl(remaining, 60)
            if raw and ttl > 0:
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError, TypeError):
                    # A damaged entry is a miss, not a connection outage.
                    return None
                if _valid_search_results(data):
                    _memory_put(mem_key, {"data": data}, ttl)
                    # Keep database failures outside Redis error handling.
                    redis_data = data
                else:
                    redis_data = None
            else:
                redis_data = None
        except Exception as exc:
            await _redis_failed(client)
            logger.warning("Redis search read failed: %s", exc)
            redis_data = None
        if redis_data is not None:
            return await _validate_search_results(redis_data)
    return None


async def _validate_search_results(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Recheck current guest visibility/state on every cache hit in one batch.

    Post deletion/moderation and permission changes do not necessarily change
    search cache keys. Cached bodies are never an authority for access.
    """
    fields = ("faiss_results", "combined_results", "vector_results", "elasticsearch_results")
    items = {}

    def identity(item):
        if item.get("post_id"):
            return ("post", int(item["post_id"]))
        if item.get("thread_id"):
            return ("thread", int(item["thread_id"]))
        return None

    for field in fields:
        for item in data.get(field) or []:
            key = identity(item)
            if key is not None:
                items[key] = item
    if not items:
        return data
    from xf_embed.db import fetch_details_for_posts_and_threads, get_db_pool
    from xf_embed.visibility import VisibilityUnavailable
    try:
        pool = await get_db_pool()
    except Exception as exc:
        raise VisibilityUnavailable("Could not verify cached search permissions") from exc
    details = await fetch_details_for_posts_and_threads(pool, list(items.values()))
    permitted = {identity(item): item for item in details}
    validated = dict(data)
    content_fields = ("type", "post_id", "thread_id", "thread_title", "message", "post_date", "username", "user_id", "node_id")
    for field in fields:
        if data.get(field) is None:
            continue
        refreshed = []
        for item in data[field]:
            details = permitted.get(identity(item))
            if details is not None:
                refreshed.append({**item, **{key: details[key] for key in content_fields if key in details}})
        validated[field] = refreshed
    primary = next((field for field in ("combined_results", "faiss_results", "elasticsearch_results")
                    if field in validated), None)
    if primary is not None:
        if len(validated[primary]) != len(data[primary]):
            # Re-run retrieval to replace deleted/private candidates with the
            # next eligible hits rather than cache a shortened result page.
            return None
        validated["count"] = len(validated[primary])
    return validated


async def cache_search_results(cache_key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
    if not settings.CACHE_ENABLED or not _valid_search_results(data):
        return
    target_ttl = settings.SEARCH_RESULT_CACHE_TTL if ttl is None else ttl
    if target_ttl <= 0:
        return
    _memory_put(f"res:{cache_key}", {"data": data}, min(target_ttl, 60))
    client = await get_redis_client()
    if client is not None:
        try:
            await client.set(f"xf_embed:search:{cache_key}", json.dumps(data).encode("utf-8"), ex=target_ttl)
        except Exception as exc:
            await _redis_failed(client)
            logger.warning("Redis search write failed: %s", exc)
