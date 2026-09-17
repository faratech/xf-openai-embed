import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
import numpy as np
import redis.asyncio as aioredis

from xf_embed.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None
_local_memory_cache: Dict[str, Dict[str, Any]] = {}
_MEMORY_CACHE_MAX_SIZE = 2048


def compute_query_hash(query: str) -> str:
    """Computes MD5 hash of lowercased and stripped query string."""
    norm = query.strip().lower().encode("utf-8")
    return hashlib.md5(norm).hexdigest()


def compute_search_cache_key(query: str, mode: str, max_results: int, algorithm: Optional[str] = None) -> str:
    """Computes a unique cache key for a search request."""
    key_str = f"{query.strip().lower()}|{mode}|{max_results}|{algorithm or ''}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


async def get_redis_client() -> Optional[aioredis.Redis]:
    """Returns singleton Redis client if caching is enabled."""
    global _redis_client
    if not settings.CACHE_ENABLED:
        return None

    if _redis_client is None:
        try:
            _redis_client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
            )
            await _redis_client.ping()
            logger.info(f"Connected to Redis cache on {settings.REDIS_HOST}:{settings.REDIS_PORT} (DB {settings.REDIS_DB}).")
        except Exception as e:
            logger.warning(f"Could not connect to Redis ({e}). Using in-memory fallback cache.")
            _redis_client = None
    return _redis_client


async def close_cache() -> None:
    """Closes Redis client."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def get_cached_query_vector(query: str) -> Optional[np.ndarray]:
    """
    Retrieves cached query embedding vector.
    Tries Redis first (checking both xf:xfai:qemb: and xfai:qemb: prefixes),
    then falls back to in-memory cache.
    """
    if not settings.CACHE_ENABLED:
        return None

    h = compute_query_hash(query)
    now = time.time()

    # Check in-memory cache first (0.01ms)
    mem_key = f"qemb:{h}"
    if mem_key in _local_memory_cache:
        entry = _local_memory_cache[mem_key]
        if entry["expires_at"] > now:
            return entry["vector"]
        else:
            del _local_memory_cache[mem_key]

    # Check Redis
    r = await get_redis_client()
    if r is not None:
        try:
            keys_to_try = [f"xfai:qemb:{h}", f"xf:xfai:qemb:{h}", f"xf_embed:qemb:{h}"]
            raw_val = None
            for k in keys_to_try:
                raw_val = await r.get(k)
                if raw_val:
                    break

            if raw_val:
                vec = None
                try:
                    # Stored as JSON float array
                    val = json.loads(raw_val.decode("utf-8"))
                    vec = np.array(val, dtype=np.float32)
                except Exception:
                    try:
                        # Stored as PHP serialized format by XenForo Doctrine Cache
                        import phpserialize
                        php_data = phpserialize.loads(raw_val, array_hook=list)
                        if isinstance(php_data, list):
                            if php_data and isinstance(php_data[0], tuple):
                                vec = np.array([v for _, v in php_data], dtype=np.float32)
                            else:
                                vec = np.array(php_data, dtype=np.float32)
                        elif isinstance(php_data, dict):
                            vec = np.array([v for _, v in sorted(php_data.items())], dtype=np.float32)
                    except Exception:
                        vec = None

                if vec is not None and len(vec) == settings.OPENAI_DIMENSIONS:
                    # Populate in-memory cache for subsequent instant hits
                    _local_memory_cache[mem_key] = {"vector": vec, "expires_at": now + 600}
                    return vec
        except Exception as e:
            logger.warning(f"Redis get_cached_query_vector failed: {e}")

    return None


async def cache_query_vector(query: str, vector: np.ndarray, ttl: Optional[int] = None) -> None:
    """
    Caches query embedding vector in both in-memory and Redis caches.
    """
    if not settings.CACHE_ENABLED:
        return

    h = compute_query_hash(query)
    target_ttl = ttl or settings.QUERY_VECTOR_CACHE_TTL
    now = time.time()

    # Store in memory
    if len(_local_memory_cache) > _MEMORY_CACHE_MAX_SIZE:
        _local_memory_cache.clear()
    _local_memory_cache[f"qemb:{h}"] = {"vector": vector, "expires_at": now + min(target_ttl, 600)}

    # Store in Redis
    r = await get_redis_client()
    if r is not None:
        try:
            json_bytes = json.dumps(vector.tolist()).encode("utf-8")
            # Store under xf_embed and xfai:qemb namespace for mutual sharing
            await r.set(f"xf_embed:qemb:{h}", json_bytes, ex=target_ttl)
            await r.set(f"xfai:qemb:{h}", json_bytes, ex=target_ttl)

            # Also serialize in PHP format for direct XenForo Doctrine Cache sharing
            try:
                import phpserialize
                php_bytes = phpserialize.dumps(vector.tolist())
                await r.set(f"xf:xfai:qemb:{h}", php_bytes, ex=target_ttl)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Redis cache_query_vector failed: {e}")


async def get_cached_search_results(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached search results from Redis or in-memory."""
    if not settings.CACHE_ENABLED:
        return None

    now = time.time()
    mem_key = f"res:{cache_key}"
    if mem_key in _local_memory_cache:
        entry = _local_memory_cache[mem_key]
        if entry["expires_at"] > now:
            return entry["data"]
        else:
            del _local_memory_cache[mem_key]

    r = await get_redis_client()
    if r is not None:
        try:
            redis_key = f"xf_embed:search:{cache_key}"
            raw = await r.get(redis_key)
            if raw:
                data = json.loads(raw.decode("utf-8"))
                _local_memory_cache[mem_key] = {"data": data, "expires_at": now + 60}
                return data
        except Exception as e:
            logger.warning(f"Redis get_cached_search_results failed: {e}")

    return None


async def cache_search_results(cache_key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
    """Caches search results in Redis and in-memory."""
    if not settings.CACHE_ENABLED:
        return

    target_ttl = ttl or settings.SEARCH_RESULT_CACHE_TTL
    now = time.time()

    mem_key = f"res:{cache_key}"
    if len(_local_memory_cache) > _MEMORY_CACHE_MAX_SIZE:
        _local_memory_cache.clear()
    _local_memory_cache[mem_key] = {"data": data, "expires_at": now + min(target_ttl, 60)}

    r = await get_redis_client()
    if r is not None:
        try:
            redis_key = f"xf_embed:search:{cache_key}"
            encoded = json.dumps(data).encode("utf-8")
            await r.set(redis_key, encoded, ex=target_ttl)
        except Exception as e:
            logger.warning(f"Redis cache_search_results failed: {e}")
