import asyncio
import numpy as np
from xf_embed.cache import (
    compute_query_hash,
    compute_search_cache_key,
    cache_query_vector,
    get_cached_query_vector,
    cache_search_results,
    get_cached_search_results,
)


def test_query_hashing():
    h1 = compute_query_hash("Blue Screen Error")
    h2 = compute_query_hash("  blue screen error  ")
    assert h1 == h2
    assert len(h1) == 32


def test_search_cache_key():
    k1 = compute_search_cache_key("test", "combined", 10, "rrf")
    k2 = compute_search_cache_key("TEST ", "combined", 10, "rrf")
    assert k1 == k2


def test_in_memory_vector_caching():
    async def run():
        test_query = "pytest_test_query_string_123"
        vec = np.ones(1536, dtype=np.float32)

        await cache_query_vector(test_query, vec, ttl=60)
        cached = await get_cached_query_vector(test_query)

        assert cached is not None
        assert len(cached) == 1536
        assert np.allclose(cached, vec)

    asyncio.run(run())


def test_in_memory_search_result_caching():
    async def run():
        cache_key = "pytest_result_key_456"
        test_payload = {"count": 1, "query": "test", "combined_results": [{"post_id": 999}]}

        await cache_search_results(cache_key, test_payload, ttl=60)
        cached = await get_cached_search_results(cache_key)

        assert cached is not None
        assert cached["count"] == 1
        assert cached["combined_results"][0]["post_id"] == 999

    asyncio.run(run())
