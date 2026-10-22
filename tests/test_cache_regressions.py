import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from xf_embed import cache

real_get_redis_client = cache.get_redis_client
real_validate_search_results = cache._validate_search_results


class FakeRedis:
    def __init__(self, values):
        self.values = values
        self.commands = []
        self.round_trips = 0
        self.aclose = AsyncMock()

    def pipeline(self, transaction=False):
        self.commands = []
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    def get(self, key):
        self.commands.append(('get', key))
        return self

    def pttl(self, key):
        self.commands.append(('pttl', key))
        return self

    def set(self, key, value, ex=None):
        self.commands.append(('set', key, value, ex))
        return self

    async def execute(self):
        self.round_trips += 1
        return [self.values.get(command[1], (None, -2))[command[0] == 'pttl']
                if command[0] != 'set' else True for command in self.commands]


def test_redis_promotions_stay_bounded_and_preserve_hot_entries(monkeypatch):
    async def run():
        monkeypatch.setattr(cache, '_MEMORY_CACHE_MAX_SIZE', 3)
        redis = FakeRedis({f'xf_embed:search:{i}': (b'{"count":0}', 30000) for i in range(5)})
        monkeypatch.setattr(cache, 'get_redis_client', AsyncMock(return_value=redis))
        for i in range(3):
            await cache.get_cached_search_results(str(i))
        await cache.get_cached_search_results('0')
        await cache.get_cached_search_results('3')
        assert len(cache._local_memory_cache) == 3
        assert 'res:0' in cache._local_memory_cache
        assert 'res:1' not in cache._local_memory_cache
        await cache.cache_query_vector('query', np.ones(1536))
        assert len(cache._local_memory_cache) == 3
        assert 'res:0' in cache._local_memory_cache
    asyncio.run(run())


def test_promotion_respects_remaining_ttl_and_batches_reads(monkeypatch):
    async def run():
        clock = [100.0]
        monkeypatch.setattr(cache.time, 'monotonic', lambda: clock[0])
        h = cache.compute_query_hash('test')
        redis = FakeRedis({
            'xf_embed:search:test': (b'{"count":1}', 50),
            f'xfai:qemb:{h}': (b'null', 1000),
            f'xf_embed:qemb:{h}': (json.dumps([1.0] * 1536).encode(), 50),
        })
        monkeypatch.setattr(cache, 'get_redis_client', AsyncMock(return_value=redis))
        assert await cache.get_cached_search_results('test') == {'count': 1}
        assert (await cache.get_cached_query_vector('test')).shape == (1536,)
        assert redis.round_trips == 2
        clock[0] += .051
        assert cache._memory_get('res:test') is None
        assert cache._memory_get(cache._vector_cache_keys('test')[0]) is None
    asyncio.run(run())


def test_redis_client_initialization_is_single_flight_and_outages_back_off(monkeypatch):
    async def run():
        monkeypatch.setattr(cache, 'get_redis_client', real_get_redis_client)
        monkeypatch.setattr(cache, '_redis_retry_at', 0.0)
        monkeypatch.setattr(cache, '_redis_lock', asyncio.Lock())
        client = type('Client', (), {'ping': AsyncMock(return_value=True), 'aclose': AsyncMock()})()
        factory = AsyncMock()  # constructor is synchronous
        from unittest.mock import Mock
        factory = Mock(return_value=client)
        monkeypatch.setattr(cache.aioredis, 'Redis', factory)
        clients = await asyncio.gather(*(cache.get_redis_client() for _ in range(20)))
        assert all(item is client for item in clients)
        factory.assert_called_once()
        await cache.close_cache()
        client.ping.side_effect = OSError('offline')
        for _ in range(20):
            assert await cache.get_redis_client() is None
        assert factory.call_count == 2
        assert client.aclose.await_count == 2
    asyncio.run(run())


def test_cache_keys_include_effective_ranking_configuration(monkeypatch):
    key = cache.compute_search_cache_key('query', 'combined', 10)
    assert key == cache.compute_search_cache_key('QUERY ', 'combined', 10, cache.settings.HYBRID_ALGORITHM)
    monkeypatch.setattr(cache.settings, 'HYBRID_ALPHA', .123)
    assert key != cache.compute_search_cache_key('query', 'combined', 10)


def test_vector_cache_separates_same_dimension_models(monkeypatch):
    async def run():
        await cache.cache_query_vector('query', np.ones(1536))
        assert await cache.get_cached_query_vector('query') is not None
        monkeypatch.setattr(cache.settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        assert await cache.get_cached_query_vector('query') is None
        _, keys = cache._vector_cache_keys('query')
        assert len(keys) == 1  # Unqualified XenForo keys belong to its default model.
    asyncio.run(run())


def test_corrupt_search_entry_does_not_close_healthy_redis(monkeypatch):
    async def run():
        redis = FakeRedis({'xf_embed:search:test': (b'broken json', 30000)})
        monkeypatch.setattr(cache, 'get_redis_client', AsyncMock(return_value=redis))
        assert await cache.get_cached_search_results('test') is None
        redis.aclose.assert_not_awaited()
    asyncio.run(run())


def test_cached_search_rechecks_all_views_once_and_keeps_ranking(monkeypatch):
    from xf_embed import db
    async def run():
        payload = {'combined_results': [{'post_id': 1, 'message': 'stale', 'score': .8},
                                        {'post_id': 2, 'message': 'now hidden', 'score': .6}],
                   'vector_results': [{'post_id': 1, 'score': .9}],
                   'elasticsearch_results': [{'post_id': 2, 'score': 8}], 'count': 2}
        hydrate = AsyncMock(return_value=[{'post_id': 1, 'thread_id': 5, 'message': 'fresh'}])
        monkeypatch.setattr(db, 'get_db_pool', AsyncMock(return_value=object()))
        monkeypatch.setattr(db, 'fetch_details_for_posts_and_threads', hydrate)
        monkeypatch.setattr(cache, '_validate_search_results', real_validate_search_results)
        await cache.cache_search_results('test', payload)
        result = await cache.get_cached_search_results('test')
        hydrate.assert_awaited_once()
        assert len(hydrate.call_args.args[1]) == 2
        assert result['count'] == 1
        assert result['combined_results'] == [{'post_id': 1, 'thread_id': 5, 'message': 'fresh', 'score': .8}]
        assert result['vector_results'][0]['score'] == .9
        assert result['elasticsearch_results'] == []
        assert payload['count'] == 2
    asyncio.run(run())


@pytest.mark.parametrize('raw', [b'null', b'[[1,2]]', b'[NaN]', b'"text"'])
def test_invalid_cached_vectors_are_ignored(raw):
    assert cache._decode_vector(raw) is None
