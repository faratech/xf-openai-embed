import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from xf_embed import embeddings


def test_literal_special_tokens_are_ordinary_user_text():
    text = 'literal <|endoftext|> from a forum post'
    assert embeddings.truncate_text(text) == text
    assert embeddings.count_tokens(text) > 0


@pytest.mark.parametrize('vector', [[float('nan')] * 1536, [[1]] * 1536, [0] * 1536])
def test_malformed_local_vector_falls_back(vector, monkeypatch):
    response = SimpleNamespace(status_code=200, json=lambda: {'embedding': vector})
    client = SimpleNamespace(post=AsyncMock(return_value=response))
    monkeypatch.setattr(embeddings, 'get_httpx_client', lambda: client)
    assert asyncio.run(embeddings._generate_via_local_service('query')) is None


def test_custom_model_skips_incompatible_local_model(monkeypatch):
    async def run():
        monkeypatch.setattr(embeddings.settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        local = AsyncMock()
        direct = AsyncMock(return_value=np.ones(1536))
        monkeypatch.setattr(embeddings, '_generate_via_local_service', local)
        monkeypatch.setattr(embeddings, '_generate_via_openai_api', direct)
        await embeddings.generate_embedding('query', use_cache=False)
        local.assert_not_awaited()
        direct.assert_awaited_once()
    asyncio.run(run())


def test_identical_embedding_misses_share_work_and_cancellation(monkeypatch):
    async def run():
        monkeypatch.setattr(embeddings, 'get_cached_query_vector', AsyncMock(return_value=None))
        monkeypatch.setattr(embeddings, 'cache_query_vector', AsyncMock())
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def generate(text):
            calls.append(text)
            started.set()
            await release.wait()
            return np.ones(1536, dtype=np.float32)
        monkeypatch.setattr(embeddings, '_generate_via_local_service', generate)
        tasks = [asyncio.create_task(embeddings.generate_embedding('cold query', normalize=False)) for _ in range(12)]
        await started.wait()
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        release.set()
        results = await asyncio.gather(*tasks[1:])
        assert len(calls) == 1
        assert all(np.array_equal(vector, np.ones(1536)) for vector in results)
        embeddings.cache_query_vector.assert_awaited_once()
        assert not embeddings._inflight
    asyncio.run(run())


def test_failed_embedding_can_retry_and_uncached_requests_stay_uncached(monkeypatch):
    async def run():
        monkeypatch.setattr(embeddings, 'cache_query_vector', AsyncMock())
        mock = AsyncMock(side_effect=[RuntimeError('fail'), np.ones(1536)])
        monkeypatch.setattr(embeddings, '_generate_via_local_service', mock)
        with pytest.raises(RuntimeError):
            await embeddings.generate_embedding('query', use_cache=False)
        await embeddings.generate_embedding('query', use_cache=False)
        assert mock.await_count == 2
        embeddings.cache_query_vector.assert_not_awaited()
        assert not embeddings._inflight
    asyncio.run(run())


def test_batch_limits_and_response_indices(monkeypatch):
    async def run():
        monkeypatch.setattr(embeddings.settings, 'OPENAI_DIMENSIONS', 2)
        monkeypatch.setattr(embeddings, 'MAX_BATCH_TOKENS', 5)
        monkeypatch.setattr(embeddings, 'MAX_BATCH_INPUTS', 2)
        monkeypatch.setattr(embeddings, 'MAX_TOKENS', 4)
        monkeypatch.setattr(embeddings, 'get_tokenizer', lambda: SimpleNamespace(encode=lambda text, **kw: [int(text)] * 3))
        calls = []
        async def create(**kwargs):
            inputs = kwargs['input']
            calls.append(inputs)
            assert sum(map(len, inputs)) <= 5
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[tokens[0], 1])
                                         for i, tokens in reversed(list(enumerate(inputs)))])
        monkeypatch.setattr(embeddings, 'get_openai_client', lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
        monkeypatch.setattr(embeddings, 'direct_metering', None)
        result = await embeddings.generate_embeddings_batch(['1', '2', '3'], normalize=False)
        assert result.tolist() == [[1, 1], [2, 1], [3, 1]]
        assert len(calls) == 3
    asyncio.run(run())


def test_malformed_response_does_not_repeat_paid_request(monkeypatch):
    async def run():
        client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(data=[]))))
        monkeypatch.setattr(embeddings, 'get_openai_client', lambda: client)
        monkeypatch.setattr(embeddings, 'direct_metering', None)
        with pytest.raises(ValueError, match='Incomplete'):
            await embeddings._generate_via_openai_api('query')
        client.embeddings.create.assert_awaited_once()
    asyncio.run(run())


def test_metering_failure_does_not_repeat_successful_generation(monkeypatch):
    async def run():
        monkeypatch.setattr(embeddings.settings, 'OPENAI_DIMENSIONS', 2)
        response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[3, 4])])
        create = AsyncMock(return_value=response)
        monkeypatch.setattr(embeddings, 'get_openai_client', lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
        monkeypatch.setattr(embeddings, 'direct_metering', SimpleNamespace(meter_embedding=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('ledger'))))
        assert (await embeddings._generate_via_openai_api('query')).tolist() == [3, 4]
        create.assert_awaited_once()
    asyncio.run(run())


def test_sdk_retries_do_not_multiply_outer_attempts(monkeypatch):
    import httpx
    from openai import AsyncOpenAI, RateLimitError
    from tenacity import stop_after_attempt, wait_none

    async def run():
        calls = []
        def reject(request):
            calls.append(request)
            return httpx.Response(429, json={'error': {'message': 'rate limited'}})
        transport = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        real_client = AsyncOpenAI
        monkeypatch.setattr(embeddings, '_openai_client', None)
        monkeypatch.setattr(embeddings, 'AsyncOpenAI', lambda **kwargs: real_client(api_key='test', http_client=transport,
                                                                                 max_retries=kwargs['max_retries']))
        monkeypatch.setattr(embeddings._generate_via_openai_api.retry, 'stop', stop_after_attempt(2))
        monkeypatch.setattr(embeddings._generate_via_openai_api.retry, 'wait', wait_none())
        try:
            with pytest.raises(RateLimitError):
                await embeddings._generate_via_openai_api('query')
            assert len(calls) == 2
            assert embeddings.get_openai_client().max_retries == 0
        finally:
            await embeddings.close_embedding_clients()
    asyncio.run(run())
