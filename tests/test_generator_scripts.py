"""Offline regression coverage for the legacy embedding maintenance scripts."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI as SDKAsyncOpenAI


class Tokenizer:
    def __init__(self):
        self.calls = 0

    def encode(self, text, **kwargs):
        self.calls += 1
        return [ord(char) for char in text]

    def decode(self, tokens):
        return ''.join(chr(token) for token in tokens)


@pytest.fixture(params=['post', 'thread'])
def generator(request, monkeypatch):
    import openai
    import tiktoken
    import xf_embed.config
    client = SimpleNamespace(embeddings=SimpleNamespace(create=None))
    monkeypatch.setattr(openai, 'AsyncOpenAI', lambda **kwargs: setattr(client, 'init_kwargs', kwargs) or client)
    tokenizer = Tokenizer()
    monkeypatch.setattr(tiktoken, 'encoding_for_model', lambda _: tokenizer)
    script = Path(__file__).resolve().parents[1] / f'embed_generate_xf_{request.param}.py'
    spec = importlib.util.spec_from_file_location(f'generator_{request.param}', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module.get_embedding_batch, 'retry'):
        from tenacity import stop_after_attempt
        module.get_embedding_batch.retry.stop = stop_after_attempt(1)
    module.entity = request.param
    return module, client, tokenizer


def test_batch_respects_aggregate_and_per_input_limits(generator):
    module, client, tokenizer = generator
    requests = []

    async def create(*, input, model):
        lengths = [len(value) for value in input]
        assert max(lengths) <= module.MAX_TOKENS
        assert sum(lengths) <= 300_000
        assert len(input) <= 2048
        requests.append(lengths)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i + 1)] * 1536)
                                     for i in range(len(input))])

    client.embeddings.create = create
    values = ['a' * 9000] * 50
    embeddings = asyncio.run(module.get_embedding_batch(values))
    assert len(embeddings) == len(values)
    assert len(requests) == 2
    assert tokenizer.calls == len(values)


def test_response_indexes_preserve_input_order(generator):
    module, client, _ = generator

    async def create(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[2.0] * 1536),
                                     SimpleNamespace(index=0, embedding=[1.0] * 1536)])

    client.embeddings.create = create
    result = asyncio.run(module.get_embedding_batch(['one', 'two']))
    assert result[0][0] == 1.0
    assert result[1][0] == 2.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'bad', True, 0.0])
def test_validate_rejects_invalid_coordinates(generator, bad):
    module, _, _ = generator
    assert module.validate_embedding([bad] * 1536)[0] is False


class Pool:
    def __init__(self, rows=()):
        self.rows = rows
        self.queries = []
        self.commits = 0
        self.closed = False

    def acquire(self):
        return self

    def cursor(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, query, params=None):
        self.queries.append((' '.join(query.split()), params))

    async def fetchone(self):
        return self.rows[0]

    async def fetchall(self):
        return self.rows

    async def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True

    async def wait_closed(self):
        assert self.closed


def test_post_rebuild_preserves_threads(generator, monkeypatch):
    module, _, _ = generator
    if module.entity != 'post':
        return
    pool = Pool()

    async def process(pool):
        pass

    monkeypatch.setattr(module, 'process_posts', process)
    asyncio.run(module.rebuild_database(pool))
    assert 'WHERE post_id IS NOT NULL' in pool.queries[0][0]


def test_generator_main_reports_failures_and_closes_pool(generator, monkeypatch):
    module, _, _ = generator
    pool = Pool()

    async def get_pool():
        return pool

    async def fail(pool):
        raise RuntimeError('batch failed')

    monkeypatch.setattr(module, 'get_db_pool', get_pool)
    monkeypatch.setattr(module, f'process_{module.entity}s', fail)
    args = SimpleNamespace(check=False, repair=False, rebuild=False, post=None, thread=None)
    with pytest.raises(RuntimeError, match='batch failed'):
        asyncio.run(module.main(args))
    assert pool.closed


@pytest.mark.parametrize('repair', [False, True])
def test_batch_skips_empty_text_preserves_truncation_and_tokenizes_once(generator, repair):
    module, client, tokenizer = generator
    requests = []

    async def create(*, input, model):
        requests.append(input)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0] * 1536)
                                     for i in range(len(input))])

    client.embeddings.create = create
    field = 'message' if module.entity == 'post' else 'title'
    batch = [{f'{module.entity}_id': 1, field: ''},
             {f'{module.entity}_id': 2, field: 'a' * 9000}]
    pool = Pool()
    func = module.process_repair_batch_with_semaphore if repair else module.process_batch_with_semaphore
    result = asyncio.run(func(pool, batch))
    assert result == 1
    assert tokenizer.calls == 2
    assert len(requests) == 1
    query, params = pool.queries[0]
    assert params[2 if repair else 3] is True
    assert pool.commits == 1


def test_malformed_response_does_not_retry_or_write(generator):
    module, client, _ = generator
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[])

    client.embeddings.create = create
    field = 'message' if module.entity == 'post' else 'title'
    pool = Pool()
    with pytest.raises(ValueError, match='response indexes'):
        asyncio.run(module.process_batch_with_semaphore(pool, [{f'{module.entity}_id': 1, field: 'text'}]))
    assert len(calls) == 1
    assert pool.queries == []
    assert pool.commits == 0


@pytest.mark.parametrize('repair', [False, True])
def test_check_and_repair_use_scoped_keyset_pages(generator, monkeypatch, repair):
    module, _, _ = generator
    pool = Pool(rows=[(2,)])
    page_calls = []
    valid = [1.0] * 1536

    class Pages(Pool):
        async def fetchall(self):
            query, params = self.queries[-1]
            assert f'WHERE e.{module.entity}_id IS NOT NULL' in query
            assert 'e.id > %s ORDER BY e.id LIMIT %s' in query
            page_calls.append(params[0])
            rows = [(2, 11, valid, 1536), (9, 12, None, 0)] if params[0] == 0 else []
            return [row + ('unchanged long content',) for row in rows] if repair else rows

    pool = Pages(rows=[(2,)])
    repaired = []

    async def repair_batch(pool, batch):
        repaired.extend(batch)
        return len(batch)

    monkeypatch.setattr(module, 'process_repair_batch_with_semaphore', repair_batch)
    result = asyncio.run(module.repair_embeddings(pool) if repair else module.check_embeddings(pool))
    assert result == 1
    assert page_calls == [0, 9]
    if repair:
        assert repaired == [{f'{module.entity}_id': 12,
                             'message' if module.entity == 'post' else 'title': 'unchanged long content'}]


def test_rebuild_only_deletes_owned_entity_rows(generator, monkeypatch):
    import sqlite3
    module, _, _ = generator
    conn = sqlite3.connect(':memory:')
    conn.executescript('''
        CREATE TABLE openai_embeddings(id, post_id, thread_id, section);
        INSERT INTO openai_embeddings VALUES (1, 101, 7, 0), (2, NULL, 7, 0), (3, 102, NULL, 0), (4, 101, 7, 1);
    ''')

    class SqlPool(Pool):
        async def execute(self, query, params=None):
            conn.execute(query)

    async def process(pool):
        pass

    monkeypatch.setattr(module, f'process_{module.entity}s', process)
    asyncio.run(module.rebuild_database(SqlPool()))
    expected = [(2,), (4,)] if module.entity == 'post' else [(1,), (3,), (4,)]
    assert conn.execute('SELECT id FROM openai_embeddings ORDER BY id').fetchall() == expected


def test_thread_discovery_does_not_confuse_post_vector_for_title_vector(generator):
    import sqlite3
    module, _, _ = generator
    if module.entity != 'thread':
        return
    conn = sqlite3.connect(':memory:')
    conn.executescript('''
        CREATE TABLE xf_thread(thread_id, title);
        CREATE TABLE openai_embeddings(post_id, thread_id);
        INSERT INTO xf_thread VALUES (7, 'A thread');
        INSERT INTO openai_embeddings VALUES (101, 7);
    ''')

    class SqlPool(Pool):
        async def execute(self, query, params=None):
            self.cursor_result = conn.execute(query.replace('%s', '?'), params)

        async def fetchall(self):
            return self.cursor_result.fetchall()

    rows, last_id = asyncio.run(module.fetch_threads(SqlPool(), 0, 100))
    assert rows == [{'thread_id': 7, 'title': 'A thread'}]
    assert last_id == 7


@pytest.mark.parametrize('status,expected_attempts', [(429, 2), (401, 1)])
def test_retry_policy_has_one_layer_at_the_http_transport(generator, monkeypatch, status, expected_attempts):
    import httpx
    from openai import APIStatusError
    from tenacity import stop_after_attempt, wait_none
    module, client, _ = generator
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={'error': {'message': 'offline failure', 'type': 'test_error'}})

    async def scenario():
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        # Reuse actual module-constructor options so enabling SDK retry layers
        # regresses transport counts, rather than merely a source assertion.
        sdk = SDKAsyncOpenAI(**client.init_kwargs, http_client=http_client)
        monkeypatch.setattr(module, 'openai', sdk)
        monkeypatch.setattr(module._request_embeddings.retry, 'stop', stop_after_attempt(2))
        monkeypatch.setattr(module._request_embeddings.retry, 'wait', wait_none())
        try:
            with pytest.raises(APIStatusError):
                await module.get_embedding_batch(['text'])
        finally:
            await sdk.close()

    asyncio.run(scenario())
    assert len(calls) == expected_attempts
