import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def bulk(monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'embed-elastic.py'
    spec = importlib.util.spec_from_file_location('bulk_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_EMBEDDING_LENGTH = 2
    return module


class Pool:
    def acquire(self):
        return self

    def cursor(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, *args):
        pass

    async def fetchone(self):
        return (2,)


@pytest.mark.parametrize('invalid', [b'bad', '[{},{}]', '[[1],[2]]', '[0,0]'])
def test_bulk_skips_malformed_vectors_and_keeps_processing(bulk, monkeypatch, invalid):
    sent = []

    async def fetch(pool, last_id):
        if last_id:
            return []
        return [(1, 1, None, invalid, 2), (2, 2, None, np.array([1, 2], dtype=np.float32).tobytes(), 2)]

    async def send(**kwargs):
        sent.extend(kwargs['actions'])
        return len(kwargs['actions']), []

    monkeypatch.setattr(bulk, 'fetch_data_from_mysql', fetch)
    monkeypatch.setattr(bulk, 'async_bulk', send)
    asyncio.run(bulk.process_data(Pool(), SimpleNamespace()))
    assert [item['_id'] for item in sent] == ['post-2']


def test_bulk_failure_does_not_claim_success(bulk, monkeypatch):
    async def fetch(pool, last_id):
        return [] if last_id else [(1, 1, None, np.array([1, 2], dtype=np.float32).tobytes(), 2)]

    async def send(**kwargs):
        return 0, [{'index': {'status': 400}}]

    monkeypatch.setattr(bulk, 'fetch_data_from_mysql', fetch)
    monkeypatch.setattr(bulk, 'async_bulk', send)
    with pytest.raises(RuntimeError, match='index'):
        asyncio.run(bulk.process_data(Pool(), SimpleNamespace()))
