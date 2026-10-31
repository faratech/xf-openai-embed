"""Keep the unit suite isolated from this checkout's production services."""

import socket
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def offline_tokenizer(monkeypatch):
    import tiktoken
    from xf_embed import embeddings

    # Exercise real encode/decode and special-token handling without a BPE download.
    tokenizer = tiktoken.Encoding(
        name="unit-test-bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    monkeypatch.setattr(embeddings, "get_tokenizer", lambda: tokenizer)
    return tokenizer


@pytest.fixture(autouse=True)
def isolate_unit_services(monkeypatch):
    from xf_embed import api, cache

    original_connect = socket.socket.connect

    def offline_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"Unit tests must stub network access: {address!r}")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", offline_connect)
    monkeypatch.setattr(api, "get_db_pool", AsyncMock(side_effect=RuntimeError("offline unit test")))
    monkeypatch.setattr(api, "get_faiss_index", AsyncMock(return_value=None))
    monkeypatch.setattr(api, "get_es_client", lambda: type("OfflineES", (), {"ping": AsyncMock(return_value=False)})())
    monkeypatch.setattr(api, "get_redis_client", AsyncMock(return_value=None))
    monkeypatch.setattr(cache, "get_redis_client", AsyncMock(return_value=None))
    async def validated_fake(data):
        return data
    monkeypatch.setattr(cache, "_validate_search_results", validated_fake)
    monkeypatch.setattr(cache, "_redis_client", None)
    cache._local_memory_cache.clear()
    yield
    cache._local_memory_cache.clear()
