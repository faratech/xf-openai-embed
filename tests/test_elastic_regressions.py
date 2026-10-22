"""Elasticsearch compatibility and bounded bulk processing without network I/O."""

import asyncio
from typing import get_type_hints
from unittest.mock import AsyncMock

import pytest

from xf_embed import elastic_engine


@pytest.fixture(autouse=True)
def public_search_fakes(monkeypatch):
    monkeypatch.setattr(elastic_engine, "get_db_pool", AsyncMock(return_value=object()), raising=False)
    monkeypatch.setattr(elastic_engine, "guest_visible_node_ids", AsyncMock(return_value=frozenset({4})), raising=False)


def test_bulk_type_annotations_resolve_on_supported_python_versions():
    assert get_type_hints(elastic_engine.bulk_index_documents)["return"] is not None


def test_empty_bulk_skips_client_and_helper(monkeypatch):
    def unexpected_client():
        raise AssertionError("empty bulk must not initialize a client")

    monkeypatch.setattr(elastic_engine, "get_es_client", unexpected_client)
    assert asyncio.run(elastic_engine.bulk_index_documents([])) == (0, 0)


def test_bulk_streams_actions_in_bounded_chunks_and_counts_failures(monkeypatch):
    client = object()
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    documents = [{"post_id": i, "message": "text"} for i in range(1201)]

    async def fake_bulk(**kwargs):
        assert kwargs["client"] is client
        assert kwargs["chunk_size"] == 500
        assert kwargs["raise_on_error"] is False
        actions = kwargs["actions"]
        assert not isinstance(actions, list)
        collected = list(actions)
        assert len(collected) == len(documents)
        assert collected[0] == {"_index": "test-index", "_id": "post-0", "_source": documents[0]}
        return 1200, [{"index": {"status": 400}}]

    monkeypatch.setattr(elastic_engine, "async_bulk", fake_bulk)
    assert asyncio.run(elastic_engine.bulk_index_documents(documents, "test-index")) == (1200, 1)


def test_search_backend_failure_is_not_reported_as_empty_success(monkeypatch):
    search = AsyncMock(side_effect=RuntimeError("elasticsearch unavailable"))
    client = type("FakeES", (), {"search": search})()
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    with pytest.raises(RuntimeError, match="elasticsearch unavailable"):
        asyncio.run(elastic_engine.search_elasticsearch("query"))


def test_bm25_filters_to_public_forum_types_nodes_and_visible_documents():
    dsl = elastic_engine.build_bm25_query_dsl("query", visible_node_ids=frozenset({4, 84}))
    filtered_query = dsl["query"]["function_score"]["query"]["bool"]
    assert {"terms": {"type": ["post", "thread"]}} in filtered_query["filter"]
    assert {"terms": {"node": [4, 84]}} in filtered_query["filter"]
    assert {"term": {"hidden": True}} in filtered_query["must_not"]


def test_search_hydrates_forum_hits_and_drops_nonforum_or_stale_candidates(monkeypatch):
    response = {"hits": {"hits": [
        {"_id": "conversation_message-99", "_score": 8, "_source": {
            "type": "conversation_message", "discussion_id": 99, "message": "private"}},
        {"_id": "post-7", "_score": 5, "_source": {
            "type": "post", "discussion_id": 42, "message": "deleted post"}},
        {"_id": "thread-42", "_score": 4, "_source": {
            "type": "thread", "discussion_id": 42, "title": "stale title"}},
    ]}}
    search = AsyncMock(return_value=response)
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: type("FakeES", (), {"search": search})())

    async def hydrate(pool, items, visible_node_ids=None):
        assert visible_node_ids == frozenset({4})
        assert [(item["type"], item["post_id"], item["thread_id"]) for item in items] == [
            ("post", 7, 42), ("thread", None, 42),
        ]
        return [{**items[1], "thread_title": "current title", "message": "", "username": "current user"}]

    monkeypatch.setattr(elastic_engine, "fetch_details_for_posts_and_threads", hydrate, raising=False)
    results = asyncio.run(elastic_engine.search_elasticsearch("query"))
    assert len(results) == 1
    assert results[0]["thread_title"] == "current title"
    assert results[0]["score"] == results[0]["es_score"] == 4
    assert results[0]["username"] == "current user"


def test_search_with_no_public_nodes_skips_elasticsearch(monkeypatch):
    monkeypatch.setattr(elastic_engine, "guest_visible_node_ids", AsyncMock(return_value=frozenset()))
    search = AsyncMock(return_value={"hits": {"hits": []}})
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: type("FakeES", (), {"search": search})())
    assert asyncio.run(elastic_engine.search_elasticsearch("query")) == []
    search.assert_not_called()


def test_bulk_derives_typed_identity_from_thread_and_nested_source_documents(monkeypatch):
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: object())

    async def fake_bulk(**kwargs):
        actions = list(kwargs["actions"])
        assert [action["_id"] for action in actions] == ["thread-7", "thread-8", "post-9", "explicit"]
        return len(actions), []

    monkeypatch.setattr(elastic_engine, "async_bulk", fake_bulk)
    documents = [
        {"thread_id": 7}, {"thread_id": 8}, {"source": {"post_id": 9}},
        {"_id": "explicit", "source": {"thread_id": 10}},
    ]
    assert asyncio.run(elastic_engine.bulk_index_documents(documents)) == (4, 0)


def test_bulk_rejects_missing_identity_before_sending_any_documents(monkeypatch):
    bulk = AsyncMock()
    monkeypatch.setattr(elastic_engine, "async_bulk", bulk)
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(elastic_engine.bulk_index_documents([{"post_id": 1}, {"message": "no id"}]))
    bulk.assert_not_called()
