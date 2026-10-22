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


def test_search_replenishes_rejected_pages_without_rehydrating_candidates(monkeypatch):
    import copy

    calls = []
    hydrations = []

    def hit(identifier):
        return {"_id": f"post-{identifier}", "_score": 100 - identifier,
                "_source": {"type": "post", "discussion_id": 42},
                "sort": [100 - identifier, 1700000000, identifier]}

    async def search(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        body = kwargs["body"]
        if "pit" not in body:
            return {"hits": {"hits": [hit(1), hit(2)]}}
        start = body.get("search_after", [0, 0, 0])[-1]
        rows = [hit(i) for i in range(start + 1, min(start + body["size"], 24) + 1)]
        return {"pit_id": f"snapshot-{start}", "hits": {"hits": rows}}

    async def hydrate(pool, items, visible_node_ids):
        hydrations.append([item["post_id"] for item in items])
        return [{**item, "thread_title": "current"} for item in items if item["post_id"] > 20]

    client = type("FakeES", (), {})()
    client.search = search
    client.open_point_in_time = AsyncMock(return_value={"id": "snapshot"})
    client.close_point_in_time = AsyncMock()
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    monkeypatch.setattr(elastic_engine, "fetch_details_for_posts_and_threads", hydrate)
    results = asyncio.run(elastic_engine.search_elasticsearch("query", max_results=2))
    assert [item["post_id"] for item in results] == [21, 22]
    assert [item["es_score"] for item in results] == [79, 78]
    assert hydrations == [[1, 2], list(range(3, 21)), [21, 22, 23, 24]]
    assert len(calls) == 3
    assert "index" not in calls[1] and "index" not in calls[2]
    assert calls[2]["body"]["search_after"] == [80, 1700000000, 20]
    assert calls[2]["body"]["pit"]["id"] == "snapshot-0"
    assert all(call["body"]["track_total_hits"] is False for call in calls)
    client.close_point_in_time.assert_awaited_once_with(id="snapshot-20")


def test_full_valid_first_page_does_not_open_a_snapshot(monkeypatch):
    client = type("FakeES", (), {})()
    client.search = AsyncMock(return_value={"hits": {"hits": [
        {"_id": "post-1", "_score": 5, "_source": {"type": "post"}},
    ]}})
    client.open_point_in_time = AsyncMock(side_effect=AssertionError("unnecessary snapshot"))
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    async def hydrate(pool, items, visible_node_ids):
        return items
    monkeypatch.setattr(elastic_engine, "fetch_details_for_posts_and_threads", hydrate)
    assert len(asyncio.run(elastic_engine.search_elasticsearch("query", max_results=1))) == 1
    client.search.assert_awaited_once()
    client.open_point_in_time.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("hydration failed"), asyncio.CancelledError()])
def test_search_snapshot_closes_after_failure_or_cancellation(monkeypatch, failure):
    hit = {"_id": "post-1", "_score": 5, "_source": {"type": "post"}}
    client = type("FakeES", (), {})()
    client.search = AsyncMock(side_effect=[
        {"hits": {"hits": [hit]}},
        {"pit_id": "latest", "hits": {"hits": [{**hit, "_id": "post-2"}]}},
    ])
    client.open_point_in_time = AsyncMock(return_value={"id": "snapshot"})
    client.close_point_in_time = AsyncMock()
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    monkeypatch.setattr(elastic_engine, "fetch_details_for_posts_and_threads",
                        AsyncMock(side_effect=[[], failure]))
    with pytest.raises(type(failure)):
        asyncio.run(elastic_engine.search_elasticsearch("query", max_results=1))
    client.close_point_in_time.assert_awaited_once_with(id="latest")


def test_repeated_search_after_cursor_fails_without_looping(monkeypatch):
    hit = {"_id": "post-1", "_score": 5, "_source": {"type": "post"}, "sort": [5, 1, 1]}
    client = type("FakeES", (), {})()
    client.search = AsyncMock(side_effect=[
        {"hits": {"hits": [hit]}},
        {"hits": {"hits": [hit] * 20}},
        {"hits": {"hits": [hit] * 20}},
    ])
    client.open_point_in_time = AsyncMock(return_value={"id": "snapshot"})
    client.close_point_in_time = AsyncMock()
    hydrate = AsyncMock(return_value=[])
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    monkeypatch.setattr(elastic_engine, "fetch_details_for_posts_and_threads", hydrate)
    with pytest.raises(RuntimeError, match="pagination did not advance"):
        asyncio.run(elastic_engine.search_elasticsearch("query", max_results=1))
    assert client.search.await_count == 3
    hydrate.assert_awaited_once()
    client.close_point_in_time.assert_awaited_once_with(id="snapshot")


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_search_rejects_nonpositive_limits_before_backend_work(monkeypatch, limit):
    pool = AsyncMock(side_effect=AssertionError("invalid request launched work"))
    monkeypatch.setattr(elastic_engine, "get_db_pool", pool)
    with pytest.raises(ValueError, match="max_results"):
        asyncio.run(elastic_engine.search_elasticsearch("query", max_results=limit))
    pool.assert_not_called()


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
