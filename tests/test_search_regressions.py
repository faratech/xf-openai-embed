"""Search regressions use fakes only; no embedding or search service is called."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from xf_embed import api, hybrid_search
from xf_embed.hybrid_search import reciprocal_rank_fusion, weighted_score_fusion
from xf_embed.visibility import VisibilityUnavailable


def test_rrf_credits_an_item_once_per_source_and_preserves_best_rank():
    fused = reciprocal_rank_fusion([
        [{"post_id": 1}, {"post_id": 1, "message": "hydrated"}],
        [{"post_id": 2}, {"post_id": 1}],
    ])
    first = fused[0]
    assert first["post_id"] == 1
    assert first["rrf_score"] == round(1 / 61 + 1 / 62, 6)
    assert first["ranks"] == {"vector": 1, "elastic": 2}
    assert first["message"] == "hydrated"


def test_weighted_singletons_keep_their_source_weight():
    fused = weighted_score_fusion(
        [{"post_id": 1, "score": 0.9}],
        [{"post_id": 2, "es_score": 5.0}],
        alpha=0.6,
    )
    assert [(item["post_id"], item["score"]) for item in fused] == [(1, 0.6), (2, 0.4)]


def test_weighted_duplicates_keep_the_best_source_score_and_merge_details():
    fused = weighted_score_fusion(
        [{"post_id": 1, "score": 0.9}, {"post_id": 2, "score": 0.5},
         {"post_id": 1, "score": 0.1}],
        [{"post_id": 1, "es_score": 5.0, "message": "hydrated"}],
    )
    assert fused[0]["post_id"] == 1
    assert fused[0]["vector_norm_score"] == 1.0
    assert fused[0]["message"] == "hydrated"


def test_weighted_equal_scores_keep_first_seen_source_order():
    fused = weighted_score_fusion(
        [{"post_id": 10, "score": 0.9}, {"post_id": 2, "score": 0.9}],
        [{"post_id": 30, "es_score": 5.0}],
        alpha=0.5,
    )
    assert [item["post_id"] for item in fused] == [10, 2, 30]


def test_failed_hybrid_backend_cancels_and_awaits_its_sibling(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def failing_backend(*args, **kwargs):
            await started.wait()
            raise RuntimeError("backend failed")

        async def sibling_backend(*args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                stopped.set()

        monkeypatch.setattr(hybrid_search, "perform_vector_search", failing_backend)
        monkeypatch.setattr(hybrid_search, "search_elasticsearch", sibling_backend)
        with pytest.raises(RuntimeError, match="backend failed"):
            await hybrid_search.perform_hybrid_search("query")
        assert stopped.is_set()

    asyncio.run(scenario())


def test_cancelled_hybrid_request_awaits_both_backend_cleanups(monkeypatch):
    async def scenario():
        started = [asyncio.Event(), asyncio.Event()]
        stopped = []

        async def backend(index, *args, **kwargs):
            started[index].set()
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0)
                stopped.append(index)

        monkeypatch.setattr(hybrid_search, "perform_vector_search", lambda *a, **k: backend(0))
        monkeypatch.setattr(hybrid_search, "search_elasticsearch", lambda *a, **k: backend(1))
        task = asyncio.create_task(hybrid_search.perform_hybrid_search("query"))
        await asyncio.gather(*(event.wait() for event in started))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(stopped) == [0, 1]

    asyncio.run(scenario())


def test_invalid_hybrid_algorithm_fails_before_search(monkeypatch):
    vector = AsyncMock(return_value=[])
    elastic = AsyncMock(return_value=[])
    monkeypatch.setattr(hybrid_search, "perform_vector_search", vector)
    monkeypatch.setattr(hybrid_search, "search_elasticsearch", elastic)
    with pytest.raises(ValueError, match="algorithm"):
        asyncio.run(hybrid_search.perform_hybrid_search("query", algorithm="typo"))
    vector.assert_not_called()
    elastic.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_result_limits_fail_before_embedding_or_search(monkeypatch, limit):
    embedding = AsyncMock(side_effect=AssertionError("invalid request generated embedding"))
    vector = AsyncMock(side_effect=AssertionError("invalid request launched vector search"))
    elastic = AsyncMock(side_effect=AssertionError("invalid request launched keyword search"))
    monkeypatch.setattr(hybrid_search, "generate_embedding", embedding)
    monkeypatch.setattr(hybrid_search, "search_faiss", vector)
    monkeypatch.setattr(hybrid_search, "search_elasticsearch", elastic)
    with pytest.raises(ValueError, match="max_results"):
        asyncio.run(hybrid_search.perform_vector_search("query", max_results=limit))
    with pytest.raises(ValueError, match="max_results"):
        asyncio.run(hybrid_search.perform_hybrid_search("query", max_results=limit))
    embedding.assert_not_called()
    vector.assert_not_called()
    elastic.assert_not_called()


def test_invalid_request_algorithm_is_rejected_before_search(monkeypatch):
    search = AsyncMock(return_value=([], [], []))
    monkeypatch.setattr(api, "perform_hybrid_search", search)
    response = TestClient(api.create_app()).post(
        "/faiss/combined/", json={"query": "query", "algorithm": "typo"}
    )
    assert response.status_code == 422
    search.assert_not_called()


@pytest.mark.parametrize("path", ["/faiss/search/", "/faiss/elastic/", "/faiss/combined/"])
@pytest.mark.parametrize("query", ["", " ", "\t\r\n", "\u2003\u00a0"])
def test_blank_query_is_rejected_before_cache_or_backend_work(monkeypatch, path, query):
    collaborators = []
    for name in ("get_cached_search_results", "cache_search_results", "perform_vector_search",
                 "search_elasticsearch", "perform_hybrid_search"):
        collaborator = AsyncMock(side_effect=AssertionError("Blank query launched work"))
        collaborators.append(collaborator)
        monkeypatch.setattr(api, name, collaborator)
    response = TestClient(api.create_app()).post(path, json={"query": query})
    assert response.status_code == 422
    for collaborator in collaborators:
        collaborator.assert_not_called()


def test_query_validation_preserves_nonblank_input():
    assert api.QueryRequest(query="  Windows\t11  ").query == "  Windows\t11  "


@pytest.mark.parametrize(("path", "search_name", "result", "payload"), [
    ("/faiss/search/", "perform_vector_search", [], {"faiss_results": []}),
    ("/faiss/elastic/", "search_elasticsearch", [], {"elasticsearch_results": []}),
    ("/faiss/combined/", "perform_hybrid_search", ([], [], []), {"combined_results": [], "count": 0}),
])
def test_incomplete_cached_response_is_replaced_by_fresh_search(monkeypatch, path, search_name, result, payload):
    monkeypatch.setattr(api, "get_cached_search_results", AsyncMock(return_value=payload))
    search = AsyncMock(return_value=result)
    write = AsyncMock()
    monkeypatch.setattr(api, search_name, search)
    monkeypatch.setattr(api, "cache_search_results", write)
    response = TestClient(api.create_app()).post(path, json={"query": "query"})
    assert response.status_code == 200
    assert response.json()["cached"] is False
    search.assert_awaited_once()
    write.assert_awaited_once()


def test_elasticsearch_outage_does_not_cache_an_empty_response(monkeypatch):
    from xf_embed import elastic_engine

    search = AsyncMock(side_effect=RuntimeError("elasticsearch unavailable"))
    client = type("FakeES", (), {"search": search})()
    cache_write = AsyncMock()
    monkeypatch.setattr(elastic_engine, "get_es_client", lambda: client)
    monkeypatch.setattr(elastic_engine, "get_db_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(elastic_engine, "guest_visible_node_ids", AsyncMock(return_value=frozenset({4})))
    monkeypatch.setattr(api, "get_cached_search_results", AsyncMock(return_value=None))
    monkeypatch.setattr(api, "cache_search_results", cache_write)
    response = TestClient(api.create_app()).post("/faiss/elastic/", json={"query": "query"})
    assert response.status_code == 500
    cache_write.assert_not_called()


@pytest.mark.parametrize(("path", "payload"), [
    ("/faiss/search/", {"faiss_results": [], "count": 0, "query": "QUERY", "cached": False}),
    ("/faiss/elastic/", {"elasticsearch_results": [], "count": 0, "query": "QUERY", "cached": False}),
    ("/faiss/combined/", {"combined_results": [], "count": 0, "query": "QUERY", "algorithm": "rrf", "cached": False}),
])
def test_cache_hits_echo_the_current_query_without_mutating_cached_payload(monkeypatch, path, payload):
    monkeypatch.setattr(api, "get_cached_search_results", AsyncMock(return_value=payload))
    response = TestClient(api.create_app()).post(path, json={"query": "query"})
    assert response.status_code == 200
    assert response.json()["query"] == "query"
    assert response.json()["cached"] is True
    assert payload["query"] == "QUERY"
    assert payload["cached"] is False


@pytest.mark.parametrize(("path", "search_name"), [
    ("/faiss/search/", "perform_vector_search"),
    ("/faiss/elastic/", "search_elasticsearch"),
    ("/faiss/combined/", "perform_hybrid_search"),
])
@pytest.mark.parametrize("cached", [True, False])
def test_missing_visibility_authority_returns_503_without_caching(monkeypatch, path, search_name, cached):
    unavailable = VisibilityUnavailable("sensitive database failure")
    cache_read = AsyncMock(side_effect=unavailable) if cached else AsyncMock(return_value=None)
    search = AsyncMock(side_effect=unavailable)
    cache_write = AsyncMock()
    monkeypatch.setattr(api, "get_cached_search_results", cache_read)
    monkeypatch.setattr(api, search_name, search)
    monkeypatch.setattr(api, "cache_search_results", cache_write)
    response = TestClient(api.create_app()).post(path, json={"query": "query"})
    assert response.status_code == 503
    assert "sensitive" not in response.text
    cache_write.assert_not_called()
    if cached:
        search.assert_not_called()


def test_lifespan_closes_all_resources_when_application_body_fails(monkeypatch):
    closers = [AsyncMock() for _ in range(4)]
    for name, closer in zip(
        ["close_db_pool", "close_es_client", "close_cache", "close_embedding_clients"], closers
    ):
        monkeypatch.setattr(api, name, closer)

    async def scenario():
        async with api.lifespan(api.create_app()):
            raise RuntimeError("application failed")

    with pytest.raises(RuntimeError, match="application failed"):
        asyncio.run(scenario())
    for closer in closers:
        closer.assert_awaited_once()


def test_lifespan_continues_cleanup_after_one_closer_fails(monkeypatch):
    closers = [AsyncMock(side_effect=RuntimeError("close failed"))] + [AsyncMock() for _ in range(3)]
    for name, closer in zip(
        ["close_db_pool", "close_es_client", "close_cache", "close_embedding_clients"], closers
    ):
        monkeypatch.setattr(api, name, closer)

    async def scenario():
        async with api.lifespan(api.create_app()):
            pass

    with pytest.raises(RuntimeError, match="close failed"):
        asyncio.run(scenario())
    for closer in closers:
        closer.assert_awaited_once()


def test_mariadb_backend_skips_unused_faiss_startup_and_health_loads(monkeypatch):
    loader = AsyncMock(side_effect=AssertionError("MariaDB must not load a FAISS index"))
    monkeypatch.setattr(api.settings, "VECTOR_BACKEND", "mariadb")
    monkeypatch.setattr(api, "get_faiss_index", loader)
    for name in ("close_db_pool", "close_es_client", "close_cache", "close_embedding_clients"):
        monkeypatch.setattr(api, name, AsyncMock())
    with TestClient(api.create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["vector_backend"] == "mariadb"
    assert response.json()["faiss_index_loaded"] is False
    assert response.json()["faiss_total_vectors"] == 0
    loader.assert_not_called()
