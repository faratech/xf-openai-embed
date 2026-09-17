import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from xf_embed.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_healthcheck(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "elasticsearch" in data
    assert "vector_backend" in data


@patch("xf_embed.api.perform_vector_search", new_callable=AsyncMock)
def test_vector_search_endpoint(mock_search, client):
    mock_search.return_value = [
        {
            "type": "post",
            "post_id": 101,
            "thread_id": 202,
            "thread_title": "Windows Blue Screen",
            "message": "Sample post content",
            "post_date": 1700000000,
            "score": 0.88,
        }
    ]

    response = client.post("/faiss/search/", json={"query": "blue screen", "max_results": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["query"] == "blue screen"
    assert data["faiss_results"][0]["post_id"] == 101
    assert data["faiss_results"][0]["score"] == 0.88


@patch("xf_embed.api.search_elasticsearch", new_callable=AsyncMock)
def test_elastic_search_endpoint(mock_search, client):
    mock_search.return_value = [
        {
            "type": "post",
            "post_id": 303,
            "thread_id": 404,
            "thread_title": "Elasticsearch Result",
            "message": "Keyword hit",
            "post_date": 1700000000,
            "score": 4.5,
        }
    ]

    response = client.post("/faiss/elastic/", json={"query": "keyword test", "max_results": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["elasticsearch_results"][0]["post_id"] == 303


@patch("xf_embed.api.perform_hybrid_search", new_callable=AsyncMock)
def test_combined_search_endpoint(mock_search, client):
    mock_search.return_value = (
        [
            {
                "type": "post",
                "post_id": 505,
                "thread_id": 606,
                "thread_title": "Hybrid Result",
                "message": "Fused post",
                "post_date": 1700000000,
                "score": 0.032,
                "rrf_score": 0.032,
            }
        ],
        [],
        [],
    )

    response = client.post("/faiss/combined/", json={"query": "hybrid test", "max_results": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["combined_results"][0]["post_id"] == 505
    assert data["combined_results"][0]["rrf_score"] == 0.032
