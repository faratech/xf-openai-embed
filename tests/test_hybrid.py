from xf_embed.hybrid_search import reciprocal_rank_fusion, weighted_score_fusion, compute_item_key


def test_compute_item_key():
    assert compute_item_key({"post_id": 123}) == "post:123"
    assert compute_item_key({"thread_id": 456}) == "thread:456"
    assert compute_item_key({"record_id": 789}) == "rec:789"


def test_reciprocal_rank_fusion():
    list_a = [
        {"post_id": 1, "thread_title": "Doc 1"},
        {"post_id": 2, "thread_title": "Doc 2"},
    ]
    list_b = [
        {"post_id": 2, "thread_title": "Doc 2"},
        {"post_id": 3, "thread_title": "Doc 3"},
    ]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    # Post 2 appears in list_a at rank 2 and list_b at rank 1:
    # score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.03252
    assert fused[0]["post_id"] == 2
    expected_score = (1.0 / 62.0) + (1.0 / 61.0)
    assert abs(fused[0]["rrf_score"] - expected_score) < 1e-4

    # Post 1 appears in list_a at rank 1: 1/61 = ~0.01639
    # Post 3 appears in list_b at rank 2: 1/62 = ~0.01612
    assert fused[1]["post_id"] == 1
    assert fused[2]["post_id"] == 3


def test_weighted_score_fusion():
    vec_results = [
        {"post_id": 1, "score": 0.9},
        {"post_id": 2, "score": 0.5},
    ]
    es_results = [
        {"post_id": 2, "es_score": 10.0},
        {"post_id": 1, "es_score": 2.0},
    ]

    fused = weighted_score_fusion(vec_results, es_results, alpha=0.5)
    assert len(fused) == 2
    # Both items have normalized scores and combined score is computed
    assert "vector_norm_score" in fused[0]
    assert "elastic_norm_score" in fused[0]
