import numpy as np
from xf_embed.embeddings import normalize_vector, truncate_text, count_tokens


def test_normalize_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_vector(v)
    assert np.isclose(np.linalg.norm(normed), 1.0)
    assert np.isclose(normed[0], 0.6)
    assert np.isclose(normed[1], 0.8)


def test_count_tokens_and_truncate():
    text = "Hello world! This is a test string."
    tok_count = count_tokens(text)
    assert tok_count > 0

    truncated = truncate_text(text, max_tokens=3)
    assert len(truncated) <= len(text)
    assert count_tokens(truncated) <= 3
