from xf_embed.config import Settings


def test_default_settings():
    s = Settings()
    assert s.OPENAI_DIMENSIONS == 1536
    assert s.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert s.FAISS_INDEX_PATH == "faiss_index.bin"
    assert s.RRF_K == 60
    assert s.HYBRID_ALGORITHM in ("rrf", "weighted")
    assert s.VECTOR_BACKEND in ("faiss", "mariadb")
