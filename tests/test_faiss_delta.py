"""index-faiss-delta: add only the rows written after the index on disk was built."""

import asyncio

import faiss
import numpy as np
import pytest

from xf_embed import db, vector_engine


def _index_with(tmp_path, ids):
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(3))
    if ids:
        index.add_with_ids(np.eye(3, dtype=np.float32)[: len(ids)], np.asarray(ids, dtype=np.int64))
    path = tmp_path / "faiss_index.bin"
    faiss.write_index(index, str(path))
    return path


def _stub_db(monkeypatch, pages):
    cursors = []
    pages = iter(pages)

    async def get_pool():
        return object()

    async def fetch(pool, last_id, batch_size):
        cursors.append(last_id)
        return next(pages)

    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_embeddings_batch", fetch)
    return cursors


def _page(ids, last_id):
    vectors = np.ones((len(ids), 3), dtype=np.float32) if ids else np.empty((0, 3), dtype=np.float32)
    return list(ids), [0] * len(ids), [None] * len(ids), vectors, last_id


def test_delta_starts_after_the_highest_indexed_id_and_saves(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 9, 7])
    cursors = _stub_db(monkeypatch, [_page([10, 12], 12), _page([], 13), _page([14], 14), _page([], 14)])
    start_id, added = asyncio.run(vector_engine.update_faiss_index_delta(batch_size=2, path=str(path)))
    assert (start_id, added) == (9, 3)
    assert cursors == [9, 12, 13, 14], "a page of invalid rows advances the cursor without stopping"
    saved = faiss.read_index(str(path))
    assert sorted(faiss.vector_to_array(saved.id_map).tolist()) == [5, 7, 9, 10, 12, 14]


def test_no_new_rows_leaves_the_file_untouched(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5])
    before = path.stat().st_mtime_ns
    monkeypatch.setattr(vector_engine, "save_faiss_index", lambda *a: pytest.fail("nothing to save"))
    _stub_db(monkeypatch, [_page([], 5)])
    assert asyncio.run(vector_engine.update_faiss_index_delta(path=str(path))) == (5, 0)
    assert path.stat().st_mtime_ns == before


def test_a_missing_index_needs_a_full_build_first(tmp_path, monkeypatch):
    _stub_db(monkeypatch, [])
    with pytest.raises(RuntimeError, match="index-faiss"):
        asyncio.run(vector_engine.update_faiss_index_delta(path=str(tmp_path / "absent.bin")))


def test_an_empty_index_starts_from_zero(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [])
    cursors = _stub_db(monkeypatch, [_page([1], 1), _page([], 1)])
    assert asyncio.run(vector_engine.update_faiss_index_delta(path=str(path))) == (0, 1)
    assert cursors == [0, 1]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_nonpositive_batch_sizes(tmp_path, batch_size):
    with pytest.raises(ValueError):
        asyncio.run(vector_engine.update_faiss_index_delta(batch_size=batch_size, path=str(tmp_path / "x.bin")))
