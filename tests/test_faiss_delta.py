"""index-faiss-delta: reconcile the index on disk with openai_embeddings without a full rebuild."""

import asyncio
import json
import os

import faiss
import numpy as np
import pytest

from xf_embed import db, vector_engine

NOW = 1_800_000_000


def _vec(i):
    v = np.zeros(3, dtype=np.float32)
    v[i % 3] = 1.0
    v[(i + 1) % 3] = 0.5 * (i % 5)
    return v / np.linalg.norm(v)


def _index_with(tmp_path, ids, inner=None):
    index = faiss.IndexIDMap2(inner or faiss.IndexFlatIP(3))
    if ids:
        index.add_with_ids(np.vstack([_vec(i) for i in ids]), np.asarray(ids, dtype=np.int64))
    path = tmp_path / "faiss_index.bin"
    faiss.write_index(index, str(path))
    return path


def _stub_db(monkeypatch, table, changed_since=(), new_vectors=None):
    """table: ids in openai_embeddings. changed_since: ids whose last_updated is past the watermark.
    new_vectors: id -> current vector (None = invalid vector, skipped like the real decoder)."""
    new_vectors = new_vectors or {}
    seen = {"since": [], "fetched": []}

    async def get_pool():
        return object()

    async def now(pool):
        return NOW

    async def ids(pool):
        return np.asarray(sorted(table), dtype=np.int64)

    async def updated(pool, since):
        seen["since"].append(since)
        return np.asarray(sorted(changed_since), dtype=np.int64)

    async def by_ids(pool, record_ids):
        seen["fetched"].append(list(record_ids))
        rows = [(i, new_vectors.get(i, _vec(i))) for i in record_ids if i in table]
        rows = [(i, v) for i, v in rows if v is not None]
        vectors = np.vstack([v for _, v in rows]) if rows else np.empty((0, 3), dtype=np.float32)
        return [i for i, _ in rows], [None] * len(rows), [None] * len(rows), vectors, 0

    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "db_unix_now", now)
    monkeypatch.setattr(db, "fetch_embedding_ids", ids)
    monkeypatch.setattr(db, "fetch_ids_updated_since", updated)
    monkeypatch.setattr(db, "fetch_embeddings_by_ids", by_ids)
    return seen


def _ids(path):
    index = faiss.read_index(str(path))   # keep the index alive while its id_map is read
    return sorted(faiss.vector_to_array(index.id_map).tolist())


def _run(path, **kwargs):
    return asyncio.run(vector_engine.update_faiss_index_delta(path=str(path), **kwargs))


def _state(path, since):
    vector_engine.index_state_path(str(path)).write_text(json.dumps({"since": since, "ntotal": 0}))


def test_adds_new_rows_including_ones_below_the_highest_indexed_id(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 9])
    _state(path, NOW - 3600)
    _stub_db(monkeypatch, table={5, 7, 9, 12})       # 7 committed after 9 had been indexed
    r = _run(path)
    assert (r["added"], r["updated"], r["removed"], r["invalid"]) == (2, 0, 0, 0)
    assert _ids(path) == [5, 7, 9, 12]


def test_removes_rows_deleted_from_the_table(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 7, 9])
    _state(path, NOW - 3600)
    _stub_db(monkeypatch, table={5, 9})
    r = _run(path)
    assert (r["added"], r["removed"]) == (0, 1)
    assert _ids(path) == [5, 9]


def test_re_embedded_rows_get_their_current_vector(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 7])
    _state(path, NOW - 3600)
    fresh = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    seen = _stub_db(monkeypatch, table={5, 7}, changed_since={7}, new_vectors={7: fresh})
    r = _run(path)
    assert (r["added"], r["updated"], r["removed"]) == (0, 1, 0)
    assert seen["since"] == [NOW - 3600], "the watermark comes from the state file"
    saved = faiss.read_index(str(path))
    assert saved.ntotal == 2
    np.testing.assert_allclose(saved.reconstruct(7), fresh, atol=1e-6)
    np.testing.assert_allclose(saved.reconstruct(5), _vec(5), atol=1e-6)


def test_a_re_embedded_row_with_an_invalid_vector_is_dropped(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 7])
    _state(path, NOW - 3600)
    _stub_db(monkeypatch, table={5, 7}, changed_since={7}, new_vectors={7: None})
    r = _run(path)
    assert (r["updated"], r["invalid"]) == (0, 1)
    assert _ids(path) == [5]


def test_the_state_advances_to_the_run_start_minus_the_margin(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5])
    _state(path, NOW - 99999)
    _stub_db(monkeypatch, table={5})
    before = path.stat().st_mtime_ns
    monkeypatch.setattr(vector_engine, "save_faiss_index", lambda *a: pytest.fail("nothing to save"))
    r = _run(path)
    assert (r["added"], r["updated"], r["removed"]) == (0, 0, 0)
    assert path.stat().st_mtime_ns == before
    assert vector_engine.read_index_state(str(path))["since"] == NOW - vector_engine.STATE_MARGIN_S


def test_without_a_state_file_it_falls_back_to_the_index_mtime(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5])
    os.utime(path, (NOW - 5000, NOW - 5000))
    seen = _stub_db(monkeypatch, table={5})
    _run(path)
    assert seen["since"] == [NOW - 5000 - vector_engine.STATE_MARGIN_S]


def test_an_explicit_since_overrides_the_state(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5])
    _state(path, NOW - 3600)
    seen = _stub_db(monkeypatch, table={5})
    _run(path, since=1_790_000_000)
    assert seen["since"] == [1_790_000_000]


def test_refetches_in_batches(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [1])
    _state(path, NOW - 3600)
    seen = _stub_db(monkeypatch, table={1, 2, 3, 4, 5, 6})
    assert _run(path, batch_size=2)["added"] == 5
    assert seen["fetched"] == [[2, 3], [4, 5], [6]]


def test_an_index_that_cannot_remove_refuses_instead_of_going_stale(tmp_path, monkeypatch):
    path = _index_with(tmp_path, [5, 7], inner=faiss.IndexHNSWFlat(3, 8, faiss.METRIC_INNER_PRODUCT))
    _state(path, NOW - 3600)
    _stub_db(monkeypatch, table={5})
    with pytest.raises(RuntimeError, match="full rebuild"):
        _run(path)
    assert _ids(path) == [5, 7]


def test_a_missing_index_needs_a_full_build_first(tmp_path, monkeypatch):
    _stub_db(monkeypatch, table=set())
    with pytest.raises(RuntimeError, match="index-faiss"):
        _run(tmp_path / "absent.bin")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_nonpositive_batch_sizes(tmp_path, batch_size):
    with pytest.raises(ValueError):
        _run(tmp_path / "x.bin", batch_size=batch_size)
