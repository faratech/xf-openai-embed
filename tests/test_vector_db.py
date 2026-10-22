"""Vector/database regressions using local fakes, never configured services."""

import asyncio
import sqlite3
import threading
from pathlib import Path

import faiss
import numpy as np
import pytest

from xf_embed import db, vector_engine


@pytest.fixture(autouse=True)
def isolate_services(monkeypatch):
    async def forbidden_connection(*args, **kwargs):
        raise AssertionError("Unexpected connection to a configured service")

    monkeypatch.setattr(db.asyncmy, "create_pool", forbidden_connection)
    monkeypatch.setattr(vector_engine, "get_db_pool", forbidden_connection)
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_lock", asyncio.Lock())
    monkeypatch.setattr(vector_engine, "_faiss_index", None)
    monkeypatch.setattr(vector_engine, "_index_lock", asyncio.Lock())
    monkeypatch.setattr(db.settings, "OPENAI_DIMENSIONS", 3)
    monkeypatch.setattr(db.settings, "FAISS_METRIC", "cosine")


class Context:
    def __init__(self, value, on_exit=None):
        self.value = value
        self.on_exit = on_exit

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        if self.on_exit:
            self.on_exit()


class FakePool:
    def __init__(self, rows=(), connection=None):
        self.rows = rows
        self.connection = connection
        self.active = False
        self.calls = []

    def acquire(self):
        self.active = True
        return Context(self.connection or self, lambda: setattr(self, "active", False))

    def cursor(self, *args):
        return Context(self)

    async def execute(self, query, params):
        self.calls.append((query, params))

    async def fetchall(self):
        return self.rows


def test_batch_skips_malformed_vectors_and_releases_connection_before_decoding(monkeypatch):
    rows = [
        (1, 10, None, b"x"),  # Binary size is not divisible by sizeof(float32).
        (2, 20, None, "42"),  # JSON scalar.
        (3, 30, None, "[[1, 2, 3]]"),
        (4, 40, None, "[1, 2]"),
        (5, 50, None, "[NaN, 0, 1]"),
        (6, 60, None, np.array([1, np.inf, 0], dtype=np.float32).tobytes()),
        (7, 70, None, "[1, 2, 3]"),
        (8, None, 80, np.array([3, 2, 1], dtype=np.float32).tobytes()),
        (9, None, 90, "invalid JSON"),
    ]
    pool = FakePool(rows)
    original_decode = db._decode_embeddings_batch
    caller_thread = threading.get_ident()

    def decode(*args):
        assert not pool.active
        assert threading.get_ident() != caller_thread
        return original_decode(*args)

    monkeypatch.setattr(db, "_decode_embeddings_batch", decode)
    ids, posts, threads, vectors, last_id = asyncio.run(db.fetch_embeddings_batch(pool, batch_size=9))
    assert ids == [7, 8]
    assert posts == [70, None]
    assert threads == [None, 80]
    assert last_id == 9
    np.testing.assert_array_equal(vectors, [[1, 2, 3], [3, 2, 1]])
    assert vectors.dtype == np.float32
    assert pool.calls[0][1] == (0, 9)


def test_batch_all_invalid_preserves_cursor():
    pool = FakePool([(11, None, None, "null"), (12, None, None, b"x")])
    ids, _, _, vectors, last_id = asyncio.run(db.fetch_embeddings_batch(pool, last_id=10))
    assert ids == []
    assert vectors.shape == (0, 3)
    assert last_id == 12


@pytest.mark.parametrize("blob", ["[0,0,0]", np.zeros(3, dtype=np.float32).tobytes()])
def test_batch_rejects_zero_vectors_without_stopping_pagination(blob):
    pool = FakePool([(11, 7, None, blob), (12, 7, None, "[1,0,0]")])
    ids, _, _, vectors, last_id = asyncio.run(db.fetch_embeddings_batch(pool, last_id=10))
    assert ids == [12]
    np.testing.assert_array_equal(vectors, [[1, 0, 0]])
    assert last_id == 12


@pytest.mark.parametrize("batch_size", [0, -1])
def test_reject_nonpositive_batch_sizes_before_connecting(batch_size):
    pool = FakePool()
    with pytest.raises(ValueError, match="batch_size"):
        asyncio.run(db.fetch_embeddings_batch(pool, batch_size=batch_size))
    assert not pool.calls
    with pytest.raises(ValueError, match="batch_size"):
        asyncio.run(vector_engine.build_faiss_index(batch_size=batch_size))


def test_build_continues_after_invalid_page_and_offloads_cpu_and_save(monkeypatch):
    batches = iter([
        ([], [], [], np.empty((0, 3), dtype=np.float32), 2),
        ([3], [30], [None], np.array([[3, 0, 0]], dtype=np.float32), 3),
        ([], [], [], np.empty((0, 3), dtype=np.float32), 3),
    ])
    cursors = []
    progress = []
    caller_thread = threading.get_ident()

    class Index:
        ntotal = 0

        def add_with_ids(self, vectors, ids):
            assert threading.get_ident() != caller_thread
            np.testing.assert_array_equal(vectors, [[1, 0, 0]])
            np.testing.assert_array_equal(ids, [3])
            self.ntotal += len(ids)

    async def get_pool():
        return object()

    async def fetch(pool, last_id, batch_size):
        cursors.append(last_id)
        return next(batches)

    def save(index, path):
        assert threading.get_ident() != caller_thread
        assert index.ntotal == 1
        assert path == "unused.bin"

    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_embeddings_batch", fetch)
    monkeypatch.setattr(vector_engine, "create_base_faiss_index", lambda _: Index())
    monkeypatch.setattr(vector_engine, "save_faiss_index", save)
    index = asyncio.run(vector_engine.build_faiss_index(
        batch_size=2, save_path="unused.bin", progress_callback=lambda *args: progress.append(args),
    ))
    assert index.ntotal == 1
    assert vector_engine._faiss_index is index
    assert cursors == [0, 2, 3]
    assert progress == [(1, 1)]


def test_pool_initialization_is_single_flight(monkeypatch):
    calls = []
    pool = object()

    async def create(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0)
        return pool

    monkeypatch.setattr(db.asyncmy, "create_pool", create)

    async def run():
        return await asyncio.gather(*(db.get_db_pool() for _ in range(20)))

    assert all(result is pool for result in asyncio.run(run()))
    assert len(calls) == 1


def test_failed_pool_initialization_can_be_retried(monkeypatch):
    calls = 0
    pool = object()

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary connection failure")
        return pool

    monkeypatch.setattr(db.asyncmy, "create_pool", create)

    async def run():
        with pytest.raises(RuntimeError, match="temporary"):
            await db.get_db_pool()
        assert await db.get_db_pool() is pool

    asyncio.run(run())
    assert calls == 2


def test_pool_reopen_waits_for_close(monkeypatch):
    async def run():
        closing = asyncio.Event()
        release = asyncio.Event()
        replacement = object()

        class Pool:
            def close(self):
                closing.set()

            async def wait_closed(self):
                await release.wait()

        async def create(**kwargs):
            assert release.is_set()
            return replacement

        monkeypatch.setattr(db, "_pool", Pool())
        monkeypatch.setattr(db.asyncmy, "create_pool", create)
        close_task = asyncio.create_task(db.close_db_pool())
        await closing.wait()
        get_task = asyncio.create_task(db.get_db_pool())
        await asyncio.sleep(0)
        assert not get_task.done()
        release.set()
        await close_task
        assert await get_task is replacement

    asyncio.run(run())


def test_search_keeps_loop_responsive_and_preserves_ranking(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()

    class Index:
        ntotal = 2

        def search(self, vector, top_k):
            assert threading.get_ident() != caller_thread
            np.testing.assert_array_equal(vector, [[1, 0, 0]])
            assert top_k == 2
            started.set()
            assert release.wait(timeout=2), "event loop was blocked by search"
            return np.array([[0.9, 0.8]]), np.array([[22, 11]])

    async def get_index():
        return Index()

    async def get_pool():
        return object()

    async def fetch(pool, ids):
        assert ids == [22, 11]
        return {11: {"post_id": 1}, 22: {"post_id": 2}}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", fetch)

    async def heartbeat():
        while not started.is_set():
            await asyncio.sleep(0)
        release.set()

    async def run():
        result, _ = await asyncio.gather(
            vector_engine.search_faiss(np.array([3, 0, 0], dtype=np.float32), top_k=3), heartbeat(),
        )
        return result

    results = asyncio.run(run())
    assert [result["post_id"] for result in results] == [2, 1]
    assert [result["score"] for result in results] == [0.9, 0.8]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_faiss_rejects_invalid_limits_before_loading_index(monkeypatch, top_k):
    async def unexpected_index():
        raise AssertionError("invalid limits must fail before loading an index")

    monkeypatch.setattr(vector_engine, "get_faiss_index", unexpected_index)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=top_k))


@pytest.mark.parametrize("first_visible", [2049, 2051])
def test_faiss_backfills_filtered_hits_without_rehydrating_records(monkeypatch, first_visible):
    searches = []
    hydrated = []

    class Index:
        ntotal = 2050

        def search(self, vector, top_k):
            searches.append(top_k)
            ids = np.arange(1, top_k + 1, dtype=np.int64)
            return np.array([1 - ids / 10000]), np.array([ids])

    async def get_index():
        return Index()

    async def get_pool():
        return object()

    async def hydrate(pool, ids):
        assert len(ids) <= 1000
        hydrated.extend(ids)
        return {rec_id: {"post_id": rec_id} for rec_id in ids if rec_id >= first_visible}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", hydrate)
    results = asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=2))
    assert [result["post_id"] for result in results] == list(range(first_visible, 2051))
    assert searches == [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2050]
    assert hydrated == list(range(1, 2051))


def test_faiss_counts_unique_forum_items_and_retains_best_section(monkeypatch):
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(3))
    index.add_with_ids(
        np.array([[1, 0, 0], [.9, .4358899, 0], [.8, .6, 0], [.7, .7141428, 0]], dtype=np.float32),
        np.array([1, 2, 3, 4], dtype=np.int64),
    )
    hydrated = []

    async def get_index():
        return index

    async def get_pool():
        return object()

    async def hydrate(pool, ids):
        hydrated.extend(ids)
        details = {1: {"post_id": 7}, 2: {"post_id": 7}, 3: {"thread_id": 8}, 4: {"post_id": 9}}
        return {rec_id: details[rec_id] for rec_id in ids}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", hydrate)
    results = asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=2))
    assert [(result.get("post_id"), result.get("thread_id")) for result in results] == [(7, None), (None, 8)]
    assert results[0]["score"] == 1.0
    assert hydrated == [1, 2, 3, 4]


def test_faiss_stops_hydrating_after_filling_unique_results(monkeypatch):
    hydrated = []

    class Index:
        ntotal = 8192

        def search(self, vector, top_k):
            ids = np.arange(1, top_k + 1, dtype=np.int64)
            return np.array([1 - ids / 10000]), np.array([ids])

    async def get_index():
        return Index()

    async def get_pool():
        return object()

    async def hydrate(pool, ids):
        assert len(ids) <= 1000
        hydrated.extend(ids)
        return {rec_id: {"post_id": rec_id} for rec_id in ids if rec_id >= 4097}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", hydrate)
    results = asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=3))
    assert [result["post_id"] for result in results] == [4097, 4098, 4099]
    assert len(hydrated) == len(set(hydrated))
    assert max(hydrated) < Index.ntotal


def test_faiss_exhaustion_skips_sentinels_and_returns_available_unique_items(monkeypatch):
    class Index:
        ntotal = 4

        def search(self, vector, top_k):
            assert top_k == self.ntotal
            return np.array([[.9, .8, .7, -np.inf]]), np.array([[1, 2, 3, -1]])

    async def get_index():
        return Index()

    async def get_pool():
        return object()

    async def hydrate(pool, ids):
        assert ids == [1, 2, 3]
        return {1: {"post_id": 7}, 2: {"post_id": 7}}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", hydrate)
    results = asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=10))
    assert len(results) == 1
    assert results[0]["post_id"] == 7
    assert results[0]["score"] == .9


@pytest.mark.parametrize("metric", ["cosine", "l2"])
@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_faiss_backfill_keeps_metric_order_for_real_indexes(monkeypatch, metric, index_type):
    monkeypatch.setattr(vector_engine.settings, "FAISS_METRIC", metric)
    monkeypatch.setattr(vector_engine.settings, "FAISS_INDEX_TYPE", index_type)
    index = vector_engine.create_base_faiss_index(3)
    vectors = np.array([[1, 0, 0], [.9, .4358899, 0], [.8, .6, 0], [.7, .7141428, 0]], dtype=np.float32)
    index.add_with_ids(vectors, np.array([1, 2, 3, 4], dtype=np.int64))

    async def get_index():
        return index

    async def get_pool():
        return object()

    async def hydrate(pool, ids):
        return {rec_id: {"post_id": rec_id} for rec_id in ids if rec_id >= 3}

    monkeypatch.setattr(vector_engine, "get_faiss_index", get_index)
    monkeypatch.setattr(vector_engine, "get_db_pool", get_pool)
    monkeypatch.setattr(db, "fetch_details_by_record_ids", hydrate)
    results = asyncio.run(vector_engine.search_faiss(np.array([1, 0, 0]), top_k=2))
    assert [result["post_id"] for result in results] == [3, 4]
    assert results[0]["score"] > results[1]["score"]


def test_disk_load_is_offloaded_and_shared(monkeypatch):
    caller_thread = threading.get_ident()
    calls = []
    index = object()

    def load():
        assert threading.get_ident() != caller_thread
        calls.append(True)
        return index

    monkeypatch.setattr(vector_engine, "load_faiss_index_from_disk", load)

    async def run():
        return await asyncio.gather(*(vector_engine.get_faiss_index() for _ in range(10)))

    assert all(result is index for result in asyncio.run(run()))
    assert len(calls) == 1


def test_failed_save_preserves_previous_index_and_cleans_tempfile(monkeypatch, tmp_path):
    target = tmp_path / "index.bin"
    target.write_bytes(b"previous index")

    def fail_write(index, path):
        Path(path).write_bytes(b"partial index")
        raise RuntimeError("disk failure")

    monkeypatch.setattr(vector_engine.faiss, "write_index", fail_write)
    with pytest.raises(RuntimeError, match="disk failure"):
        vector_engine.save_faiss_index(type("Index", (), {"ntotal": 1})(), str(target))
    assert target.read_bytes() == b"previous index"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_save_roundtrip_preserves_permissions(tmp_path):
    target = tmp_path / "index.bin"
    target.touch(mode=0o640)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(3))
    index.add_with_ids(np.array([[1, 0, 0]], dtype=np.float32), np.array([42], dtype=np.int64))
    vector_engine.save_faiss_index(index, str(target))
    loaded = vector_engine.load_faiss_index_from_disk(str(target))
    assert loaded.ntotal == 1
    assert loaded.d == 3
    assert target.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [target]


class SQLiteCursor:
    """Exercise actual SQL joins in memory behind the asyncmy cursor seam."""

    def __init__(self, connection):
        self.connection = connection

    async def execute(self, query, params):
        self.rows = self.connection.execute(query.replace("%s", "?"), params).fetchall()

    async def fetchall(self):
        return [dict(row) for row in self.rows]


class SQLiteConnection:
    def __init__(self, connection):
        self.connection = connection

    def cursor(self, *args):
        return Context(SQLiteCursor(self.connection))


@pytest.fixture
def metadata_pool():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE openai_embeddings(id, post_id, thread_id, embedding);
        CREATE TABLE xf_post(post_id, thread_id, message, post_date, username, user_id, message_state);
        CREATE TABLE xf_thread(thread_id, title, post_date, last_post_date, username, user_id, node_id, discussion_state, first_post_id);
        CREATE TABLE xf_node(node_id, node_type_id);
        CREATE TABLE xf_permission_combination(permission_combination_id, user_id, user_group_list);
        CREATE TABLE xf_permission_cache_content(content_type, content_id, permission_combination_id, cache_value);
        INSERT INTO xf_permission_combination VALUES (1, 0, '1');
        INSERT INTO xf_node VALUES (1, 'Forum'), (2, 'Forum'), (3, 'Forum');
        INSERT INTO xf_permission_cache_content VALUES
            ('node', 1, 1, '{"view": true, "viewOthers": true, "viewContent": true}'),
            ('node', 2, 1, '{"view": false, "viewOthers": true, "viewContent": true}');
        INSERT INTO xf_thread VALUES (8, 'Thread title', 1, 2, 'thread author', 5, 1, 'visible', 7);
        INSERT INTO xf_post VALUES (7, 8, 'post body', 4, 'post author', 6, 'visible');
        INSERT INTO openai_embeddings VALUES (1, 7, NULL, '0.1'), (2, NULL, 8, '0.2');
    """)
    connection.create_function("VEC_FromText", 1, lambda value: value)
    connection.create_function("VEC_DISTANCE_COSINE", 2, lambda stored, query: float(stored))
    # Numeric strings stand in for valid three-dimensional vectors with a
    # predetermined cosine distance; binary cases exercise actual byte widths.
    connection.create_function("OCTET_LENGTH", 1, lambda stored: len(stored) if isinstance(stored, bytes) else 12)

    def euclidean(stored, zero):
        if not isinstance(stored, bytes):
            return 1.0
        if len(stored) != len(zero):
            return None
        return float(np.linalg.norm(np.frombuffer(stored, dtype=np.float32)))

    connection.create_function("VEC_DISTANCE_EUCLIDEAN", 2, euclidean)
    yield FakePool(connection=SQLiteConnection(connection))
    connection.close()


def test_faiss_metadata_resolves_post_parent_thread(metadata_pool):
    results = asyncio.run(db.fetch_details_by_record_ids(metadata_pool, [1, 2]))
    assert results[1]["thread_id"] == 8
    assert results[1]["thread_title"] == "Thread title"
    assert results[1]["username"] == "post author"
    assert results[1]["user_id"] == 6
    assert results[1]["node_id"] == 1
    assert results[2]["thread_id"] == 8
    assert results[2]["type"] == "thread"


def test_native_metadata_includes_thread_titles_authors_and_parent_ids(metadata_pool):
    results = asyncio.run(db.search_mariadb_native_vector(metadata_pool, np.array([1, 0, 0]), top_k=2))
    assert [result["thread_id"] for result in results] == [8, 8]
    assert [result["thread_title"] for result in results] == ["Thread title", "Thread title"]
    assert [result["username"] for result in results] == ["post author", "thread author"]
    assert [result["user_id"] for result in results] == [6, 5]
    assert [result["node_id"] for result in results] == [1, 1]
    assert [result["post_date"] for result in results] == [4, 2]
    assert [result["type"] for result in results] == ["post", "thread"]
    assert [result["score"] for result in results] == [0.9, 0.8]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_native_rejects_invalid_limits_before_reading_permissions(monkeypatch, top_k):
    async def unexpected_visibility(pool):
        raise AssertionError("invalid limits must fail before reading permissions")

    monkeypatch.setattr(db, "guest_visible_node_ids", unexpected_visibility)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(db.search_mariadb_native_vector(object(), np.array([1, 0, 0]), top_k=top_k))


def test_native_limits_unique_items_and_computes_each_candidate_distance_once(metadata_pool):
    connection = metadata_pool.connection.connection
    connection.execute("INSERT INTO openai_embeddings VALUES (3, 7, NULL, '0.05')")
    evaluated = []

    def distance(stored, query):
        evaluated.append(stored)
        return float(stored)

    connection.create_function("VEC_DISTANCE_COSINE", 2, distance)
    results = asyncio.run(db.search_mariadb_native_vector(metadata_pool, np.array([1, 0, 0]), top_k=2))
    assert [(result["post_id"], result["thread_id"]) for result in results] == [(7, 8), (None, 8)]
    assert [result["distance"] for result in results] == [0.05, 0.2]
    assert sorted(evaluated) == ['0.05', '0.1', '0.2']


def add_hidden_content(pool):
    connection = pool.connection.connection
    connection.executescript("""
        INSERT INTO xf_thread VALUES
            (9, 'private title', 1, 2, 'author', 5, 2, 'visible', 11),
            (10, 'moderated title', 1, 2, 'author', 5, 1, 'moderated', 10),
            (11, 'deleted title', 1, 2, 'author', 5, 1, 'deleted', 0),
            (12, 'unclassified title', 1, 2, 'author', 5, 3, 'visible', 0);
        INSERT INTO xf_post VALUES
            (8, 8, 'deleted body', 4, 'author', 6, 'deleted'),
            (9, 8, 'moderated body', 4, 'author', 6, 'moderated'),
            (10, 10, 'moderated parent body', 4, 'author', 6, 'visible'),
            (11, 9, 'private body', 4, 'author', 6, 'visible'),
            (12, 99, 'missing parent body', 4, 'author', 6, 'visible');
        INSERT INTO openai_embeddings VALUES
            (3, 8, 8, '0.01'), (4, 9, 8, '0.01'), (5, 10, 10, '0.01'),
            (6, 11, 9, '0.01'), (7, 12, 99, '0.01'), (8, NULL, 9, '0.01'),
            (9, NULL, 10, '0.01'), (10, NULL, 11, '0.01'), (11, NULL, 12, '0.01');
    """)


def test_record_hydration_excludes_nonpublic_content(metadata_pool):
    add_hidden_content(metadata_pool)
    results = asyncio.run(db.fetch_details_by_record_ids(metadata_pool, list(range(1, 12))))
    assert set(results) == {1, 2}


def test_native_filters_hidden_content_before_top_k_limit(metadata_pool):
    add_hidden_content(metadata_pool)
    results = asyncio.run(db.search_mariadb_native_vector(metadata_pool, np.array([1, 0, 0]), top_k=1))
    assert len(results) == 1
    assert results[0]["post_id"] == 7
    assert results[0]["distance"] == 0.1


@pytest.mark.parametrize("stored", [
    "malformed",
    np.zeros(3, dtype=np.float32).tobytes(),
    np.array([-0.0, 0.0, -0.0], dtype=np.float32).tobytes(),
    np.array([1.0, 0.0], dtype=np.float32).tobytes(),
    np.array([np.inf, 1.0, 0.0], dtype=np.float32).tobytes(),
    b"x",
])
def test_native_excludes_invalid_or_zero_vectors_before_limit(metadata_pool, stored):
    connection = metadata_pool.connection.connection
    connection.execute("INSERT INTO openai_embeddings VALUES (3, 7, NULL, ?)", (stored,))

    def distance(value, query):
        if value == "malformed":
            return None
        if isinstance(value, bytes):
            if len(value) != 12 or not np.isfinite(np.frombuffer(value, dtype=np.float32)).all():
                return None
            # MariaDB reports a zero distance for an all-zero stored vector.
            return 0.0
        return float(value)

    connection.create_function("VEC_DISTANCE_COSINE", 2, distance)
    results = asyncio.run(db.search_mariadb_native_vector(metadata_pool, np.array([1, 0, 0]), top_k=1))
    assert len(results) == 1
    assert results[0]["distance"] == 0.1


def test_item_hydration_drops_stale_private_and_missing_items(metadata_pool):
    add_hidden_content(metadata_pool)
    items = [
        {"post_id": 7, "score": 0.9, "message": "stale body"},
        {"post_id": 8, "thread_id": 8, "message": "stale deleted body"},
        {"post_id": 9}, {"post_id": 10}, {"post_id": 11}, {"post_id": 12}, {"post_id": 99},
        {"thread_id": 8, "score": 0.8, "thread_title": "stale title"},
        {"thread_id": 9}, {"thread_id": 10}, {"thread_id": 11}, {"thread_id": 12},
    ]
    results = asyncio.run(db.fetch_details_for_posts_and_threads(metadata_pool, items))
    assert len(results) == 2
    assert [result["type"] for result in results] == ["post", "thread"]
    assert [result["score"] for result in results] == [0.9, 0.8]
    assert results[0]["message"] == "post body"
    assert results[0]["thread_id"] == 8
    assert results[0]["user_id"] == 6
    assert results[0]["node_id"] == 1
    assert results[1]["thread_title"] == "Thread title"
    assert results[1]["message"] == "post body"
    assert results[1]["user_id"] == 5
    assert results[1]["node_id"] == 1


def test_thread_hydration_never_returns_hidden_opening_post(metadata_pool):
    metadata_pool.connection.connection.execute("UPDATE xf_post SET message_state = 'deleted' WHERE post_id = 7")
    results = asyncio.run(db.fetch_details_for_posts_and_threads(metadata_pool, [{"thread_id": 8}]))
    assert len(results) == 1
    assert results[0]["message"] == ""


def test_permission_revocation_is_observed_on_next_hydration(metadata_pool):
    items = [{"post_id": 7}]
    assert len(asyncio.run(db.fetch_details_for_posts_and_threads(metadata_pool, items))) == 1
    metadata_pool.connection.connection.execute("DELETE FROM xf_permission_cache_content")
    assert asyncio.run(db.fetch_details_for_posts_and_threads(metadata_pool, items)) == []
    assert asyncio.run(db.fetch_details_by_record_ids(metadata_pool, [1, 2])) == {}
    assert asyncio.run(db.search_mariadb_native_vector(metadata_pool, np.array([1, 0, 0]))) == []
