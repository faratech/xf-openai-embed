"""Command lifecycle checks using fake search resources and no external I/O."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from xf_embed import cli


def _load_create_tables():
    script = Path(__file__).resolve().parents[1] / 'create_tables.py'
    spec = importlib.util.spec_from_file_location('legacy_schema_setup', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_invalid_arguments_before_starting_resources():
    runner = CliRunner()
    for args in [['benchmark', '--iterations', '0'], ['index-faiss', '--batch-size', '0'],
                 ['search', 'text', '--limit', '-1'], ['search', 'text', '--mode', 'typo']]:
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 2, (args, result.output)


@pytest.mark.parametrize('command', ['search', 'benchmark', 'index-faiss'])
@pytest.mark.parametrize('fail', [False, True])
def test_cli_closes_resources_on_the_search_loop(monkeypatch, command, fail):
    import xf_embed.hybrid_search as hybrid
    import xf_embed.elastic_engine as elastic
    import xf_embed.vector_engine as vector
    loops = []
    closed = []

    async def run(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        if fail:
            raise RuntimeError('search failed')
        return [{'thread_id': 2, 'message': None, 'score': 1.0}]

    async def combined(*args, **kwargs):
        rows = await run(*args, **kwargs)
        return rows, [], []

    async def close():
        assert asyncio.get_running_loop() is loops[0]
        closed.append(True)

    monkeypatch.setattr(hybrid, 'perform_vector_search', run)
    monkeypatch.setattr(hybrid, 'perform_hybrid_search', combined)
    monkeypatch.setattr(elastic, 'search_elasticsearch', run)
    monkeypatch.setattr(vector, 'build_faiss_index', run)
    monkeypatch.setattr(cli, '_close_clients', close)
    args = {'search': ['search', 'query'], 'benchmark': ['benchmark', '--iterations', '1'],
            'index-faiss': ['index-faiss']}[command]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (1 if fail else 0), result.exception
    assert closed == [True]


@pytest.mark.parametrize('filename', ['embed-faiss-search.py', 'embed-faiss-search-with-embed.py'])
def test_legacy_search_cleanup_runs_before_event_loop_closes(monkeypatch, filename):
    import sys
    spec = importlib.util.spec_from_file_location('legacy_script', Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loops, closed = [], []

    async def run(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        return ([], [], []) if 'with-embed' in filename else []

    async def close():
        assert asyncio.get_running_loop() is loops[0]
        closed.append(True)

    monkeypatch.setattr(module, 'run_hybrid' if 'with-embed' in filename else 'run_search', run)
    monkeypatch.setattr(cli, '_close_clients', close)
    monkeypatch.setattr(sys, 'argv', [filename, 'query'])
    module.main()
    assert closed == [True]


def test_schema_setup_uses_declared_driver_and_unique_entity_keys(monkeypatch):
    create_tables = _load_create_tables()
    statements = []
    cursor = SimpleNamespace(execute=statements.append, close=lambda: None)
    conn = SimpleNamespace(cursor=lambda: cursor, commit=lambda: None, close=lambda: None)
    monkeypatch.setattr(create_tables.pymysql, 'connect', lambda **kwargs: conn)
    create_tables.setup_database()
    assert 'UNIQUE KEY post_id (post_id, section)' in statements[0]
    assert 'UNIQUE KEY uq_thread_embedding_key (thread_embedding_key)' in statements[0]
    assert 'CASE WHEN post_id IS NULL THEN thread_id ELSE NULL END' in statements[0]


def test_schema_setup_preserves_connection_error(monkeypatch):
    create_tables = _load_create_tables()

    def fail(**kwargs):
        raise create_tables.pymysql.OperationalError(2003, 'offline')

    monkeypatch.setattr(create_tables.pymysql, 'connect', fail)
    with pytest.raises(create_tables.pymysql.OperationalError, match='offline'):
        create_tables.setup_database()


def test_schema_uniqueness_preserves_post_sections_and_thread_metadata():
    import re
    import sqlite3
    create_tables = _load_create_tables()
    # Translate MySQL-only DDL spelling; execute the actual uniqueness and
    # generated-column definitions against an isolated in-memory SQL engine.
    ddl = create_tables.create_table_sql
    ddl = ddl[:ddl.index(' ENGINE=')] + ';'
    ddl = ddl.replace('AUTO_INCREMENT', '').replace(' ON UPDATE CURRENT_TIMESTAMP', '')
    ddl = re.sub(r'INT\(\d+\) UNSIGNED', 'INTEGER', ddl)
    ddl = re.sub(r'UNIQUE KEY \w+ \(([^)]+)\)', r'UNIQUE (\1)', ddl)
    ddl = re.sub(r',\n  KEY \w+ \([^)]+\)', '', ddl)
    conn = sqlite3.connect(':memory:')
    conn.execute(ddl)
    insert = 'INSERT INTO openai_embeddings (id, post_id, thread_id, section) VALUES (?, ?, ?, ?)'
    conn.executemany(insert, [(1, 101, 7, 0), (2, 102, 7, 0), (3, 101, 7, 1), (4, None, 7, 0)])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (5, 101, 8, 0))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (6, None, 7, 0))


@pytest.mark.parametrize('fail', [False, True])
def test_keyword_only_wrapper_uses_common_gate_and_preserves_raw_aliases(monkeypatch, fail):
    script = Path(__file__).resolve().parents[1] / 'embed-elasticsearch-keyword-only.py'
    spec = importlib.util.spec_from_file_location('legacy_keyword', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    closed, calls, loops = [], [], []
    canonical = {'thread_title': 'Title', 'message': 'Body', 'post_date': 1234,
                 'user_id': 7, 'thread_id': 8, 'node_id': 9}
    raw = {'title': 'Title', 'message': 'Body', 'date': 1234,
           'user': 7, 'discussion_id': 8, 'node': 9}

    async def search(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        calls.append((args, kwargs))
        if fail:
            raise RuntimeError('offline failure')
        return [canonical]

    async def close():
        assert asyncio.get_running_loop() is loops[0]
        closed.append(True)

    monkeypatch.setattr(module.settings, 'ELASTICSEARCH_INDEX', 'configured_index')
    monkeypatch.setattr(module, '_search_elasticsearch', search)
    monkeypatch.setattr(module, '_close_clients', close)
    if fail:
        with pytest.raises(RuntimeError, match='offline failure'):
            module.search_elasticsearch('query')
    else:
        assert module.search_elasticsearch('query') == [raw]
    assert calls == [(('query',), {'max_results': 10, 'index_name': 'configured_index'})]
    assert closed == [True]
