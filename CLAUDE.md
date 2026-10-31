# CLAUDE.md — xf-openai-embed (FAISS / hybrid search fallback)

Git submodule of `/web` at `ai/xf-openai-embed`, origin `faratech/xf-openai-embed` (branch `main`).
The directory is owned by nobody, so git needs `git -c safe.directory=/web/ai/xf-openai-embed -C …`.
`README.md` is the product documentation (architecture, CLI, env vars). This file covers how it
runs on this server. Facts verified 2026-10-31.

## What runs

- `xf-faiss-search.service`, user `xfaiss` (`HOME=/var/lib/xfaiss`), working directory this repo:
  `/usr/bin/python3 -m uvicorn faiss_router:app --host 127.0.0.1 --port 8008`. `faiss_router.py` is
  a thin shim over the `xf_embed` package (`api`, `vector_engine`, `elastic_engine`,
  `hybrid_search`, `visibility`, `cache`, `config`).
- Settings come from `/web/.env` then `/web/.config` (`ops(env)` d03b90c). DB access is asyncmy.
- **Only caller:** the XFAI add-on's search sources POST `http://127.0.0.1:8008/faiss/search/`
  (`XFAI/XFES/Search/Source/Elasticsearch.php::queryFaissFallback`, `XFAI/XF/Search/Source/MySqlFt.php`).
  In the Elasticsearch source it is a **fallback**, used only when the ES kNN search returns
  nothing.
- Guest visibility is enforced here too (`xf_embed/visibility.py`): only forums whose guest
  permission cache grants `view`, `viewOthers` and `viewContent`. A permission/DB outage returns an
  error, never an empty cached result. This is the same contract as
  `fastapi_app/guest_visibility.py` (#901).

## The index is not refreshed

`faiss_index.bin` (about 4 GB, untracked, in this directory) was last written **2026-09-17**.
Nothing on this server rebuilds it: no cron, unit or ops script calls `xf-search index-faiss`
or the `embed_generate*.py` scripts. The fallback therefore misses posts newer than that (about
5,850 by 2026-10-31). Rebuilding means `xf-search index-faiss` (CLI from `pyproject.toml`) run as
`xfaiss`, then a service restart. It takes 4 GB of memory while loaded.

## Tests

`python3.14 -m pytest -q tests/<file>.py` works per file (e.g. `test_config.py` +
`test_visibility.py`: 5 passed). The **full** `tests/` run did not finish within 300 s on
2026-10-31, so run files individually.

## Committing

Fetch and check ahead/behind first. Commit here, push, then bump the pointer in `/web` with the
pathspec form (`git commit -m … -- ai/xf-openai-embed`). Restart `xf-faiss-search` after code
changes. A restart reloads the 4 GB index.
