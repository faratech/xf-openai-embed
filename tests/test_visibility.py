"""Guest visibility trusts only XenForo's current computed permission cache."""

import asyncio
import json

import pytest

from xf_embed import visibility


class Context:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


class MetadataPool:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def acquire(self):
        return Context(self)

    def cursor(self, *args):
        return Context(self)

    async def execute(self, query, params):
        self.calls.append((query, params))

    async def fetchall(self):
        return self.rows


def test_computed_guest_permissions_require_all_three_flags_and_forum_type():
    permissions = {"view": True, "viewOthers": True, "viewContent": True}
    rows = [{"node_id": 1, "node_type_id": b"Forum", "cache_value": json.dumps(permissions).encode()}]
    for node_id, changed in enumerate([
        {"view": False}, {"viewOthers": False}, {"viewContent": False},
        {"view": "true"}, {"view": 1}, {"view": None},
    ], start=2):
        rows.append({"node_id": node_id, "node_type_id": "Forum", "cache_value": json.dumps({**permissions, **changed})})
    rows.extend([
        {"node_id": 10, "node_type_id": "Forum", "cache_value": None},
        {"node_id": 11, "node_type_id": "Forum", "cache_value": "malformed JSON"},
        {"node_id": 12, "node_type_id": "Forum", "cache_value": "{}"},
        {"node_id": 13, "node_type_id": "Category", "cache_value": json.dumps(permissions)},
        {"node_id": 14, "node_type_id": "Forum", "cache_value": "[]"},
        {"node_id": 15, "node_type_id": "Forum", "cache_value": b"\xff"},
    ])
    pool = MetadataPool(rows)
    assert asyncio.run(visibility.guest_visible_node_ids(pool)) == frozenset({1})
    assert len(pool.calls) == 1
    assert pool.calls[0][1] == ("1",)
    assert "pc.user_id = 0" in pool.calls[0][0]
    assert "xf_permission_cache_content" in pool.calls[0][0]


def test_missing_guest_combination_is_unavailable_and_cannot_cache_empty_success():
    with pytest.raises(visibility.VisibilityUnavailable, match="combination or node tree"):
        asyncio.run(visibility.guest_visible_node_ids(MetadataPool([])))


def test_permission_outage_never_reuses_previous_allowlist():
    class FailingPool(MetadataPool):
        async def execute(self, *args):
            raise RuntimeError("offline")

    permissions = json.dumps({"view": True, "viewOthers": True, "viewContent": True})
    pool = MetadataPool([{"node_id": 1, "node_type_id": "Forum", "cache_value": permissions}])
    assert asyncio.run(visibility.guest_visible_node_ids(pool)) == {1}
    with pytest.raises(visibility.VisibilityUnavailable, match="Could not verify"):
        asyncio.run(visibility.guest_visible_node_ids(FailingPool([])))


def test_verified_nodes_without_guest_access_return_empty_allowlist():
    pool = MetadataPool([{"node_id": 1, "node_type_id": "Forum", "cache_value": "{}"}])
    assert asyncio.run(visibility.guest_visible_node_ids(pool)) == frozenset()
