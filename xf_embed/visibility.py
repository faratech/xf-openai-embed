"""Authoritative XenForo guest permissions for unauthenticated search.

Read XenForo's computed permission cache, rather than reconstructing inheritance
or maintaining a host-specific denylist. Permissions are loaded per operation;
an earlier successful lookup never authorizes a later failed lookup.
"""

import json
from typing import Optional

import asyncmy.cursors


class VisibilityUnavailable(RuntimeError):
    """Guest permissions could not be verified; callers must not serve content."""


def _guest_can_read_forum(raw) -> bool:
    try:
        permissions = json.loads(raw)
    except (TypeError, ValueError, UnicodeError):
        return False
    return isinstance(permissions, dict) and all(
        permissions.get(permission) is True for permission in ("view", "viewOthers", "viewContent")
    )


async def guest_visible_node_ids(pool: Optional[asyncmy.Pool] = None) -> frozenset[int]:
    """Read current guest-readable forums with one metadata-only query.

    Empty/malformed node caches never grant access. Database failures propagate
    as an explicit error so callers cannot cache an outage as an empty search.
    No permission snapshot or fallback denylist is retained between requests.
    """
    query = """
        SELECT n.node_id, n.node_type_id, c.cache_value
        FROM xf_permission_combination pc
        CROSS JOIN xf_node n
        LEFT JOIN xf_permission_cache_content c
            ON c.content_type = 'node' AND c.content_id = n.node_id
            AND c.permission_combination_id = pc.permission_combination_id
        WHERE pc.user_id = 0 AND pc.user_group_list = %s
    """
    try:
        if pool is None:
            from xf_embed.db import get_db_pool
            pool = await get_db_pool()
        async with pool.acquire() as connection:
            async with connection.cursor(asyncmy.cursors.DictCursor) as cursor:
                await cursor.execute(query, ("1",))
                rows = await cursor.fetchall()
    except Exception as exc:
        raise VisibilityUnavailable("Could not verify XenForo guest permissions") from exc
    if not rows:
        raise VisibilityUnavailable("XenForo guest permission combination or node tree is unavailable")

    allowed = set()
    for row in rows:
        # XenForo stores this column with a binary collation; asyncmy can
        # therefore return bytes even with an utf8mb4 connection charset.
        node_type = row.get("node_type_id")
        if isinstance(node_type, (bytes, bytearray)):
            try:
                node_type = node_type.decode("ascii")
            except UnicodeError:
                continue
        if node_type != "Forum" or not _guest_can_read_forum(row.get("cache_value")):
            continue
        try:
            node_id = int(row["node_id"])
        except (ValueError, TypeError, KeyError):
            continue
        if node_id > 0:
            allowed.add(node_id)
    return frozenset(allowed)
