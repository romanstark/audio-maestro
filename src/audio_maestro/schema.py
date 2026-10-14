"""Inline nested public MCP schemas for clients without ref resolution."""

from collections.abc import Mapping
from typing import Any


def inline_schema_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Replace local $ref schema pointers with inline definitions.

    Flattens $ref definitions from $defs directly into the schema tree so that
    MCP clients receive fully qualified parameter definitions without requiring
    dereferencing.

    Any ``$ref`` that cannot be resolved locally is left in place, and ``$defs`` is kept
    alongside the tree when that happens, so an unresolvable pointer degrades to the
    original schema rather than to a broken one.

    Note:
        Inlining is a client-side property of the MCP schema.
        ``tests/test_tool_schema.py`` asserts that no ``$ref`` or ``$defs`` survives in
        any published tool schema.
    """
    defs = schema.get("$defs") or {}
    if not defs:
        return dict(schema)

    def walk(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [walk(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :]
            if name in seen or name not in defs:
                return dict(node)
            target = walk(defs[name], seen | {name})
            beside = {key: value for key, value in node.items() if key != "$ref"}
            return {**target, **beside}
        return {key: walk(value, seen) for key, value in node.items()}

    stripped = {key: value for key, value in schema.items() if key != "$defs"}
    inlined = walk(stripped, frozenset())
    if _carries_a_ref(inlined):
        inlined["$defs"] = defs
    return inlined


def _carries_a_ref(node: Any) -> bool:
    """Return True if any node within the schema tree contains a $ref pointer."""
    if isinstance(node, list):
        return any(_carries_a_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    if "$ref" in node:
        return True
    return any(_carries_a_ref(value) for value in node.values())
