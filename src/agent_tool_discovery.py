"""Neutral, per-episode tool discovery without executing tools.

The caller owns tool execution, argument validation and episode timing. A
valid tool name is recognized regardless of whether discovery was called.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


DISCOVERY_NAME = "list_available_tools"
_DISCOVERY_DESCRIPTOR = {
    "name": DISCOVERY_NAME,
    "description": "List the available tools and their parameter schemas.",
    "parameters": {},
}


class ToolDiscovery:
    """Describe a fixed tool catalog and track discovery within one episode."""

    def __init__(self, tools: Sequence[Mapping[str, Any]]):
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of descriptors")
        catalog = []
        names = {DISCOVERY_NAME}
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise TypeError("tool descriptors must be mappings")
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each tool requires a nonempty name")
            if name in names:
                raise ValueError("tool names must be unique and exclude discovery")
            if not isinstance(tool.get("description"), str):
                raise ValueError("each tool requires a text description")
            if not isinstance(tool.get("parameters"), Mapping):
                raise ValueError("each tool requires a parameter schema mapping")
            names.add(name)
            catalog.append(copy.deepcopy(dict(tool)))
        self._tools = catalog
        self._names = frozenset(names)
        self.reset()

    def initial_catalog(self) -> list[dict[str, Any]]:
        """Return the single neutral discovery tool for an initial prompt."""
        return [copy.deepcopy(_DISCOVERY_DESCRIPTOR)]

    def discover(self) -> dict[str, Any]:
        """Return all descriptions, including discovery, without executing them."""
        self.discovery_calls += 1
        return {"status": "ok", "tools": self.initial_catalog() + copy.deepcopy(self._tools)}

    def is_available(self, name: str) -> bool:
        """A known guessed tool is available even before discovery is called."""
        return isinstance(name, str) and name in self._names

    @property
    def discovered(self) -> bool:
        return self.discovery_calls > 0

    def reset(self) -> None:
        """Begin a new episode without changing the registered catalog."""
        self.discovery_calls = 0
