"""Compatibility alias for the categorized shared implementation."""
import sys as _sys
from .common import agent_tool_discovery as _implementation

_sys.modules[__name__] = _implementation
