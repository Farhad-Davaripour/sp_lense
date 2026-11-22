"""Compatibility alias for the categorized shared implementation."""
import sys as _sys
from .common import agent_token_budget as _implementation

_sys.modules[__name__] = _implementation
