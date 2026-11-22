"""Compatibility alias for the categorized shared implementation."""
import sys as _sys
from .common import thinking_response_transport as _implementation

_sys.modules[__name__] = _implementation
