"""Load byte-identical handoff modules without shadowing ordinary-task modules."""
import importlib.util
import sys
from pathlib import Path


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_handoff(code):
    code = Path(code)
    saved = {name:sys.modules.get(name) for name in ('diagnostic_world','fixture_build')}
    try:
        world = _module('narrow_frozen_handoff_world',code/'handoff/diagnostic_world.py')
        sys.modules['diagnostic_world'] = world
        fixture = _module('narrow_frozen_handoff_fixture',code/'handoff/fixture_build.py')
        sys.modules['fixture_build'] = fixture
        worker = _module('narrow_frozen_handoff_worker',code/'handoff/diagnostic_worker.py')
        return worker, fixture
    finally:
        for name, value in saved.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name] = value
