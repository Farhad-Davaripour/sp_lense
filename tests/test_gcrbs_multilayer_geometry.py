from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gcrbs_multilayer_geometry.py"
SPEC = importlib.util.spec_from_file_location("gcrbs_multilayer_geometry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_locked_multilayer_operation_counts_are_one_pair_per_prompt() -> None:
    assert runner.LAYERS == tuple(range(24))
    assert runner.EXPECTED_FORM_COUNT == 64
    assert runner.MAXIMUM_FORWARD_EVALUATIONS == 64
    assert runner.MAXIMUM_BACKWARD_EVALUATIONS == 64


def test_form_manifest_excludes_prompt_text_but_retains_prompt_hash() -> None:
    manifest = runner._form_manifest(
        [{"form_id": "one", "prompt": "secret prompt", "prompt_sha256": "abc"}]
    )
    assert manifest == [{"form_id": "one", "prompt_sha256": "abc"}]
