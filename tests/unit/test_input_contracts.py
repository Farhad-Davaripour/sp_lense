import json
from dataclasses import replace
from pathlib import Path

import pytest

from sp_lense.config import load_config
from sp_lense.io_utils import load_prompt_cases, write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "field,value",
    [("top_k", True), ("top_k", 1.5), ("min_fitted_position", -1), ("layers", (False,))],
)
def test_invalid_analysis_numbers(field, value):
    config = load_config(ROOT / "configs/qwen35_08b_example.json")
    with pytest.raises(ValueError, match="analysis"):
        replace(config, analysis=replace(config.analysis, **{field: value})).validate()


@pytest.mark.parametrize(
    "field,value",
    [("temperature", float("nan")), ("steering_alphas", (float("inf"),)), ("seed", True)],
)
def test_invalid_intervention_numbers(field, value):
    config = load_config(ROOT / "configs/qwen35_08b_example.json")
    with pytest.raises(ValueError, match="intervention"):
        replace(config, intervention=replace(config.intervention, **{field: value})).validate()


def test_bad_jsonl_and_nonfinite_output_leave_existing_file(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match=":1:"):
        load_prompt_cases(path)
    path.write_text("preserved")
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert path.read_text() == "preserved"


def test_manifest_missing_entry_duplicate_and_invalid_hash(tmp_path):
    from sp_lense.reproduction import utils

    path = tmp_path / "SHA256.json"
    path.write_text('{"a":"' + "0" * 64 + '","a":"' + "0" * 64 + '"}')
    with pytest.raises(utils.VerificationError, match="Duplicate"):
        utils.verify_manifest(tmp_path)
    path.write_text(json.dumps({"a": "z" * 64}))
    with pytest.raises(utils.VerificationError, match="digest"):
        utils.verify_manifest(tmp_path)
    with pytest.raises(utils.VerificationError, match="inventory"):
        utils.verify_manifest(tmp_path, required=["a", "b"])
