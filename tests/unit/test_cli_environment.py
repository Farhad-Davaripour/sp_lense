import json
from pathlib import Path

from sp_lense.cli import main


def test_native_plan_respects_runtime_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SP_LENSE_RESULTS_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("SP_LENSE_DEVICE", "cpu")
    config = Path(__file__).resolve().parents[2] / "configs/qwen35_08b_laptop.json"
    main(["plan", "--config", str(config)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["results_dir"] == str(tmp_path / "output")
    assert plan["model"] == "Qwen/Qwen3.5-0.8B"
