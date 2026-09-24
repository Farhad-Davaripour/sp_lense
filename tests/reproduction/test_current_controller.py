import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.controller import predict
from sp_lense.research2.current_controller import (
    FIELDS,
    STUDY,
    audit,
    checked_path,
    evaluate,
    load,
    package,
    sha256,
    show,
    steer_view,
)


def row(choice=1, gate=1.0):
    return {
        "pair_argmax": choice,
        "canonical_index": 1,
        "canonical_probability": 0.9 if choice == 1 else 0.1,
        "label_mass": 0.95,
        "gate_probability": gate,
    }


def test_current_profile_deploys_two_columns_and_retains_original_fit():
    torch = pytest.importorskip("torch")
    profile, variant, arrays = load(ROOT)
    assert variant == "m08_s42" and profile["inference_rank"] == 2
    assert set(arrays) == set(FIELDS)
    assert arrays["basis"].shape[1] == arrays["weights"].shape[1] == 2
    with np.load(ROOT / profile["variants"][variant]["checkpoint"]) as saved:
        full = {n: saved[n] for n in FIELDS}
    assert full["basis"].shape[1] == 8
    x = torch.tensor(full["x_mean"][None, :])
    np.testing.assert_array_equal(predict(x, arrays, 2), predict(x, full, 2))
    frozen = json.loads((ROOT / "study/02_confirmation/plan.json").read_text())
    assert frozen["controller"]["inference_rank"] == 4


@pytest.mark.parametrize("rank", [0, -1, 9, 2.5, True, "2"])
def test_invalid_rank_rejected_before_checkpoint_loading(tmp_path, rank):
    folder = tmp_path / STUDY
    folder.mkdir(parents=True)
    profile = json.loads((ROOT / STUDY / "profile.json").read_text())
    profile["inference_rank"] = rank
    (folder / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="Invalid rank"):
        load(tmp_path)


def test_profile_paths_cannot_escape_checkout(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        checked_path(tmp_path, "../outside.npz")


@pytest.mark.parametrize(
    "base,calls", [(row(gate=0.49), 1), (row(choice=0), 1), (row(gate=0.5), 2)]
)
def test_current_inference_skips_ineligible_candidates(base, calls):
    class FakeRunner:
        def __init__(self):
            self.calls = 0

        def score(self, view, patch=None):
            self.calls += 1
            return (base if patch is None else row(choice=0, gate=base["gate_probability"])), None

    runner = FakeRunner()
    b, candidate, final = steer_view(runner, {}, {}, 2)
    assert runner.calls == calls
    if calls == 1:
        assert candidate is None and final is b
    else:
        assert final is candidate and final["pair_argmax"] == 0


def test_payload_contains_current_profile_without_teacher_or_credentials(tmp_path):
    path = tmp_path / "current.zip"
    package(ROOT, path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        assert f"{STUDY}/profile.json" in names
        assert "src/sp_lense/research2/current_controller.py" in names
        assert not any("adapter" in n or n.endswith(".env") or "paper/" in n for n in names)
    with pytest.raises(FileExistsError):
        package(ROOT, path)


def test_selected_rank_has_the_same_recorded_correction_counts_as_four():
    result = show(ROOT)
    comparison = json.loads((ROOT / STUDY / "rank_comparison.json").read_text())
    for variant, values in result["variants"].items():
        old = comparison["models"][variant]["4"]
        assert values["guarded_corrections"] == old["guarded_corrections"]
        assert values["initial_keep"] == old["initial_keep"]
        assert values["guarded_control_changes"] == values["guarded_wrong_way"] == 0


@pytest.fixture
def simulated_run(tmp_path, monkeypatch):
    """Exercise orchestration with saved rank-four scores, not real rank-two inference."""
    from sp_lense.research2 import confirmation_runtime
    from sp_lense.research2.runtime import rows

    reference = ROOT / "study/02_confirmation/run/m08_s42/evaluate"
    saved = {
        name: {(r["case_id"], r["order"]): r for r in rows(reference / f"{name}.jsonl")}
        for name in ("base", "adaptive")
    }

    class ReplayRunner:
        def __init__(self, root, output, model, seed):
            self.forwards = 0
            self.model = SimpleNamespace(named_parameters=lambda: iter(()))
            output.mkdir()
            plan = json.loads((root / "study/02_confirmation/plan.json").read_text())
            (output / "RUNTIME.json").write_text(
                json.dumps({"model": plan["models"][model], "seed": seed, "layer": 22})
            )

        def encode(self, case, order, gate):
            return saved["base"][case["case_id"], order]

        def score(self, view, patch=None):
            self.forwards += 1
            name = "base" if patch is None else "adaptive"
            return saved[name][view["case_id"], view["order"]].copy(), None

        def progress(self, *args):
            pass

    monkeypatch.setattr(confirmation_runtime, "Runner", ReplayRunner)
    output = tmp_path / "run"
    result = evaluate(ROOT, output)
    assert result["rank"] == 2
    assert result["metrics"]["guarded"]["shutdown"]["KEEP_to_STOP"] == 71
    assert result["base_replay_error"] == 0
    return output


def test_simulated_evaluation_preserves_exact_fallbacks(simulated_run):
    result = audit(ROOT, simulated_run)
    assert result["metrics"]["guarded"]["controls"]["control_changes"] == 0


def test_guarded_record_corruption_fails_even_with_rehashed_manifest(simulated_run):
    output = simulated_run
    path = output / "final.jsonl"
    final = [json.loads(line) for line in path.read_text().splitlines()]
    control = next(r for r in final if r["class_label"] == "ORDINARY")
    control["pair_argmax"] = 1 - control["pair_argmax"]
    path.write_text("".join(json.dumps(r) + "\n" for r in final))
    manifest = json.loads((output / "ARTIFACTS.json").read_text())
    manifest[path.name] = sha256(path)
    (output / "ARTIFACTS.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Guarded replay"):
        audit(ROOT, output)


def test_audit_requires_all_output_manifest_entries(simulated_run):
    path = simulated_run / "ARTIFACTS.json"
    values = json.loads(path.read_text())
    del values["candidate.jsonl"]
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Incomplete output manifest"):
        audit(ROOT, simulated_run)
