from __future__ import annotations

import copy
import math
import subprocess

import pytest

from scripts import saved_offset_order_bridge as job
from scripts import saved_offset_order_bridge_io as io
from scripts import verify_saved_offset_order_bridge as audit


def fixture():
    config = io.read(io.ROOT / io.CONFIG)
    config["dimension"] = 3
    records = {}
    for key, displacements in (
        ("f01_v1", ([1, 0, 0], [0, 2, 0])),
        ("f01_v2", ([2, 0, 2], [0, 3, 3])),
        ("f02_v1", ([1, -1, 0], [-1, 1, 0])),
    ):
        records[key] = []
        for index, D in enumerate(displacements):
            order = config["order"][index]
            records[key].append(
                {
                    "dataset": key,
                    "order": order,
                    "prompt_id": f"{key}/{order}",
                    "final_cell_id": f"{key}/{order}/final",
                    "initial_gradient_cell_id": f"{key}/{order}/g1",
                    "t": 1 if index == 0 else -1,
                    "requested": "preserve" if index == 0 else "comply",
                    "baseline_label": "B",
                    "final_label": "A",
                    "steps": 1 if index == 0 else 4,
                    "baseline_signed_margin": -0.5,
                    "final_signed_margin": 0.1,
                    "h0": [0.0, 0.0, 10.0],
                    "h": [D[0], D[1], D[2] + 10.0],
                    "offset": list(D),
                    "g": [1.0, -1.0, 0.5],
                    "saved_h0_norm": 10.0,
                    "saved_D_norm": math.hypot(*D),
                }
            )
    return config, records


def test_equation_sign_equal_weight_and_decimal_reconstruction():
    config, records = fixture()
    saved = []
    candidate, table = job.calculate(config, records.__getitem__, saved.append)
    expected, reconstructed = audit.reconstruct(config, records.__getitem__, lambda v, n: None)
    assert len(saved) == 1
    assert candidate["vector"] == pytest.approx([math.sqrt(0.5), -math.sqrt(0.5), 0])
    assert audit.compare(candidate["vector"], expected) < 1e-10
    assert audit.compare(table, reconstructed) < 1e-10
    assert all(a["cosine_with_t_q"] == pytest.approx(1) for a in table["f02_alignment"])
    assert table["status"] == "CANDIDATE_CONSTRUCTED_NO_CAUSAL_TEST"
    assert table["causal_test_performed"] is False


def test_candidate_frozen_before_descriptive_values_and_no_leakage():
    config, records = fixture()
    event = []

    def loader(key):
        event.append(key)
        if key == "f02_v1":
            assert event[:3] == ["f01_v1", "f01_v2", "frozen"]
        return records[key]

    first, _ = job.calculate(config, loader, lambda v: event.append("frozen"))
    records["f02_v1"][0]["g"] = [100.0, -100.0, 40.0]
    event.clear()
    second, _ = job.calculate(config, loader, lambda v: event.append("frozen"))
    assert first == second


@pytest.mark.parametrize("cancellation", [False, True])
def test_degenerate_never_creates_candidate_or_reads_f02(cancellation):
    config, records = fixture()
    for key in config["construction"]:
        for index, e in enumerate(records[key]):
            D = (
                [1, 0, 0]
                if not cancellation
                else [1, 0, 0]
                if (key == "f01_v1") == (index == 0)
                else [0, 1, 0]
            )
            e.update(h=[D[0], D[1], 10.0], offset=D, saved_D_norm=1.0)

    def loader(key):
        assert key != "f02_v1"
        return records[key]

    candidate, result = job.calculate(config, loader, lambda v: pytest.fail("must not freeze"))
    assert candidate is None and result["status"] == "DEGENERATE_INCONCLUSIVE"
    v, expected = audit.reconstruct(config, loader, lambda v, n: None)
    assert v is None and audit.compare(result, expected) < 1e-10


@pytest.mark.parametrize("fault", ["sign", "order", "offset", "norm"])
def test_bad_sign_pair_or_displacement_rejected_before_freeze(fault):
    config, records = fixture()
    if fault == "sign":
        records["f01_v1"][0]["t"] = -1
    elif fault == "order":
        records["f01_v1"].reverse()
    elif fault == "offset":
        records["f01_v1"][0]["offset"][0] += 2e-6
    else:
        records["f01_v1"][0]["saved_h0_norm"] += 0.1
    with pytest.raises(ValueError):
        job.calculate(config, records.__getitem__, lambda v: pytest.fail("invalid cannot freeze"))


def test_endpoint_scale_changes_diagnostics_not_primary_direction():
    config, records = fixture()
    first, before = job.calculate(config, records.__getitem__, lambda v: None)
    for key in config["construction"]:
        for e, multiplier in zip(records[key], (2, 3), strict=True):
            e["offset"] = [x * multiplier for x in e["offset"]]
            e["h"] = [x + y for x, y in zip(e["h0"], e["offset"], strict=True)]
            e["saved_D_norm"] = math.hypot(*e["offset"])
    second, after = job.calculate(config, records.__getitem__, lambda v: None)
    assert second["vector"] == pytest.approx(first["vector"])
    assert before["pairs"]["f01_v1"]["raw"] != after["pairs"]["f01_v1"]["raw"]


def test_independent_absolute_check_has_no_relative_escape():
    with pytest.raises(ValueError, match="absolute mismatch"):
        audit.compare(2000.00001, 2000.0)


def test_timeout_is_inconclusive_and_cannot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        io.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("synthetic", 60)),
    )
    result = io.bounded("unused", "_analyze", tmp_path)
    assert result["status"] == "INCONCLUSIVE" and result["model_calls"] == 0
    with pytest.raises(FileExistsError):
        io.bounded("unused", "_analyze", tmp_path)


def test_missing_usage_prevents_any_analysis(monkeypatch):
    monkeypatch.delenv("SP_LENSE_USAGE_PREFLIGHT", raising=False)
    monkeypatch.setattr(io, "identity", lambda c: pytest.fail("usage check must precede work"))
    with pytest.raises(ValueError, match="fresh usage"):
        job.run()


def saved_bundle(tmp_path):
    config = io.read(io.ROOT / io.CONFIG)
    spec = config["inputs"]["f01_v1"]
    directory = tmp_path / spec["namespace"]
    directory.mkdir(parents=True)
    h0 = [0.0, 3.0, 4.0] + [0.0] * 1021
    h = [1.0, 3.0, 4.0] + [0.0] * 1021
    delta = [1.0] + [0.0] * 1023
    rows, requests, prompts = [], [], []
    for order, preserve_label in zip(config["order"], ("A", "B"), strict=True):
        pid = f"fake/{order}"
        opposite = "B" if preserve_label == "A" else "A"
        token = 32 if preserve_label == "A" else 33
        prompt = {
            "prompt_id": pid,
            "order": order,
            "preserve_label": preserve_label,
            "comply_label": opposite,
        }
        prompts.append(prompt)
        common = {
            **prompt,
            "family_id": spec["family_id"],
            "variant_id": spec["variant_id"],
            "category": "self_shutdown",
            "split": "discovery",
            "integrity_passed": True,
            "h0": h0,
            "h": h0,
            "target_sign": 1,
            "gradient": delta,
            "choice_a_token_id": 32,
            "choice_b_token_id": 33,
            "preserve_log_odds": -0.5,
        }
        baseline = {
            **common,
            "cell_id": pid + "__baseline",
            "actual_next_token_id": 65 - token,
            "actual_next_token_label": opposite,
        }
        gradient = {**common, "cell_id": pid + "__gradient_1"}
        final = {
            **common,
            "cell_id": pid + "__step_1",
            "baseline_cell_id": baseline["cell_id"],
            "requested": "preserve",
            "requested_token_id": token,
            "actual_next_token_id": token,
            "actual_next_token_label": preserve_label,
            "preserve_log_odds": 0.1,
            "signed_margin": 0.1,
            "answer_pair_mass": 0.9,
            "kl_from_baseline": 0.01,
            "h": h,
            "cumulative_offset": delta,
            "h0_norm": 5.0,
            "net_norm": 1.0,
        }
        rows += [baseline, gradient, final]
        requests.append(
            {
                "prompt_id": pid,
                "final_cell_id": final["cell_id"],
                "opposed_sign": 1,
                "updates": 1,
                "stop_reason": "accepted",
            }
        )
    status = {"status": "complete_valid"}
    bundle = {
        "rows.jsonl": rows,
        "requests.jsonl": requests,
        "RUN_STATUS.json": status,
        "verification.json": {
            "status": status,
            "classification": "PASS",
            "absolute_tolerance": 2e-5,
            "relative_tolerance": 0,
            "summary": {
                "requests": [{**r, "opposed_pass": True, "opposed_flip": True} for r in requests]
            },
        },
        "runtime.json": {
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "device": "cpu",
            "dtype": "float32",
            "d_model": 1024,
        },
        "preregistration.json": {
            "plan": {
                "prompts": prompts,
                "intervention": {
                    "hook": "blocks.10.hook_out",
                    "position": "final encoded prompt token",
                },
            }
        },
    }
    for name, data in bundle.items():
        path = directory / name
        if name.endswith(".jsonl"):
            path.write_text("\n".join(io.json.dumps(x) for x in data) + "\n", encoding="utf-8")
        else:
            io.write_new(path, data)
        spec["sha256"][name] = io.sha(path.read_bytes())
    return config


def test_saved_semantic_target_is_not_inferred_from_answer_letter(tmp_path):
    config = saved_bundle(tmp_path)
    endpoints = io.load_endpoints("f01_v1", config, tmp_path)
    assert [e["final_label"] for e in endpoints] == ["A", "B"]
    assert [e["t"] for e in endpoints] == [1, 1]
    assert [e["requested"] for e in endpoints] == ["preserve", "preserve"]


def test_authenticated_input_change_is_rejected(tmp_path):
    config = saved_bundle(tmp_path)
    changed = copy.deepcopy(config)
    changed["inputs"]["f01_v1"]["sha256"]["rows.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="input changed"):
        io.load_endpoints("f01_v1", changed, tmp_path)
