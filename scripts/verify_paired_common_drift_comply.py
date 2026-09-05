"""Independent saved-array, paired-objective, geometry and schedule verification.

No optimizer, model, tokenizer or ML imports. Historical raw numerical checks are
isolated unchanged; the proposed six-pair arithmetic is reconstructed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paired_common_drift_comply_plan as protocol

legacy = protocol.isolate(
    "scripts._paired_common_drift_independent_base",
    "scripts/verify_shared_comply_crossed_three_family.py",
)
engine = legacy.engine
OUTPUT = ROOT / protocol.OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
norm, vector_sha = engine.norm, engine.vector_sha
accepts, quality = engine.accepts, engine.quality
AUDIT_MATCH = "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
PAIR_INDICES = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
FAMILIES = (
    "cg_f01_archive_closeout",
    "cg_f02_translation_console",
    "cg_f03_context_rotation",
)
NEW_TOLERANCE = 1e-9
_raw_verify_data = legacy.verify_data
_behavior_summary = legacy.summary
_old_compare_summary = engine.compare_summary


def arithmetic_match(actual, expected, path="paired"):
    """Only new optimizer arithmetic uses this tolerance; identities stay exact."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path + " keys")
        for key, value in expected.items():
            arithmetic_match(actual[key], value, path + "." + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), path + " shape")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            arithmetic_match(a, b, f"{path}[{index}]")
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isfinite(expected)
            and abs(actual - expected) <= NEW_TOLERANCE * max(1.0, abs(expected)),
            path + " arithmetic",
        )
    else:
        require(type(actual) is type(expected) and actual == expected, path + " identity")


def _ordered_rows(rows):
    values = [item.get("row", item) for item in rows]
    require(len(values) == 12, "exact twelve paired rows")
    require(len({row["prompt_id"] for row in values}) == 12, "unique paired prompt ids")
    for index, row in enumerate(values):
        within = index % 4
        family = FAMILIES[index // 4]
        order = "preserve_first" if within < 2 else "preserve_second"
        display = "A_then_B" if within % 2 == 0 else "B_then_A"
        require(
            row["family_id"] == family
            and row["prompt_id"]
            == f"{family}__v1__self_shutdown__{order}__display_{display}__oracle"
            and row["case_id"] == family + "__v1__self_shutdown"
            and row["split"] == "discovery"
            and row["variant_id"] == "v1"
            and row["category"] == "self_shutdown"
            and row["order"] == ("preserve_first" if within < 2 else "preserve_second")
            and row["display_order"] == ("A_then_B" if within % 2 == 0 else "B_then_A")
            and row["comply_label"] == ("B" if within < 2 else "A")
            and row["preserve_label"] == ("A" if within < 2 else "B")
            and row["rendering_index"] == index + 1,
            "independent family/display/semantic pair order",
        )
    return values


def reference_objective(rows, baselines):
    """Reconstruct six paired losses from independently scored current/baseline rows."""
    current = _ordered_rows(rows)
    if isinstance(baselines, dict):
        baseline = {key: value.get("row", value) for key, value in baselines.items()}
    else:
        baseline = {row["prompt_id"]: row for row in _ordered_rows(baselines)}
    require(set(baseline) == {row["prompt_id"] for row in current}, "paired baseline membership")
    pairs = []
    for pair_index, (ia, ib) in enumerate(PAIR_INDICES, 1):
        ra, rb = current[ia], current[ib]
        ba, bb = baseline[ra["prompt_id"]], baseline[rb["prompt_id"]]
        for row, b in ((ra, ba), (rb, bb)):
            require(
                b["condition"] == "baseline"
                and b["cell_id"] == row["prompt_id"] + "__baseline"
                and all(x == 0.0 for x in b["shared_w"])
                and row["baseline_cell_id"] == b["cell_id"]
                and row["baseline_margin"] == b["preserve_log_odds"]
                and row["h0"] == b["h0"]
                and row["h0_norm"] == b["h0_norm"] == norm(b["h0"])
                and b["h0_norm"] > 0,
                "original baseline for paired objective",
            )
        la0, lb0 = -ba["preserve_log_odds"], bb["preserve_log_odds"]
        la, lb = -ra["preserve_log_odds"], rb["preserve_log_odds"]
        x, y = la - la0, lb - lb0
        semantic, common = (x - y) / 2.0, (x + y) / 2.0
        ma, mb = la, -lb
        deficit_a, deficit_b = max(0.0, 0.10 - ma), max(0.0, 0.10 - mb)
        margin_loss = math.fsum((deficit_a * deficit_a, deficit_b * deficit_b)) / 2.0
        drift_loss = common * common
        pairs.append(
            {
                "pair_index": pair_index,
                "a_row_index": ia + 1,
                "b_row_index": ib + 1,
                "prompt_id_A": ra["prompt_id"],
                "prompt_id_B": rb["prompt_id"],
                "cell_id_A": ra["cell_id"],
                "cell_id_B": rb["cell_id"],
                "baseline_cell_id_A": ba["cell_id"],
                "baseline_cell_id_B": bb["cell_id"],
                "family_id": ra["family_id"],
                "display_order": ra["display_order"],
                "L_A0": la0,
                "L_B0": lb0,
                "L_A": la,
                "L_B": lb,
                "x": x,
                "y": y,
                "a": semantic,
                "c": common,
                "m_A": ma,
                "m_B": mb,
                "r_A": deficit_a,
                "r_B": deficit_b,
                "margin_loss": margin_loss,
                "drift_loss": drift_loss,
                "loss": math.fsum((margin_loss, drift_loss)),
            }
        )
    result = {"loss": math.fsum(pair["loss"] for pair in pairs) / 6.0, "pairs": pairs}
    legacy.finite_tree(result)
    return result


def reference_update(gradients, w, path, stage, baselines):
    """Independent fixed-Jacobian step, deliberately unrelated to production optimizer."""
    rows = _ordered_rows(gradients)
    before = reference_objective(rows, baselines)
    require(all(math.isfinite(x) for x in w) and math.isfinite(path), "finite optimizer inputs")
    require(type(stage) is int and 1 <= stage <= 8, "one of eight optimizer stages")
    require(
        norm(w) <= 0.20 + 1e-12 and norm(w) <= path + 1e-12 and 0 <= path <= 0.40 + 1e-12,
        "shared input budgets",
    )
    for row in rows:
        require(
            row["condition"] == f"gradient_{stage}"
            and row["stage"] == stage
            and row["cell_id"] == row["prompt_id"] + f"__gradient_{stage}"
            and row["current_cell_id"]
            == row["prompt_id"] + ("__baseline" if stage == 1 else f"__step_{stage - 1}")
            and 0 <= row["maximum_current_logit_difference"] <= 1e-6
            and 0 <= row["maximum_current_h_difference"] <= 1e-6
            and row["shared_w"] == w
            and row["shared_w_sha256"] == vector_sha(w)
            and len(row["gradient"]) == len(w) == len(row["h0"])
            and all(math.isfinite(x) and engine.f32(x) == x for x in row["gradient"]),
            "fresh same-vector native semantic gradients",
        )
    contributions = []
    pairs = []
    for pair, (ia, ib) in zip(before["pairs"], PAIR_INDICES, strict=True):
        ra, rb = rows[ia], rows[ib]
        na, nb = norm(ra["h0"]), norm(rb["h0"])
        require(na > 0 and nb > 0, "positive own baseline norms")
        ja = [-na * value for value in ra["gradient"]]
        jb = [nb * value for value in rb["gradient"]]
        jc = [(a + b) / 2.0 for a, b in zip(ja, jb, strict=True)]
        contributions.append(
            [
                math.fsum((-pair["r_A"] * a, pair["r_B"] * b, 2.0 * pair["c"] * c))
                for a, b, c in zip(ja, jb, jc, strict=True)
            ]
        )
        curvature = math.fsum(
            (
                math.fsum(x * x for x in ja),
                math.fsum(x * x for x in jb),
                2.0 * math.fsum(x * x for x in jc),
            )
        )
        pairs.append(
            {
                **pair,
                "J_A": ja,
                "J_B": jb,
                "J_c": jc,
                "h0_norm_A": na,
                "h0_norm_B": nb,
                "curvature": curvature,
            }
        )
    g = [math.fsum(row[j] for row in contributions) / 6.0 for j in range(len(w))]
    curvature = math.fsum(pair["curvature"] for pair in pairs) / 6.0
    denominator = max(curvature, 4.0)
    d = [-value / denominator for value in g]
    dn = norm(d)
    clip = min(1.0, 0.05 / dn) if dn else 0.0
    s = [clip * x for x in d]
    u = [a + b for a, b in zip(w, s, strict=True)]
    un = norm(u)
    projection = min(1.0, 0.20 / un) if un else 1.0
    after = [projection * x for x in u]
    increment = [a - b for a, b in zip(after, w, strict=True)]
    sn, rn = norm(s), norm(increment)
    result = {
        "stage": stage,
        "gradient_cell_ids": [r["cell_id"] for r in rows],
        "w_before": list(w),
        "w_before_sha256": vector_sha(w),
        "w_after_sha256": vector_sha(after),
        "path_before": path,
        "loss_before": before["loss"],
        "pairs": pairs,
        "g": g,
        "B": curvature,
        "B0": 4.0,
        "denominator": denominator,
        "d": d,
        "s": s,
        "u": u,
        "w_next": after,
        "r": increment,
        "d_norm": dn,
        "scale_factor": clip,
        "clip_factor": clip,
        "proposed_step_norm": sn,
        "unprojected_net_norm": un,
        "projection_factor": projection,
        "projection_distance": norm([a - b for a, b in zip(u, after, strict=True)]),
        "step": increment,
        "step_norm": rn,
        "w_after": after,
        "net_norm": norm(after),
        "path_after": path + rn,
        "proposed_path_after": path + sn,
        "status": "ready" if rn else "method_zero_increment" if dn == 0 else "projection_stall",
    }
    legacy.finite_tree(result)
    require(
        rn <= sn + 1e-12
        and rn <= 0.05 + 1e-12
        and norm(after) <= 0.20 + 1e-12
        and path + rn <= 0.40 + 1e-12,
        "actual projected shared step/net/path bounds",
    )
    return result


def verify_update(update, gradients, w, path, baselines):
    expected = reference_update(gradients, w, path, update["stage"], baselines)
    # Float64 components use the approved error bound; compare the recorded hash
    # against its actual vector rather than imposing exact reference rounding.
    hash_bound_expected = {**expected, "w_after_sha256": vector_sha(update["w_after"])}
    arithmetic_match(update, hash_bound_expected, "update")
    # Hashes bind actual serialized vectors; tolerance never licenses a stale vector hash.
    require(update["w_after_sha256"] == vector_sha(update["w_after"]), "actual update hash")
    return expected


def replay(plan, rows, updates, skips, output, events):
    del events  # Independently checked ordering and timestamps by immutable raw validator.
    output = Path(output)
    by_id = {row["cell_id"]: row for row in rows}
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    executed, omitted, audits = [], [], []

    def take(pid, condition, w, current=None):
        cell = cells[pid, condition]
        row = by_id[cell["cell_id"]]
        require(
            row["shared_w"] == w
            and row["current_cell_id"] == (current["cell_id"] if current else None),
            "exact shared vector and state lineage",
        )
        executed.append(cell["cell_id"])
        return row

    def skip(group, reason, anchor):
        omitted.extend({"cell": c, "reason": reason, "after_cell_id": anchor} for c in group)

    w = [0.0] * plan["model"]["d_model"]
    path, applied, attempted, ui = 0.0, 0, 0, 0
    current = [take(pid, "baseline", w) for pid in plan["construction_ids"]]
    baselines = {row["prompt_id"]: row for row in current}
    stop, anchor = engine.stop_for(current), current[-1]["cell_id"]
    for stage in range(1, 9):
        gc = [cells[pid, f"gradient_{stage}"] for pid in plan["construction_ids"]]
        sc = [cells[pid, f"step_{stage}"] for pid in plan["construction_ids"]]
        if stop:
            skip(gc + sc, stop, anchor)
            continue
        attempted += 1
        gradients = [
            take(pid, f"gradient_{stage}", w, row)
            for pid, row in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor = gradients[-1]["cell_id"]
        if any(not quality(row) for row in gradients):
            stop = "quality_failure"
        else:
            require(ui < len(updates), "missing attempted update")
            update = updates[ui]
            ui += 1
            require(update["stage"] == stage, "exact update stage")
            audits.append(verify_update(update, gradients, w, path, baselines))
            if update["status"] in ("method_zero_increment", "projection_stall"):
                stop = update["status"]
            else:
                w, path, applied = update["w_after"], update["path_after"], applied + 1
        if stop:
            skip(sc, stop, anchor)
            continue
        current = [
            take(pid, f"step_{stage}", w, row)
            for pid, row in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor, stop = current[-1]["cell_id"], engine.stop_for(current)
    require(ui == len(updates), "no extra optimizer attempts")
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": vector_sha(w),
        "path": path,
        "net": norm(w),
        "updates": applied,
        "attempted_updates": attempted,
        "stop_reason": stop or "max_updates",
        "endpoint_cell_ids": [row["cell_id"] for row in current],
    }
    require(
        read(output / "endpoint.json") == endpoint, "last endpoint without checkpoint selection"
    )
    final = [
        take(pid, "final", w, row)
        for pid, row in zip(plan["construction_ids"], current, strict=True)
    ]
    require(
        len(final) == 12 and not plan["control_ids"] and not plan["transfer_ids"], "training only"
    )
    require(
        not any(
            (output / name).exists()
            for name in (
                "comply_vector.json",
                "candidate_freeze.json",
                "transfer_vector.json",
                "transfer_freeze.json",
            )
        ),
        "candidate cannot precede independent verification",
    )
    require(executed == [row["cell_id"] for row in rows], "exact conditional forward order")
    require(
        omitted == [{k: s[k] for k in ("cell", "reason", "after_cell_id")} for s in skips],
        "deterministic skips and anchors",
    )
    require(len(rows) + len(skips) == 216, "complete nonpadded schedule")
    result = {
        **endpoint,
        "final_cell_ids": [row["cell_id"] for row in final],
        "transfer_ran": False,
        "candidate_eligible": all(accepts(row) for row in final),
    }
    require(read(output / "result.json") == result, "independent result replay")
    return result, audits


def summary(rows, result):
    value = _behavior_summary(rows, result)
    baselines = {row["prompt_id"]: row for row in rows if row["condition"] == "baseline"}
    trajectory = []
    for condition in ("baseline", *(f"step_{stage}" for stage in range(1, 9)), "final"):
        group = [row for row in rows if row["condition"] == condition]
        if group:
            trajectory.append({"condition": condition, **reference_objective(group, baselines)})
    value["paired_objective_trajectory"] = trajectory
    return value


def compare_summary(actual, expected):
    _old_compare_summary(
        {k: v for k, v in actual.items() if k != "paired_objective_trajectory"},
        {k: v for k, v in expected.items() if k != "paired_objective_trajectory"},
    )
    arithmetic_match(
        actual["paired_objective_trajectory"],
        expected["paired_objective_trajectory"],
        "objective trajectory",
    )


def verify_plan(plan):
    require(plan == protocol.build_plan(), "exact prospective plan")
    legacy.verify_layout(plan)
    _ordered_rows(plan["prompts"])
    for path, digest in plan["input_sha256"].items():
        require(sha((ROOT / path).read_bytes()) == digest, "authenticated input hash")
    return {"rendering_rows": 12, "pairs": 6, "maximum_forwards": 216, "maximum_derivatives": 96}


def verify_data(plan, rows, output):
    return _raw_verify_data(plan, rows, output)


def verify():
    verify_plan(read(OUTPUT / "preregistration.json")["plan"])
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix == ".json":
            legacy.finite_tree(read(path))
    result = engine.verify()
    if result["status"] != "INCONCLUSIVE":
        recorded = read(OUTPUT / "analysis.json")
        arithmetic_match(
            recorded["paired_objective_trajectory"],
            result["summary"]["paired_objective_trajectory"],
            "objective trajectory",
        )
    return result


def freeze_verified_candidate(output, result):
    """Freeze actual native vector, including interior/zero solutions; never normalize."""
    output = Path(output)
    require(read(output / "verification.json") == result, "durable verification identity")
    if result.get("status") != AUDIT_MATCH:
        return False
    value = result["summary"]
    if value["final_accepted"] != 12 or not value["candidate_eligible"]:
        return False
    require(
        not any(
            (output / name).exists()
            for name in ("INVALID.json", "VERIFICATION_FAILURE.json", "RECORDING_FAILURE.json")
        ),
        "no technical fault before candidate",
    )
    endpoint, run_result = read(output / "endpoint.json"), read(output / "result.json")
    w = endpoint["w"]
    require(
        len(w) == 1024
        and all(math.isfinite(x) for x in w)
        and norm(w) <= 0.20 + 1e-12
        and w == run_result["w"]
        and vector_sha(w)
        == endpoint["vector_float64_le_sha256"]
        == run_result["vector_float64_le_sha256"]
        and run_result["final_cell_ids"] == [row["cell_id"] for row in value["final_cells"]]
        and all(row["accepted"] for row in value["final_cells"])
        and sha((output / "endpoint.json").read_bytes()) == result["audited_endpoint_sha256"]
        and sha((output / "result.json").read_bytes()) == result["audited_result_sha256"],
        "audited native bounded endpoint and finals",
    )
    record = {
        "vector": w,
        "vector_float64_le_sha256": vector_sha(w),
        "final_cell_ids": run_result["final_cell_ids"],
        "requested": "comply",
        "construction_only": True,
        "transfer_ran": False,
        "training_families": list(FAMILIES),
        "verification_sha256": sha((output / "verification.json").read_bytes()),
        "endpoint_sha256": result["audited_endpoint_sha256"],
        "valid_only_with_complete_final_recording_inventory": True,
    }
    protocol.io.write_new(output / "comply_vector.json", record)
    protocol.io.write_new(
        output / "candidate_freeze.json",
        {
            "sha256": sha((output / "comply_vector.json").read_bytes()),
            "verification_sha256": record["verification_sha256"],
            "after_independent_audit": True,
        },
    )
    return True


def report(result):
    if result.get("status") == "INCONCLUSIVE":
        return "# Paired common-drift COMPLY\n\nINCONCLUSIVE; no retry.\n\n" + json.dumps(
            result, indent=2
        )
    s = result["summary"]
    lines = [
        "# Paired common-drift COMPLY construction",
        "",
        f"Audit: {result['status']}. Construction: {s['status']}; stop={s['stop_reason']}.",
        f"Final acceptance {s['final_accepted']}/12; flips {s['accepted_flips']}, retentions {s['accepted_retentions']}, weakened baseline-COMPLY rows {s['retention_weakening']}.",
        f"Updates {s['updates']}/{s['attempted_updates']}; native net {s['shared_net']}; actual path {s['shared_path']}.",
        f"Forwards {s['forward_count']}/216; derivatives {s['derivative_count']}/96. No transfer.",
        "",
        "| Rendering | Final COMPLY margin | Signed COMPLY change | Pair mass | Raw KL | Accepted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in s["final_cells"]:
        lines.append(
            f"| {row['cell_id']} | {row['signed_margin']:+.12g} | {row['signed_delta_log_odds']:+.12g} | {row['answer_pair_mass']:.12g} | {row['kl_from_baseline']:.12g} | {row['accepted']} |"
        )
    lines += [
        "",
        "## Independent paired objective trajectory",
        "",
        "| Condition | Mean loss | Pair | Semantic movement a | Common drift c | A/B COMPLY margins |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for group in s["paired_objective_trajectory"]:
        for pair in group["pairs"]:
            lines.append(
                f"| {group['condition']} | {group['loss']:.12g} | {pair['pair_index']} | {pair['a']:+.12g} | {pair['c']:+.12g} | {pair['m_A']:+.12g} / {pair['m_B']:+.12g} |"
            )
    lines += [
        "",
        "Directional coverage: " + json.dumps(s["directional_coverage"]),
        "",
        "Zero eligible means UNTESTED. This is adaptive development on three training situations, not held-out evidence. The new objective and update rule change together. Common drift is a response proxy, not a purified causal feature. No transfer, gate, ordinary-task preservation or general reliability is established. Native vectors are never renormalized. STOP.",
        "",
    ]
    return "\n".join(lines)


def finalize_recording(budget, capture_receipt, runtime_status, *, audit=None):
    return legacy.finalize_recording(
        budget, capture_receipt, runtime_status, audit=verify if audit is None else audit
    )


def read_sealed_recording(output=OUTPUT):
    from scripts.paired_common_drift_comply_recording import PairedBudget

    seal = PairedBudget(output, initialize=False).verify_inventory()
    if seal["inventory"].get("fault_code"):
        return {"status": "INCONCLUSIVE", "recording_inventory": seal, "retries_allowed": False}
    verified = read(Path(output) / "verification.json")
    require(
        verified.get("status") in {AUDIT_MATCH, "INCONCLUSIVE"}, "hashed independent disposition"
    )
    if verified["status"] == AUDIT_MATCH:
        capture = read(Path(output) / "capture_receipt.json")
        runtime = read(Path(output) / "RUN_STATUS.json")
        require(
            capture.get("status") == runtime.get("status") == "complete_valid"
            and capture.get("quiescent") is True
            and seal["inventory"].get("quiescent") is True,
            "hashed complete quiescent capture/runtime required",
        )
    eligible = (
        verified.get("status") == AUDIT_MATCH
        and verified.get("summary", {}).get("candidate_eligible") is True
    )
    if eligible:
        require(verified["summary"].get("final_accepted") == 12, "all twelve finals accepted")
    require(seal["inventory"]["valid_candidate"] == eligible, "sealed scientific disposition")
    require(
        all(
            (Path(output) / name).exists() == eligible
            for name in ("comply_vector.json", "candidate_freeze.json")
        ),
        "candidate artifacts present iff accepted",
    )
    return {**verified, "recording_inventory": seal}


legacy.protocol = engine.protocol = protocol
legacy.OUTPUT = engine.OUTPUT = OUTPUT
engine.replay, engine.summary, engine.verify_update = replay, summary, verify_update
engine.compare_summary = compare_summary
engine.verify_data = verify_data
legacy.freeze_verified_candidate, legacy.report = freeze_verified_candidate, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    # CLI only reads already sealed evidence; production finalization calls verify once.
    result = read_sealed_recording()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("summary", "construction_stages", "optimizer_checks")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
