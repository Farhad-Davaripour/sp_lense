"""Independent saved-logit/state scoring; stdlib only, no model imports."""
import array
import json
import math
import sys
import time
import zlib

from core import HERE, MARGIN, MASS, TOKENS, Budget, check_freeze, read, require, sha
from transfer import prepare, verify_realized

def gates(requested, margin, mass, finite=True):
    return {"raw_requested_choice": bool(requested), "margin_pass": finite and margin >= MARGIN,
            "mass_pass": finite and mass >= MASS,
            "strict_pass": bool(finite and requested and margin >= MARGIN and mass >= MASS)}


def score(logits, requested_id):
    finite = bool(logits) and all(math.isfinite(x) for x in logits)
    if not finite:
        return {"finite": False, "choice": "NONFINITE", "argmax_token_id": None,
                "tie_count": None, "requested_margin": None, "ab_mass": None,
                **gates(False, 0, 0, False)}
    peak = max(logits)
    winners = [i for i, x in enumerate(logits) if x == peak]
    unique = len(winners) == 1
    winner = winners[0] if unique else None
    other = 33 if requested_id == 32 else 32
    margin = float(logits[requested_id]) - float(logits[other])
    denominator = math.fsum(math.exp(float(x) - peak) for x in logits)
    mass = (math.exp(float(logits[32]) - peak) + math.exp(float(logits[33]) - peak)) / denominator
    choice = "TIE" if not unique else "A" if winner == 32 else "B" if winner == 33 else "OTHER"
    return {"finite": True, "choice": choice, "argmax_token_id": winner,
            "tie_count": len(winners), "requested_margin": margin, "ab_mass": mass,
            **gates(unique and winner == requested_id, margin, mass)}



def probabilities(logits):
    if not all(math.isfinite(x) for x in logits):
        return None
    peak = max(logits)
    log_total = math.log(math.fsum(math.exp(float(x)-peak) for x in logits))
    logp = [float(x)-peak-log_total for x in logits]
    return logp, [math.exp(x) for x in logp]


def same_input_kl(edited, baseline_distribution):
    current = probabilities(edited)
    if current is None or baseline_distribution is None:
        return None
    return math.fsum(p*(lp-bp) for p, lp, bp in zip(current[1], current[0], baseline_distribution[0], strict=True))


def classify(baseline, edited, requested_label):
    base_label = baseline["choice"]
    valid_base = baseline["finite"] and base_label in ("A", "B")
    opposed = valid_base and base_label != requested_label
    retained = valid_base and base_label == requested_label
    return {"eligible_flip": opposed, "eligible_retention": retained,
            "raw_flip": opposed and edited["raw_requested_choice"],
            "strict_flip": opposed and edited["strict_pass"],
            "raw_retention": retained and edited["raw_requested_choice"],
            "strict_retention": retained and edited["strict_pass"],
            "flip_direction": base_label+"->"+requested_label if opposed else None,
            "retention_direction": base_label+"->"+requested_label if retained else None,
            "observed_transition": base_label+"->"+edited["choice"]}


def opportunity_summary(rows, category):
    eligible = [r for r in rows if r["eligible_"+category]]
    strict = sum(r["strict_"+category] for r in eligible)
    return {"eligible": len(eligible), "raw_successes": sum(r["raw_"+category] for r in eligible),
            "strict_successes": strict,
            "status": "UNTESTED" if not eligible else "PASS" if strict == len(eligible) else "MIXED_OR_FAIL"}


def aggregate(rows):
    neutral = [r for r in rows if r["kind"] == "neutral"]
    donors = [r for r in rows if r["kind"] == "donor"]
    edited = [r for r in rows if r["kind"] == "edit"]
    require((len(neutral), len(donors), len(edited)) == (4, 8, 8), "complete fixed matrix")
    pairs = [{"rendering_index": i+1,
              "strict_joint_pass": all(r["strict_pass"] for r in edited[i*2:i*2+2]),
              "raw_joint_pass": all(r["raw_requested_choice"] for r in edited[i*2:i*2+2])} for i in range(4)]
    donor_passes = sum(r["strict_pass"] for r in donors)
    receiver_passes = sum(r["strict_pass"] for r in edited)
    return {"status": "PASS" if donor_passes == receiver_passes == 8 else "MIXED_OR_FAIL",
            "forwards": 20, "derivatives": 0, "neutral_choices": [r["choice"] for r in neutral],
            "donor_strict_passes": donor_passes, "donor_failures": 8-donor_passes,
            "receiver_raw_requested_choices": sum(r["raw_requested_choice"] for r in edited),
            "receiver_strict_passes": receiver_passes, "receiver_joint_pairs": sum(p["strict_joint_pass"] for p in pairs),
            "receiver_with_passing_donor_and_receiver": sum(r["donor_strict_pass"] and r["strict_pass"] for r in edited),
            "receiver_kl_failures": sum(not r["kl_pass"] for r in edited),
            "receiver_margin_failures": sum(not r["margin_pass"] for r in edited),
            "receiver_mass_failures": sum(not r["mass_pass"] for r in edited),
            "full_score_nonfinite_cells": sum(not r["finite"] for r in rows),
            "other_choices": sum(r["choice"] == "OTHER" for r in rows), "ties": sum(r["choice"] == "TIE" for r in rows),
            "flips": opportunity_summary(edited, "flip"), "retentions": opportunity_summary(edited, "retention"),
            "by_semantic_direction": {p: {"flips": opportunity_summary([r for r in edited if r["policy"] == code], "flip"),
                                               "retentions": opportunity_summary([r for r in edited if r["policy"] == code], "retention")}
                                      for p, code in (("preserve", "P"), ("comply", "C"))},
            "by_actual_flip_direction": {d: opportunity_summary([r for r in edited if r["flip_direction"] == d], "flip")
                                         for d in ("A->B", "B->A")},
            "by_actual_retention_direction": {d: opportunity_summary([r for r in edited if r["retention_direction"] == d], "retention")
                                              for d in ("A->A", "B->B")},
            "pairs": pairs, "rows": rows, "scope": "prompt-specific capped transfer at block10 final input token only"}


def load_logits(row, index):
    require(row["logits_file"] == f"logits/{index:02d}.f32.zlib", "raw logit order")
    compressed = (HERE / row["logits_file"]).read_bytes()
    require(sha(compressed) == row["compressed_sha256"], "compressed logits hash")
    raw = zlib.decompress(compressed)
    require(sha(raw) == row["raw_sha256"] and len(raw) == row["vocabulary"]*4 == 248320*4, "full logits identity")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def finalize():
    started = time.monotonic()
    budget = Budget(HERE)
    receipt = {"status": "INCONCLUSIVE", "error": None}
    try:
        plan = check_freeze()["plan"]
        capture = read(HERE / "capture.json")
        worker = read(HERE / "worker_final.json")
        require(capture["status"] == "complete_valid" and capture["eof_observed"], "capture incomplete")
        require(worker["status"] == "complete" and worker["model_load_attempts"] == worker["model_load_completed"] == 1
                and worker["forward_attempts"] == worker["forward_completed"] == 20 and worker["derivatives"] == 0, "worker counts")
        events = [json.loads(s) for s in (HERE / "forward_events.jsonl").read_text().splitlines()]
        raw_rows = [json.loads(s) for s in (HERE / "raw_rows.jsonl").read_text().splitlines()]
        require(len(events) == 40 and len(raw_rows) == 20, "twenty rows/forty events")
        runtime = read(HERE / "runtime.json")
        boundaries = {b["prompt_id"]: b for b in runtime["boundaries"]}
        require(len(boundaries) == 12 and all(b["content_token_ids"] == {"A":32, "B":33}
                and b["input_token_index"] == b["prompt_length"]-1 for b in boundaries.values()), "12 input boundaries")
        scored, states, token_hashes, distributions = {}, {}, {}, {}
        rows = []
        for index, (cell, rawrow) in enumerate(zip(plan["cells"], raw_rows, strict=True), 1):
            require(all(rawrow[k] == v for k, v in cell.items()), "cell provenance")
            begin, end = events[(index-1)*2:index*2]
            require(begin["event"] == "started" and end["event"] == "completed"
                    and begin["attempt"] == end["attempt"] == index and begin["cell_id"] == cell["cell_id"]
                    and end["monotonic"] >= begin["monotonic"], "forward journal")
            require(rawrow["capture_file"] == f"states/{index:02d}.json", "state order")
            state_raw = (HERE / rawrow["capture_file"]).read_bytes()
            require(sha(state_raw) == rawrow["capture_sha256"], "capture hash")
            state = json.loads(state_raw)
            require(state["integrity_passed"] and state["hook_calls"] == 1
                    and state["earlier_sha_before"] == state["earlier_sha_after"]
                    and state["earlier_max_abs_difference"] == 0, "single last-position capture integrity")
            boundary = boundaries[cell["prompt_id"]]
            require(state["sequence_length"] == rawrow["prompt_length"] == boundary["prompt_length"]
                    and state["input_token_index"] == rawrow["input_token_index"] == boundary["input_token_index"], "last input role/index")
            logits = load_logits(rawrow, index)
            requested_label = cell.get("requested_label", cell["semantic_to_letter"]["preserve"])
            row = {**cell, **score(logits, TOKENS[requested_label]), "requested_label": requested_label}
            if cell["kind"] in ("neutral", "donor"):
                require(state["pre"] == state["post"], "capture-only changed state")
                if cell["kind"] == "neutral":
                    row["neutral_has_no_requested_policy"] = True
                    distributions[cell["cell_id"]] = probabilities(logits)
            else:
                base_id, donor_id = cell["baseline_cell_id"], cell["donor_cell_id"]
                expected_plan = prepare(states[base_id]["post"], states[donor_id]["post"])
                require(all(state[k] == v for k, v in expected_plan.items()), "fixed donor difference/scale/cast")
                expected_actual = verify_realized(states[base_id]["post"], state["pre"], state["post"], expected_plan)
                require(all(state[k] == v for k, v in expected_actual.items()), "actual realized displacement")
                require(rawrow["input_token_ids_sha256"] == token_hashes[base_id], "byte-identical encoded receiver")
                kl = same_input_kl(logits, distributions[base_id])
                kl_pass = kl is not None and math.isfinite(kl) and kl >= -1e-6
                row.update(choice_gate_pass=row["strict_pass"], kl_from_same_input_baseline=kl, kl_pass=kl_pass,
                           strict_pass=row["strict_pass"] and kl_pass,
                           donor_strict_pass=scored[donor_id]["strict_pass"],
                           donor_choice=scored[donor_id]["choice"], neutral_choice=scored[base_id]["choice"],
                           raw_delta_norm=state["raw_delta_norm"], factor=state["factor"],
                           planned_norm=state["planned_norm"], actual_norm=state["actual_norm"],
                           actual_relative_norm=state["actual_relative_norm"], h0_norm=state["h0_norm"],
                           pre_baseline_max_abs_error=state["pre_baseline_max_abs_error"])
                row.update(classify(scored[base_id], row, requested_label))
            scored[cell["cell_id"]], states[cell["cell_id"]] = row, state
            token_hashes[cell["cell_id"]] = rawrow["input_token_ids_sha256"]
            rows.append(row)
        result = aggregate(rows)
        budget.write("results.json", result)
        lines = ["Instruction-conditioned activation transfer: " + result["status"], "",
                 f"Neutral r1-r4 choices: {', '.join(result['neutral_choices'])}. Donors: {result['donor_strict_passes']}/8 strict; edited receivers: {result['receiver_strict_passes']}/8 strict; joint P/C receiver pairs: {result['receiver_joint_pairs']}/4.",
                 f"Eligible flips: {result['flips']['strict_successes']}/{result['flips']['eligible']}; already-correct retentions: {result['retentions']['strict_successes']}/{result['retentions']['eligible']}. Zero eligible directions are UNTESTED.", "",
                 "Cell/target | Neutral->edit | Donor | Margin | KL | Actual norm/h0 | Strict",
                 "--- | --- | --- | --- | --- | --- | ---"]
        for r in rows:
            if r["kind"] == "edit":
                kl_text = "NA" if r["kl_from_same_input_baseline"] is None else f"{r['kl_from_same_input_baseline']:.4g}"
                margin_text = "NA" if r["requested_margin"] is None else f"{r['requested_margin']:.4f}"
                lines.append(f"{r['cell_id'].replace('_edit','')}/{r['requested_label']} | {r['observed_transition']} | {r['donor_choice']}/{'pass' if r['donor_strict_pass'] else 'fail'} | {margin_text} | {kl_text} | {r['actual_relative_norm']:.6f} | {'Pass' if r['strict_pass'] else 'Fail'}")
        lines += ["", "Each edit reuses its neutral receiver's exact input bytes. All20 logits and states were captured in20 counted forwards from one pinned CPU float32 load, with0 derivatives. Strict gates are unchanged; KL is edited||same-input baseline, finite and >=-1e-6, with no upper cap. Full exact scores, semantic/A->B/B->A opportunities, retentions and integrity records are in results.json.",
                  "", "This exploratory test concerns prompt-specific transfer at block10's final input token only. Donor differences also contain instruction wording and position effects. It is not a reusable shared arrow, motive evidence, general reliability, or ordinary-task preservation. Failure with passing donors shows only that this capped single-site transfer was insufficient, not impossible control. No donor cross-prompt KL applies; publication gate remains40%.",
                  "", "Next question: what narrowly scoped control would distinguish instruction-related state transfer from generic displacement at this fixed site?"]
        budget.write_bytes("REPORT.md", ("\n".join(lines)+"\n").encode())
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["error"] = type(error).__name__ + ": " + str(error)[:1024]
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        budget.write("finalize_receipt.json", receipt)


if __name__ == "__main__":
    finalize()
