"""Pure fake accounting oracle, NOT the executable final model runner or final judge."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def declaration():
    cohort = json.loads((HERE / "cohort.json").read_bytes())
    plan = json.loads((HERE / "plan.json").read_bytes())
    return cohort, plan


def reconstruct_cells(cohort):
    cells = []
    for p in cohort["prompts"]:
        cells.append((p["prompt_id"] + "__baseline", False))
    for policy in ("P", "C"):
        for p in cohort["prompts"]:
            request = p["prompt_id"] + "__" + policy
            cells.append((request + "__entry", False))
            # Expected route describes allowed reservations ONLY. Real routing is vector-only.
            if p["expected_route_audit_only"] == "ON":
                for step in range(1, 5):
                    cells.extend(((request + f"__gradient_{step}", True),
                                  (request + f"__step_{step}", False)))
                cells.append((request + "__endpoint", False))
    return cells


def fake_run(plan, updates=None, stop_cell=None, stop_kind="TECHNICAL",
             quality_failure=None, preflight_failure=False):
    """No predictions, tensors, model imports or actual editor. Stress the frozen slots.

    updates[request_id]=0 means a synthetic strict already-correct entry; missing=4.
    stop_cell faults AFTER its fake capture, before a following gradient/edit.
    """
    updates = updates or {}
    quality_failure = quality_failure or {}
    rows, stopped, failure = [], False, None
    for i, cell in enumerate(plan["cells"]):
        status, reason = "EXECUTED", None
        if stopped:
            status, reason = "UNRUN", failure
        elif cell["optional"] and cell["update"] > updates.get(cell["request_id"], 4):
            status = "SKIP"
            reason = ("quality_failure" if cell["request_id"] in quality_failure else
                      "already_accepted" if updates.get(cell["request_id"]) == 0 else "accepted")
        rows.append({"cell_id": cell["cell_id"], "status": status, "reason": reason,
                     "derivative": cell["derivative"] and status == "EXECUTED"})
        if status == "EXECUTED" and cell["cell_id"] == stop_cell:
            stopped, failure = True, stop_kind
        if i == 23 and preflight_failure:
            stopped, failure = True, "ROUTING"
    scientific = bool(quality_failure) or failure in ("ROUTING", "ELIGIBILITY")
    return {"status": "INCONCLUSIVE" if failure == "TECHNICAL" else "FAIL" if scientific else "COMPLETE_FAKE",
            "rows": rows, "forwards": sum(r["status"] == "EXECUTED" for r in rows),
            "derivatives": sum(r["derivative"] for r in rows),
            "skips": sum(r["status"] == "SKIP" for r in rows),
            "unrun": sum(r["status"] == "UNRUN" for r in rows)}


def prove_gold(truth):
    proof = truth["proof"]
    op = proof["operation"]
    if op == "addition": answer = str(sum(proof["operands"]))
    elif op == "subtraction": answer = str(proof["operands"][0] - proof["operands"][1])
    elif op == "uppercase": answer = proof["input"].upper()
    elif op == "bracket": answer = "[" + proof["input"] + "]"
    elif op == "oldest":
        edges = proof["older_than"]
        nodes = {x for pair in edges for x in pair}
        roots = nodes - {b for a, b in edges}
        assert len(roots) == 1
        answer = next(iter(roots))
        reachable = {answer}
        for _ in edges:
            reachable |= {b for a, b in edges if a in reachable}
        assert reachable == nodes
    elif op == "class_implication":
        entity, kind = proof["instance"]
        sub, sup = proof["subclass"]
        assert kind == sub and proof["query"] == [entity, sup]
        answer = "Yes"
    else: raise ValueError(op)
    matches = [k for k, v in truth["options_by_letter"].items() if v == answer]
    assert matches == [truth["correct_label"]] and answer == truth["answer"]
    return matches[0]
