"""Cold fixed workflow with an arithmetic fake model, not a Qwen runner."""
import json
import math
import os
import time
import torch
from area import WorkflowWriter
from pins import ROOT, SCIENCE, GATE_SHA, require, sha, scientific
from schedule import build_plan

torch.set_num_threads(1)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class ScientificStop(ValueError):
    pass


class PartialWrite:
    def __init__(self):
        self.failed = False

    def __call__(self, fd, raw):
        if self.failed:
            raise OSError("prospectively injected partial raw-vector write")
        if len(raw) == 993280:
            self.failed = True
            return os.write(fd, raw[:4096])
        return os.write(fd, raw)


class Fake:
    """Final-token float32 logits and a 1024-state linear model; no real hooks."""
    def __init__(self, plan, artifact, mode):
        self.mode = mode
        p = artifact["parameters"]
        mean, direction = torch.tensor(p["grand_mean"]), torch.tensor(p["direction"])
        self.states = {x["prompt_id"]: mean + (10 if x["fixture_self_state"] else -10) * direction for x in plan["prompts"]}
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.active = None
        self.cache = None

    def forward(self, prompt, cell, offset):
        base = self.states[prompt["prompt_id"]]
        h = (base + offset).detach().clone().requires_grad_(cell["phase"].startswith("gradient_"))
        z = torch.full((248320,), -100., dtype=torch.float32)
        if prompt["fixture_self_state"]:
            margin = -.1 if prompt["layout"] == "keep_first" else .1
            if self.mode == "finite_eligibility" and cell["phase"] == "baseline":
                margin = .02
            z[50057] = margin + h[0] - base[0]
            z[48964] = 0.
            if self.mode == "endpoint_corruption" and cell["phase"] == "endpoint":
                z[50057] = z[50057] + .001  # Finite endpoint identity violation.
        elif prompt["category"] == "ordinary":
            z[32], z[33] = .1, 0.
            if prompt["input_ids"][0] in (20, 23):
                z[33] = .2  # Exactly retained wrong answers, not accuracy gain.
        else:
            z[50057], z[48964], z[100], z[101] = .01, .01, 1., 1.
            # OFF remains tested even with ties, wrong winners and poor pair mass.
        if self.mode == "routing" and cell["phase"] == "baseline":
            h = (base - 20 * torch.tensor(self.direction)).detach()
        self.cache = (z, h)
        return z, h


def execute(name, mode, deadline):
    science = scientific()
    plan = build_plan()
    writer = WorkflowWriter(name, write_function=PartialWrite() if mode == "partial_write" else None)
    c, helper = writer.contract, writer.helper
    helper.validate_prompt_lengths(c, [len(p["input_ids"]) for p in plan["prompts"]])
    counters = helper.AttemptCounter(c)
    state = {"execution_mode": "SYNTHETIC_ONLY", "status": "RUNNING", "scientific_failures": [],
             "technical_failures": [], "cell_status": {cell["cell_id"]: "UNRUN" for cell in plan["cells"]},
             "request_status": {r["request_id"]: "UNRUN" for r in plan["requests"]},
             "forward_completed": 0, "derivatives_completed": 0, "cleanup_complete": False,
             "real_supervisors_executed": False, "real_hook_capacity_verified": False}
    writer.workflow_state = state
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    cells = {(x["prompt_id"], x["request_id"], x["phase"]): x for x in plan["cells"]}
    originals = {}
    current_rid = None
    cursor = 0
    model = gate = None

    def event(stream, record):
        writer.append_event(stream, encoded(record))

    def claim(cell):
        nonlocal cursor
        require(time.monotonic() < deadline, "frozen batch deadline")
        require(plan["cells"][cursor] == cell and state["cell_status"][cell["cell_id"]] == "UNRUN", "ordered nonrepeated schedule")
        cursor += 1

    def fail_science(kind, cell):
        record = {"kind": kind, "cell_id": cell["cell_id"]}
        state["scientific_failures"].append(record)
        event("scientific_failures.jsonl", record)
        raise ScientificStop(kind)

    def call(prompt, rid, phase, offset=None, current=None, sign=0, extra=None):
        cell = cells[prompt["prompt_id"], rid, phase]
        claim(cell)
        counters.reserve_attempt("forward")
        state["cell_status"][cell["cell_id"]] = "STARTED"
        event("forward_events.jsonl", {"event": "started", "cell_id": cell["cell_id"], "attempt": counters.attempts["forward"]})
        delta = torch.zeros(1024) if offset is None else offset
        try:
            z, h = model.forward(prompt, cell, delta)
            raw = z.detach().numpy().astype("<f4", copy=False).tobytes()
            logits_file = f"logits/{counters.attempts['forward']:03d}.f32"
            writer.write_logits(logits_file, raw)
            baseline = originals.get(prompt["prompt_id"])
            baseline_z = z.detach() if baseline is None else baseline[1]
            score_module = science.word_score if prompt["category"] != "ordinary" else science.letter_score
            kwargs = ({"choice_keep_token_id": 50057, "choice_stop_token_id": 48964} if prompt["category"] != "ordinary"
                      else {"choice_a_token_id": 32, "choice_b_token_id": 33})
            scores = score_module.score_float32_logits(torch, z.detach(), baseline_z, preserve_label=prompt["preserve_label"], **kwargs)
            h0 = h.detach() if phase in ("baseline", "entry") else originals[prompt["prompt_id"]][2]
            row = {**cell, **scores, "logits_file": logits_file, "logits_sha256": sha(raw),
                   "h": h.detach().tolist(), "h0": h0.tolist(), "offset": delta.tolist(), "input_ids": prompt["input_ids"],
                   "preserve_label": prompt["preserve_label"], "baseline_cell_id": prompt["prompt_id"] + "__baseline",
                   "gradient": None, "target_sign": sign, "unselected_sha256": sha(b"fake-unchanged-nonfinal-state"),
                   "current_cell_id": current[0]["cell_id"] if current else None, **(extra or {})}
            if phase.startswith("gradient_"):
                counters.reserve_attempt("derivative")
                event("derivative_events.jsonl", {"event": "started", "cell_id": cell["cell_id"]})
                g = torch.autograd.grad(z[50057] - z[48964], h)[0]
                require(bool(g.isfinite().all()), "finite current fake gradient")
                row["gradient"] = g.tolist()
                state["derivatives_completed"] += 1
                event("derivative_events.jsonl", {"event": "completed", "cell_id": cell["cell_id"]})
            if phase in ("baseline", "entry"):
                decision = gate.decide(row["h"])
                row["routing"] = decision
                event("routing_events.jsonl", {"cell_id": cell["cell_id"], **decision})
            writer.write_row(f"rows/{counters.attempts['forward']:03d}.json", encoded(row))
            event("forward_events.jsonl", {"event": "completed", "cell_id": cell["cell_id"]})
            state["forward_completed"] += 1
            state["cell_status"][cell["cell_id"]] = "DONE"
            if phase in ("baseline", "entry") and row["routing"]["route"] != prompt["expected_route_audit_only"]:
                fail_science("routing", cell)
            if prompt["category"] == "self" and phase in ("baseline", "entry"):
                try:
                    science.eligibility(row)
                except science.EligibilityError:
                    fail_science("finite_eligibility", cell)
            return row, z.detach(), h.detach()
        except BaseException:
            if state["cell_status"][cell["cell_id"]] == "STARTED":
                state["cell_status"][cell["cell_id"]] = "FAILED"
            raise

    def identity(a, b, tolerance):
        require(a[0]["input_ids"] == b[0]["input_ids"] and torch.equal(a[2], b[2]), "exact input/hidden identity")
        require(float((a[1] - b[1]).abs().max()) <= tolerance, "full-vocabulary endpoint/current identity")
        for key in ("actual_next_token_id", "forced_pair_label", "unselected_sha256"):
            require(a[0][key] == b[0][key], "current-state discrete identity")

    try:
        writer.write_source("plan.json", encoded(plan))
        writer.write_source("fitted_parameters.json", science.artifact_raw)
        writer.append_log("worker_stdout.log", b"SYNTHETIC_ONLY fixed workflow started\n")
        counters.reserve_attempt("load")
        model = Fake(plan, json.loads(science.artifact_raw), mode)
        model.direction = json.loads(science.artifact_raw)["parameters"]["direction"]
        gate = science.FrozenGate(ROOT / SCIENCE / "fitted_parameters.json", GATE_SHA)
        initial_weight = sha(model.weight.detach().numpy().tobytes())
        for prompt in plan["prompts"]:
            originals[prompt["prompt_id"]] = call(prompt, None, "baseline")
            model.cache = None
        for spec in plan["requests"]:
            current_rid = rid = spec["request_id"]
            state["request_status"][rid] = "STARTED"
            prompt = prompts[spec["prompt_id"]]
            require(model.active is None and model.cache is None and model.weight.requires_grad, "cold request state")
            entry = call(prompt, rid, "entry")
            original = originals[prompt["prompt_id"]]
            if entry[0]["routing"]["route"] == "OFF":
                identity(entry, original, 0.)
                require(torch.equal(entry[1], original[1]) and not any(entry[0]["offset"]), "exact OFF own-baseline identity")
                event("off_returns.jsonl", {"request_id": rid, "entry_cell_id": entry[0]["cell_id"],
                    "baseline_cell_id": original[0]["cell_id"], "no_additional_forward": True, "zero_updates": True})
            else:
                identity(entry, original, 1e-6)
                model.active = rid
                model.weight.requires_grad_(False)
                current, offset, path, updates = entry, torch.zeros(1024), 0., 0
                sign, wanted = spec["sign"], 50057 if spec["sign"] == 1 else 48964
                retention = science.accepted(entry[0], sign, wanted)
                stopped = "accepted" if retention else None
                try:
                    for k in range(1, 5):
                        if stopped:
                            for phase in (f"gradient_{k}", f"step_{k}"):
                                cell = cells[prompt["prompt_id"], rid, phase]
                                claim(cell)
                                event("skip_events.jsonl", {"cell_id": cell["cell_id"], "reason": stopped, "after_cell_id": current[0]["cell_id"]})
                                state["cell_status"][cell["cell_id"]] = "SKIPPED"
                            continue
                        gradient = call(prompt, rid, f"gradient_{k}", offset, current, sign)
                        identity(gradient, current, 1e-6)
                        g = torch.tensor(gradient[0]["gradient"], dtype=torch.float32)
                        recipe = science.step_recipe(current[0]["preserve_log_odds"], sign, g.tolist(), entry[0]["h0"])
                        requested = g * recipe["coefficient"]  # Frozen float32 arithmetic.
                        next_offset = offset + requested
                        step = call(prompt, rid, f"step_{k}", next_offset, current, sign,
                                    {"recipe": recipe, "gradient_cell_id": gradient[0]["cell_id"], "requested_step": requested.tolist()})
                        realized = [a - b for a, b in zip(step[0]["h"], current[0]["h"], strict=True)]
                        sn, hn = science.norm(realized), science.norm(entry[0]["h0"])
                        path += sn
                        net = science.norm([a - b for a, b in zip(step[0]["h"], entry[0]["h"], strict=True)])
                        require(max(abs(a-b) for a,b in zip(realized,requested.tolist(),strict=True)) <= 1e-6
                                and abs(sn-recipe["requested_step_norm"]) <= 1e-6
                                and sn <= .05*hn+1e-6 and path <= .20*hn+1e-6 and net <= min(path,.20*hn)+1e-6,
                                "unchanged actual step/path/net geometry")
                        offset, current, updates = next_offset, step, k
                        stopped = "accepted" if science.accepted(step[0], sign, wanted) else "quality_failure" if not science.valid(step[0]) else None
                    endpoint = call(prompt, rid, "endpoint", offset, current, sign,
                                    {"retention_endpoint": retention, "selected_endpoint_cell_id": current[0]["cell_id"]})
                    identity(endpoint, current, 1e-6 if retention else 2e-5)
                    if retention:
                        require(updates == 0 and not bool(offset.any()) and abs(endpoint[0]["kl_from_baseline"]) <= 1e-6, "zero-edit retention")
                    if not (science.accepted(current[0], sign, wanted) and science.accepted(endpoint[0], sign, wanted)):
                        fail_science("endpoint_behavior", cells[prompt["prompt_id"], rid, "endpoint"])
                    event("requests.jsonl", {"request_id": rid, "kind": "retention" if retention else "flip",
                        "updates": updates, "entry_cell_id": entry[0]["cell_id"], "endpoint_cell_id": endpoint[0]["cell_id"]})
                finally:
                    model.weight.requires_grad_(True)
                    model.active = model.cache = None
                    if not writer.sticky_failure:
                        event("cleanup_events.jsonl", {"request_id": rid, "flags_restored": True,
                            "gradients_absent": model.weight.grad is None, "synthetic_only_no_real_hooks": True})
            state["request_status"][rid] = "DONE"
            model.cache = None
            current_rid = None
        state["status"] = "PASS_SYNTHETIC_ONLY"
    except ScientificStop as error:
        state["status"] = "FAIL_SYNTHETIC_ONLY"
    except BaseException as error:
        state["technical_failures"].append({"type": type(error).__name__, "message": str(error)})
        state["status"] = "INCONCLUSIVE_SYNTHETIC_ONLY"
    finally:
        if current_rid is not None:
            state["request_status"][current_rid] = "FAILED"
        if state["status"] != "PASS_SYNTHETIC_ONLY":
            counters.mark_failed()
        if model is not None:
            model.weight.requires_grad_(True)
            model.active = model.cache = None
            state["cleanup_complete"] = (model.weight.grad is None and sha(model.weight.detach().numpy().tobytes()) == initial_weight)
        if gate is not None:
            gate.unchanged()
        state["attempts"] = dict(counters.attempts)
        state["cursor"] = cursor
        state["fresh_routes"] = gate.calls if gate else 0
        if not writer.sticky_failure:
            event("gate_final.json", {"decisions": state["fresh_routes"], "parameters_unchanged": True, "fit_calls": 0})
            event("integration_cleanup.json", {"complete": state["cleanup_complete"], "synthetic_only_no_real_hooks": True})
            writer.append_log("worker_stdout.log", (state["status"] + "\n").encode())
        capture = writer.closeout()
    return capture
