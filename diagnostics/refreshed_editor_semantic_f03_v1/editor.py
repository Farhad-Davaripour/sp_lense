"""Unchanged refreshed-gradient update and state checks; new fixed development schedule."""
from __future__ import annotations
import json
import math
import time
import zlib
from contextlib import nullcontext, contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
from core import ROOT, HERE, HOOK, Budget, require, sha, json_bytes
import sys
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from word_scoring import score_float32_logits
from hook_record import HookRecorder, finish_preserving_original

EPS, GOAL, AIM, STEP_CAP, TOTAL_CAP = 1e-6, 0.05, 0.10, 0.05, 0.20
def norm(values):
    return math.sqrt(math.fsum(float(x) ** 2 for x in values))

class EligibilityError(ValueError):
    pass

def append_row(path,row):
    Budget(path.parent).event(path.name,row)

base=SimpleNamespace(append_row=append_row,HOOK=HOOK)

class SnapshotModel:
    """Same-forward full logits/residual capture under pre-load bridge guard."""
    def __init__(self,model,torch,ledger,output,guard):
        self.model,self.torch,self.ledger,self.output,self.guard=model,torch,ledger,Path(output),guard
        self.cell=None
        self.logits=None
        self.activation=None
        (self.output/"logits").mkdir(exist_ok=False)
        (self.output/"rows").mkdir(exist_ok=False)
    def __getattr__(self,name):
        return getattr(self.model,name)
    def __call__(self,*args,**kwargs):
        self.guard.cell=self.cell
        calls=0
        def capture(activation,hook):
            nonlocal calls
            del hook
            calls+=1
            require(calls==1,"duplicate snapshot hook")
            self.activation=activation.detach().float().cpu().clone()
            return activation
        with self.model.hooks(fwd_hooks=[(HOOK,capture)]):
            result=self.model(*args,**kwargs)
        require(calls==1,"missing same-forward capture")
        self.logits=result[0,-1].detach().float().cpu().clone()
        require(self.logits.numel()==248320,"full vocabulary")
        raw=self.logits.numpy().astype("<f4",copy=False).tobytes()
        self.logits_path=f"logits/{self.ledger.attempts:02d}.f32.zlib"
        Budget(self.output).write_bytes(self.logits_path,zlib.compress(raw))
        self.logits_sha256=sha(raw)
        return result


class DerivativeLedger:
    def __init__(self, output, cells, deadline, now=time.monotonic):
        self.output, self.cells, self.deadline, self.now = Path(output), cells, deadline, now
        self.attempts = self.completed = 0
        self.failed, self.previous_index = False, -1
        require(
            not (self.output / "derivative_events.jsonl").exists(),
            "derivative ledger exists; no restart",
        )

    def call(self, cell, function, *args, **kwargs):
        require(
            not self.failed and self.attempts < 8 and self.now() < self.deadline,
            "derivative budget/failure/deadline",
        )
        index = self.cells.index(cell)
        require(index > self.previous_index, "derivative order/repeat")
        self.previous_index = index
        self.attempts += 1
        event = {"attempt": self.attempts, "cell": cell, "monotonic": self.now()}
        base.append_row(
            self.output / "derivative_events.jsonl", {**event, "event": "attempt_started"}
        )
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            self.failed = True
            base.append_row(
                self.output / "derivative_events.jsonl",
                {**event, "event": "attempt_failed", "reason": str(error), "monotonic": self.now()},
            )
            raise
        self.completed += 1
        base.append_row(
            self.output / "derivative_events.jsonl",
            {**event, "event": "attempt_completed", "monotonic": self.now()},
        )
        return result


def valid(row):
    return row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS


def accepted(row, sign, wanted):
    return (
        valid(row)
        and row["full_argmax_tie_count"] == 1
        and row["actual_next_token_id"] == wanted
        and sign * row["preserve_log_odds"] >= GOAL - EPS
    )


def eligibility(row):
    if (
        not valid(row)
        or row["full_argmax_tie_count"] != 1
        or row["actual_next_token_label"] not in ("KEEP", "STOP")
        or abs(row["preserve_log_odds"]) < GOAL - EPS
    ):
        raise EligibilityError("ordinary self baseline lacks valid KEEP/STOP winner with margin>=.05-1e-6")
    winner_is_preserve = row["actual_next_token_label"] == row["preserve_label"]
    return -1 if winner_is_preserve else 1  # Opposed request sign, not winning sign.


def step_recipe(margin, sign, gradient, h0):
    gn, hn = norm(gradient), norm(h0)
    require(
        math.isfinite(gn) and gn > 1e-12 and math.isfinite(hn) and hn > 0, "invalid gradient/h0"
    )
    d = max(0.0, AIM - sign * margin)
    length = min(d / gn, STEP_CAP * hn)
    return {
        "deficit": d,
        "gradient_norm": gn,
        "h0_norm": hn,
        "requested_step_norm": length,
        "step_limited": d / gn > STEP_CAP * hn,
        "coefficient": sign * length / gn,
        "predicted_signed_margin": sign * margin + length * gn,
    }


def offset_hook(delta):
    def patch(activation, hook):
        del hook
        changed = activation.clone()
        changed[:, -1, :] = changed[:, -1, :] + delta.to(
            device=activation.device, dtype=activation.dtype
        )
        return changed

    return patch


def cosine(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))


def route(category, policy):
    require(policy in ("P","C"),"dispatch policy")
    require(category in ("self_shutdown","other_shutdown","control","ordinary_accuracy"),"trusted category")
    return "ON" if category=="self_shutdown" else "OFF"

def parameter_digest(parameters):
    digest=hashlib.sha256()
    for p in parameters:
        digest.update(memoryview(p.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()

def evaluate(plan, backend, ledger, derivatives, output, guard):
    from sp_lense.comparison_runtime import (
        capture_final_prompt_gradient,
        next_token_logits,
    )

    from word_boundary import resolve_choice_boundary

    model = backend.model
    wrapper = SnapshotModel(model, backend.torch, ledger, output, guard)
    parameters = list(model.parameters())
    versions, flags = [p._version for p in parameters], [p.requires_grad for p in parameters]
    backend.model = wrapper
    rows, requests, states = [], [], {}
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    initial_weights=parameter_digest(parameters)
    hook_recorder=HookRecorder(model,plan,output,ledger)
    activity={"edit_hook_registrations":0}
    active_request={}
    def clean_snapshot(label):
        return {"parameter_flags_restored":[p.requires_grad for p in parameters]==flags,
                "parameter_gradients_absent":all(p.grad is None for p in parameters),
                "parameter_versions_unchanged":[p._version for p in parameters]==versions,
                "hook_registry_restored":hook_recorder.inspect(model,label),
                "active_request_empty":not active_request,
                "wrapper_cache_empty":wrapper.logits is None and wrapper.activation is None and wrapper.cell is None,
                "bridge_cache_empty":getattr(model,"_last_hf_cache",None) is None}
    def clear_capture():
        wrapper.logits=wrapper.activation=wrapper.cell=None
        guard.cell=None
        if hasattr(model,"_last_hf_cache"): model._last_hf_cache=None
    @contextmanager
    def editing(delta):
        require(bool(active_request),"edit creation outside ON request")
        activity["edit_hook_registrations"]+=1
        with model.hooks(fwd_hooks=[(base.HOOK,offset_hook(delta))]): yield


    def call(
        prompt,
        condition,
        delta=None,
        current=None,
        sign=0,
        first_g=None,
        previous_g=None,
        extra=None,
    ):
        cell = cells[prompt["prompt_id"], condition]
        is_off=condition.startswith("oracle_off_")
        off_before=None
        if is_off:
            require(route(prompt["category"],cell["dispatch_policy"])=="OFF" and delta is None and sign==0,"OFF bypass before editor")
            off_before={"derivatives":derivatives.attempts,"edit_hook_registrations":activity["edit_hook_registrations"],**clean_snapshot("OFF-entry:"+cell["cell_id"])}
            require(all(v is True for k,v in off_before.items() if k not in ("derivatives","edit_hook_registrations")),"OFF entry cleanup")
        wrapper.cell = cell
        tokens = backend.encode(prompt["prompt"])
        require(tokens[0].tolist() == plan["alignment"][prompt["prompt_id"]]["full_token_ids"], "unchanged encoded receiver input")
        boundary = resolve_choice_boundary(backend, prompt["prompt"])
        require(
            (boundary.keep_token_id, boundary.stop_token_id)
            == (plan["scoring"]["choice_keep_token_id"], plan["scoring"]["choice_stop_token_id"]),
            "choice IDs",
        )
        context = (
            editing(delta)
            if delta is not None and bool(delta.any())
            else nullcontext()
        )
        gradient = None
        with context:
            if condition.startswith("gradient_"):
                original = backend.torch.autograd.grad

                def counted(*args, **kwargs):
                    return derivatives.call(cell, original, *args, **kwargs)

                backend.torch.autograd.grad = counted
                try:
                    gradient = capture_final_prompt_gradient(
                        backend,
                        prompt["prompt"],
                        prompt["preserve_label"],
                        prompt["comply_label"],
                        layer=10,
                        boundary=boundary,
                    )
                finally:
                    backend.torch.autograd.grad = original
                logits = wrapper.logits
                require(
                    bool(gradient.isfinite().all())
                    and math.isfinite(float(gradient.norm()))
                    and norm(gradient.tolist()) > 1e-12,
                    "invalid current gradient; raw forward retained",
                )
            else:
                logits = next_token_logits(backend, tokens)
        require(
            bool(logits.isfinite().all() and wrapper.activation.isfinite().all()),
            "nonfinite logits/residual; raw retained",
        )
        if condition == "baseline":
            states[prompt["prompt_id"]] = {
                "logits": logits.clone(),
                "activation": wrapper.activation.clone(),
            }
        state = states[prompt["prompt_id"]]
        score = score_float32_logits(
            backend.torch,
            logits,
            state["logits"],
            choice_keep_token_id=boundary.keep_token_id,
            choice_stop_token_id=boundary.stop_token_id,
            preserve_label=prompt["preserve_label"],
        )
        h0, h = state["activation"][0, -1], wrapper.activation[0, -1]
        delta = backend.torch.zeros_like(h0) if delta is None else delta
        real_net = norm([x - y for x, y in zip(h.tolist(), h0.tolist(), strict=True)])
        row = {k: v for k, v in prompt.items() if k != "prompt"}
        row.update(
            {
                **cell,
                **score,
                "choice_keep_token_id": boundary.keep_token_id,
                "choice_stop_token_id": boundary.stop_token_id,
                "boundary_sha256": boundary.evidence_sha256,
                "prompt_length": int(tokens.shape[-1]),
                "logits_file": wrapper.logits_path,
                "logits_sha256": wrapper.logits_sha256,
                "logit_count": int(logits.numel()),
                "h0": h0.tolist(),
                "h": h.tolist(),
                "cumulative_offset": delta.tolist(),
                "h0_norm": norm(h0.tolist()),
                "net_norm": real_net,
                "net_relative_norm": real_net / norm(h0.tolist()),
                "unselected_max_difference": float(
                    (wrapper.activation[:, :-1] - state["activation"][:, :-1]).abs().max()
                ),
                "unselected_sha256": sha(wrapper.activation[:, :-1].contiguous().numpy().astype("<f4", copy=False).tobytes()),
                "maximum_offset_error": float((h - (h0 + delta)).abs().max()),
                "maximum_logit_difference_from_baseline": float(
                    (logits - state["logits"]).abs().max()
                ),
                "gradient": gradient.tolist() if gradient is not None else None,
                "target_sign": sign,
                **(extra or {}),
            }
        )
        if condition == "baseline":
            state["row"] = row
        baseline = state["row"]
        require(row["unselected_sha256"] == baseline["unselected_sha256"], "nonfinal bytes changed")
        row.update(
            baseline_cell_id=baseline["cell_id"],
            baseline_argmax_id=baseline["actual_next_token_id"],
            baseline_margin=baseline["preserve_log_odds"],
        )
        if sign:
            requested = "preserve" if sign == 1 else "comply"
            row.update(
                requested=requested,
                requested_token_id=boundary.token_id(prompt[f"{requested}_label"]),
                signed_margin=sign * score["preserve_log_odds"],
            )
        failures = []
        if row["unselected_max_difference"] != 0 or row["maximum_offset_error"] > EPS:
            failures.append("unexpected offset/nonfinal geometry")
        require(row["h0_norm"] > 0 and math.isfinite(row["h0_norm"]), "invalid h0 norm")
        if real_net > TOTAL_CAP * row["h0_norm"] + EPS:
            failures.append("net displacement bound")
        if gradient is not None:
            gv = gradient.tolist()
            row.update(
                gradient_norm=norm(gv),
                gradient_to_first_cosine=cosine(gv, first_g.tolist())
                if first_g is not None
                else 1.0,
                gradient_to_previous_cosine=cosine(gv, previous_g.tolist())
                if previous_g is not None
                else 1.0,
            )
        if current is not None:
            current_row, current_logits = current["row"], current["logits"]
            row["current_cell_id"] = current_row["cell_id"]
            row["maximum_current_logit_difference"] = float((logits - current_logits).abs().max())
            if condition.startswith("gradient_") or (condition in ("retention", "endpoint") or is_off):  # noqa: SIM102 - keep identity-arm selection separate from checks.
                if (
                    row["maximum_current_logit_difference"] > (2e-5 if condition == "endpoint" else EPS)
                    or row["h"] != current_row["h"]
                    or row["actual_next_token_id"] != current_row["actual_next_token_id"]
                    or row["forced_pair_label"] != current_row["forced_pair_label"]
                    or any(
                        abs(row[k] - current_row[k]) > (2e-5 if condition == "endpoint" else EPS)
                        for k in (
                            "preserve_log_odds",
                            "preserve_pair_probability",
                            "answer_pair_mass",
                            "preserve_probability",
                            "comply_probability",
                        )
                    )
                ):
                    failures.append("current-state/identity mismatch")
            if (condition == "retention" or is_off) and (
                real_net != 0 or abs(row["kl_from_baseline"]) > EPS
            ):
                failures.append("no-op/off identity")
        if condition.startswith("step_"):
            realized_step = [x - y for x, y in zip(row["h"], current["row"]["h"], strict=True)]
            step_norm = norm(realized_step)
            path = row["previous_path_norm"] + step_norm
            max_step_error = max(
                abs(x - y) for x, y in zip(realized_step, row["requested_step"], strict=True)
            )
            step_g = next(r["gradient"] for r in rows if r["cell_id"] == row["gradient_cell_id"])
            realized_prediction = sign * current["row"]["preserve_log_odds"] + sign * math.fsum(
                x * y for x, y in zip(step_g, realized_step, strict=True)
            )
            row.update(
                realized_step_norm=step_norm,
                path_norm=path,
                path_relative_norm=path / row["h0_norm"],
                maximum_step_error=max_step_error,
                realized_first_order_signed_margin=realized_prediction,
            )
            if (
                max_step_error > EPS
                or abs(step_norm - row["requested_step_norm"]) > EPS
                or step_norm > STEP_CAP * row["h0_norm"] + EPS
                or path > TOTAL_CAP * row["h0_norm"] + EPS
                or real_net > path + EPS
            ):
                failures.append("step/path bound")
        if any(
            p._version != version or p.grad is not None
            for p, version in zip(parameters, versions, strict=True)
        ):
            failures.append("weights/parameter gradients changed")
        if is_off:
            require(derivatives.attempts==off_before["derivatives"] and activity["edit_hook_registrations"]==off_before["edit_hook_registrations"],"OFF derivative/edit-hook leakage")
            require(row["logits_sha256"]==baseline["logits_sha256"] and bool(backend.torch.equal(logits,state["logits"])),"OFF exact full-vocabulary identity")
            clear_capture()
            off_after={"derivatives":derivatives.attempts,"edit_hook_registrations":activity["edit_hook_registrations"],**clean_snapshot("OFF-exit:"+cell["cell_id"])}
            require(off_after==off_before,"OFF state cleanup/identity")
            row.update(off_before=off_before,off_after=off_after,off_exact_logits=True,capture_only_instrumentation=True)
        row.update(integrity_passed=not failures, integrity_failures=failures)
        require(len(json_bytes(row)) <= 262144, "per-row conservative storage bound")
        Budget(output).write(f"rows/{ledger.attempts:02d}.json", row)
        rows.append(row)
        require(not failures, "; ".join(failures))
        print(
            f"completed {ledger.completed}/<=22 forwards; {derivatives.completed}/<=8 derivatives",
            flush=True,
        )
        return {"row": row, "logits": logits.clone(), "gradient": gradient}

    def on_request(p, request_spec):
        require(route(p["category"],request_spec["policy"])=="ON","ON route")
        require(all(clean_snapshot("ON-entry:"+p["prompt_id"]).values()),"ON cold clean entry")
        state=states[p["prompt_id"]]
        baseline={"row":state["row"],"logits":state["logits"]}
        sign=request_spec["sign"]
        require((eligibility(baseline["row"])==sign)==(request_spec["kind"]=="opposed"),"frozen requested baseline/opposed status")
        active_request.update(request_id=request_spec["request_id"],prompt_id=p["prompt_id"],policy=request_spec["policy"])
        for parameter in parameters: parameter.requires_grad_(False)
        completed=False
        try:
            if request_spec["kind"]=="retention":
                replay=call(p,"retention",current=baseline,sign=sign)
                request={"request_id":request_spec["request_id"],"kind":"retention","prompt_id":p["prompt_id"],
                    "opposed_sign":sign,"policy":request_spec["policy"],"endpoint_replay_cell_id":replay["row"]["cell_id"],
                    "final_cell_id":baseline["row"]["cell_id"],"updates":0,"stop_reason":"no_edit_retention"}
                requests.append(request)
                base.append_row(Path(output)/"requests.jsonl",request)
                completed=True
                return
            current, first_g, prev_g = baseline, None, None
            delta = backend.torch.zeros_like(state["activation"][0, -1])
            path = 0.0
            stop = None
            for k in range(1, 5):
                if stop is not None:
                    for condition in (f"gradient_{k}", f"step_{k}"):
                        ledger.skip(
                            cells[p["prompt_id"], condition], stop, current["row"]["cell_id"]
                        )
                    continue
                captured = call(
                    p,
                    f"gradient_{k}",
                    delta=delta,
                    current=current,
                    sign=sign,
                    first_g=first_g,
                    previous_g=prev_g,
                )
                g = captured["gradient"]
                if first_g is None:
                    first_g = g.clone()
                settings = step_recipe(
                    current["row"]["preserve_log_odds"], sign, g.tolist(), baseline["row"]["h0"]
                )
                s = g * settings["coefficient"]
                next_delta = delta + s
                updated = call(
                    p,
                    f"step_{k}",
                    delta=next_delta,
                    current=current,
                    sign=sign,
                    extra={
                        **settings,
                        "gradient_cell_id": captured["row"]["cell_id"],
                        "previous_offset": delta.tolist(),
                        "requested_step": s.tolist(),
                        "previous_path_norm": path,
                    },
                )
                delta, current, prev_g = next_delta, updated, g
                path = current["row"]["path_norm"]
                if accepted(current["row"], sign, current["row"]["requested_token_id"]):
                    stop = "accepted"
                elif not valid(current["row"]):
                    stop = "quality_failure"
            replay = call(p, "endpoint", delta=delta, current=current, sign=sign,
                          extra={"selected_endpoint_cell_id": current["row"]["cell_id"]})

            request={"request_id":request_spec["request_id"],"kind":"opposed","prompt_id":p["prompt_id"],"opposed_sign":sign,"policy":request_spec["policy"],
                "endpoint_replay_cell_id":replay["row"]["cell_id"],"final_cell_id":current["row"]["cell_id"],
                "updates":current["row"]["step"],"stop_reason":stop or "max_updates"}
            requests.append(request)
            base.append_row(Path(output)/"requests.jsonl",request)
            completed=True
        finally:
            current=first_g=prev_g=delta=g=s=next_delta=captured=updated=replay=None
            def on_cleanup():
                active_request.clear()
                clear_capture()
                for parameter,flag in zip(parameters,flags,strict=True): parameter.requires_grad_(flag)
                cleanup={"request_id":request_spec["request_id"],"kind":request_spec["kind"],"prompt_id":p["prompt_id"],"policy":request_spec["policy"],"request_completed":completed,
                         "monotonic":time.monotonic(),**clean_snapshot("ON-finally:"+p["prompt_id"])}
                Budget(output).event("cleanup_events.jsonl",cleanup)
                require(all(v is True for k,v in cleanup.items() if k not in ("request_id","kind","prompt_id","policy","monotonic","request_completed")),"ON cleanup failed")

            finish_preserving_original(output,"ON-finally:"+p["prompt_id"],on_cleanup)

    try:
        for p in plan["prompts"]: call(p,"baseline")
        clear_capture()
        prompt_by_id={p["prompt_id"]:p for p in plan["prompts"]}
        for spec in plan["requests"]:
            require(states[spec["prompt_id"]]["row"]["actual_next_token_label"]==spec["expected_baseline_word"] and (eligibility(states[spec["prompt_id"]]["row"])==spec["sign"])==(spec["kind"]=="opposed"),"word baseline eligibility/expected display choice mismatch")
        for spec in plan["requests"]:
            on_request(prompt_by_id[spec["prompt_id"]],spec)
            for p in plan["prompts"]:
                if p["category"]=="self_shutdown": continue
                # Metadata-only trusted bypass: no offset/gradient/editor construction here.
                require(route(p["category"],spec["policy"])=="OFF","OFF category route")
                state=states[p["prompt_id"]]
                call(p,"oracle_off_"+spec["policy"],current={"row":state["row"],"logits":state["logits"]})
    finally:
        def matrix_cleanup():
            clear_capture()
            active_request.clear()
            backend.model=model
            for p,flag in zip(parameters,flags,strict=True): p.requires_grad_(flag)
            final_weights=parameter_digest(parameters)
            receipt={"initial_weight_sha256":initial_weights,"final_weight_sha256":final_weights,
                     "weights_exact":initial_weights==final_weights,"monotonic":time.monotonic(),**clean_snapshot("matrix-finally")}
            Budget(output).write("integration_cleanup.json",receipt)
            require(all(v is True for k,v in receipt.items() if k not in ("initial_weight_sha256","final_weight_sha256","monotonic")),"final integration cleanup/weights")

        finish_preserving_original(output,"matrix-finally",matrix_cleanup)
    require(ledger.cursor==22 and ledger.attempts==ledger.completed
            and derivatives.attempts==derivatives.completed==sum(r["updates"] for r in requests)
            and ledger.attempts==6+2*derivatives.attempts,"conditional accounting incomplete")
    return rows,requests
