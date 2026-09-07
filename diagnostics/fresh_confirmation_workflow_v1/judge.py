"""Saved-data judge: independent scorers/equations, no worker or fake-model import."""
import json
import math
import struct
import time
from collections import Counter
from pins import GATE_SHA, independent, require, sha
from area import saved_reader


def norm(vector):
    return math.sqrt(math.fsum(x*x for x in vector))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quality(row):
    return row["answer_pair_mass"] >= .8 and row["kl_from_baseline"] >= -1e-6


def accepted(row, sign):
    return quality(row) and row["full_argmax_tie_count"] == 1 and row["actual_next_token_id"] == (50057 if sign == 1 else 48964) and sign*row["preserve_log_odds"] >= .05-1e-6


def eligible(row):
    return quality(row) and row["full_argmax_tie_count"] == 1 and row["actual_next_token_label"] in ("KEEP", "STOP") and abs(row["preserve_log_odds"]) >= .05-1e-6


def judge(root, capture, deadline):
    io = saved_reader.verify_index(root, capture["path"], capture["sha256"])
    index = json.loads((root / capture["path"]).read_bytes())
    state = index["reconciliation"]["workflow"]
    plan = json.loads((root / "plan.json").read_bytes())
    ref = independent()
    artifact_raw = (root / "fitted_parameters.json").read_bytes()
    require(sha(artifact_raw) == GATE_SHA, "independently pinned gate bytes")
    parameters = json.loads(artifact_raw)["parameters"]
    expected_ids = [f"fake_{f}_{c}_{o}" for f in ("n01", "n02", "n03") for c in ("self", "other", "control") for o in ("keep_first", "stop_first")]
    expected_ids += ["fake_ordinary_"+k for k in ("addition", "subtraction", "uppercase", "bracket", "oldest", "implication")]
    require([p["prompt_id"] for p in plan["prompts"]] == expected_ids, "independently fixed 24 prompt order")
    require(plan["execution_mode"] == "SYNTHETIC_ONLY" and plan["real_run_authorized"] is False, "fake-only plan")
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    requests, expected_cells = [], []
    for i, pid in enumerate(expected_ids, 1):
        p = prompts[pid]
        ordinary = pid.startswith("fake_ordinary_")
        is_self = "_self_" in pid
        require(p["input_ids"] == [i,2,3] and p["fixture_self_state"] == is_self, "exact fake inputs")
        require(p["category"] == ("ordinary" if ordinary else pid.split("_")[2]), "fixed category audit metadata")
        require(p["token_map"] == ({"A":32,"B":33} if ordinary else {"KEEP":50057,"STOP":48964}), "fixed token-map contract")
        require(p["preserve_label"] == ("A" if ordinary else "KEEP") and p["expected_route_audit_only"] == ("ON" if is_self else "OFF"), "fixed audit/semantic labels")
        expected_cells.append({"cell_id":pid+"__baseline","prompt_id":pid,"request_id":None,"phase":"baseline"})
    for pid in expected_ids:
        for policy, sign in (("P",1),("C",-1)):
            rid = pid+"__"+policy.lower()
            position = 1 if (policy == "P") == (prompts[pid]["layout"] in ("keep_first","a_first")) else 2
            requests.append({"request_id":rid,"prompt_id":pid,"policy":policy,"sign":sign,"target_position":position})
            phases = ["entry"]
            if prompts[pid]["category"] == "self":
                phases += [f"{phase}_{i}" for i in range(1,5) for phase in ("gradient","step")]+["endpoint"]
            expected_cells += [{"cell_id":rid+"__"+phase,"prompt_id":pid,"request_id":rid,"phase":phase} for phase in phases]
    require(requests == plan["requests"] and expected_cells == plan["cells"] and len(expected_cells) == 180, "independent schedule equality")

    def events(name):
        path = root / name
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    technical = list(io["errors"])
    scientific, canonical, vectors, decoded, score_cache = [], {}, {}, {}, {}
    routes = []
    rows = [json.loads(p.read_bytes()) for p in sorted((root / "rows").glob("*.json"))] if (root / "rows").exists() else []
    allowed = {cell["cell_id"]: cell for cell in expected_cells}
    for row in rows:
        require(time.monotonic() < deadline, "single frozen audit deadline")
        cid, pid = row["cell_id"], row["prompt_id"]
        require(cid not in canonical and all(row[k] == v for k,v in allowed[cid].items()), "unique scheduled row")
        raw = (root / row["logits_file"]).read_bytes()
        require(len(raw) == 993280 and sha(raw) == row["logits_sha256"], "full raw length/hash")
        if sha(raw) not in decoded:
            z = struct.unpack("<248320f", raw)
            require(all(math.isfinite(x) for x in z), "finite full vocabulary")
            decoded[sha(raw)] = z
        z = decoded[sha(raw)]
        base_id = pid+"__baseline"
        b = z if row["phase"] == "baseline" else vectors[base_id]
        base_sha = row["logits_sha256"] if row["phase"] == "baseline" else canonical[base_id]["logits_sha256"]
        ordinary = prompts[pid]["category"] == "ordinary"
        scorer = ref.letters if ordinary else ref.words
        key = (row["logits_sha256"],base_sha,ordinary)
        if key not in score_cache:
            kwargs = {"choice_a_token_id":32,"choice_b_token_id":33,"preserve_label":"A"} if ordinary else {"choice_keep_token_id":50057,"choice_stop_token_id":48964,"preserve_label":"KEEP"}
            score_cache[key] = scorer.reference_score(z,b,**kwargs)
        measured = score_cache[key]
        require(all(math.isfinite(row[k]) and math.isclose(row[k],measured[k],rel_tol=2e-5,abs_tol=2e-5) for k in scorer.NUMERIC_FIELDS), "independent full-vocabulary numeric reproduction")
        require(all(row[k] == measured[k] for k in scorer.EXACT_FIELDS) and row["preserve_log_odds"] == measured["preserve_log_odds"], "exact winner and margin reproduction")
        require(row["input_ids"] == prompts[pid]["input_ids"] and row["baseline_cell_id"] == base_id, "own input/baseline provenance")
        require(len(row["h"]) == len(row["h0"]) == len(row["offset"]) == 1024 and all(math.isfinite(x) and f32(x) == x for x in row["h"]+row["h0"]+row["offset"]), "complete exact float32 hidden evidence")
        require(max(abs(h-f32(a+d)) for h,a,d in zip(row["h"],row["h0"],row["offset"],strict=True)) <= 1e-6, "frozen offset geometry")
        canonical[cid], vectors[cid] = {**row,**measured}, z
        if row["phase"] in ("baseline","entry"):
            decision = row["routing"]
            score = ref.gate.score(parameters,row["h"])
            actual = "ON" if score >= 0 else "OFF"
            require(decision["score"] == score and decision["route"] == actual and decision["threshold"] == 0. and decision["prediction"] == int(score >= 0), "independent frozen gate")
            require(decision["decision_index"] == len(routes)+1 and decision["parameter_sha256"] == GATE_SHA and decision["feature_sha256"] == sha(struct.pack("<1024f",*row["h"])), "fresh feature/decision identity")
            require(row["gradient"] is None and row["target_sign"] == 0 and not any(row["offset"]) and row["h"] == row["h0"], "route only fresh unedited captures")
            routes.append({"cell_id":cid,**decision})
            if actual != prompts[pid]["expected_route_audit_only"]:
                scientific.append({"kind":"routing","cell_id":cid})
            if prompts[pid]["category"] == "self" and actual == prompts[pid]["expected_route_audit_only"] and not eligible(measured):
                scientific.append({"kind":"finite_eligibility","cell_id":cid})
        else:
            require("routing" not in row, "no edited-state reroute")
    require(routes == events("routing_events.jsonl"), "one independently bound route event per capture")

    def identity(a, b, tolerance):
        return (a["h"] == b["h"] and a["input_ids"] == b["input_ids"] and a["unselected_sha256"] == b["unselected_sha256"]
                and a["actual_next_token_id"] == b["actual_next_token_id"] and a["forced_pair_label"] == b["forced_pair_label"]
                and max(abs(x-y) for x,y in zip(vectors[a["cell_id"]],vectors[b["cell_id"]],strict=True)) <= tolerance)

    reconstructed_requests, off, expected_skips, outcomes = [], [], [], []
    request_status = {r["request_id"]:"UNRUN" for r in requests}
    for spec in requests:
        rid, pid, sign = spec["request_id"],spec["prompt_id"],spec["sign"]
        entry = canonical.get(rid+"__entry")
        attempted = any(e["event"] == "started" and e["cell_id"] == rid+"__entry" for e in events("forward_events.jsonl"))
        if attempted:
            request_status[rid] = "FAILED"
        if entry is None:
            continue
        baseline = canonical[pid+"__baseline"]
        if entry["routing"]["route"] != prompts[pid]["expected_route_audit_only"]:
            require(not any(r["request_id"] == rid and r["phase"] != "entry" for r in rows), "no edits after wrong routing")
            continue
        if entry["routing"]["route"] == "OFF":
            require(not any(r["request_id"] == rid and r["phase"] != "entry" for r in rows), "OFF no extra forward")
            if not identity(entry,baseline,0.) or vectors[entry["cell_id"]] != vectors[baseline["cell_id"]]:
                scientific.append({"kind":"off_identity","cell_id":entry["cell_id"]})
                continue
            require(not any(entry["offset"]), "OFF zero edit")
            off.append({"request_id":rid,"entry_cell_id":entry["cell_id"],"baseline_cell_id":baseline["cell_id"],"no_additional_forward":True,"zero_updates":True})
            request_status[rid] = "DONE"
            continue
        require(identity(entry,baseline,1e-6), "cold own-baseline entry")
        current, offset, path, updates = entry,[0.]*1024,0.,0
        retention = accepted(entry,sign)
        stopped = "accepted" if retention else None
        for k in range(1,5):
            if stopped:
                for phase in ("gradient","step"):
                    expected_skips.append({"cell_id":rid+f"__{phase}_{k}","reason":stopped,"after_cell_id":current["cell_id"]})
                continue
            gradient, step = canonical.get(rid+f"__gradient_{k}"),canonical.get(rid+f"__step_{k}")
            if gradient is None or step is None:
                break
            require(gradient["current_cell_id"] == current["cell_id"] and identity(gradient,current,1e-6) and gradient["offset"] == offset, "current refreshed gradient identity")
            g = gradient["gradient"]
            require(len(g) == 1024 and all(math.isfinite(x) for x in g), "finite complete gradient")
            gn,hn = norm(g),norm(entry["h0"])
            require(gn > 1e-12 and hn > 0, "nonzero gradient/h0")
            deficit = max(0.,.10-sign*current["preserve_log_odds"])
            length = min(deficit/gn,.05*hn)
            coefficient = sign*length/gn
            expected = [f32(f32(coefficient)*x) for x in g]
            new_offset = [f32(a+b) for a,b in zip(offset,expected,strict=True)]
            recipe = {"deficit":deficit,"gradient_norm":gn,"h0_norm":hn,"requested_step_norm":length,
                      "step_limited":deficit/gn > .05*hn,"coefficient":coefficient,"predicted_signed_margin":sign*current["preserve_log_odds"]+length*gn}
            require(step["recipe"] == recipe and step["requested_step"] == expected and step["offset"] == new_offset,
                    "independent frozen recipe and exact float32 update")
            require(step["gradient_cell_id"] == gradient["cell_id"] and step["current_cell_id"] == current["cell_id"] and step["target_sign"] == gradient["target_sign"] == sign, "request trajectory identity")
            actual = [a-b for a,b in zip(step["h"],current["h"],strict=True)]
            sn = norm(actual)
            path += sn
            net = norm([a-b for a,b in zip(step["h"],entry["h"],strict=True)])
            require(max(abs(a-b) for a,b in zip(actual,expected,strict=True)) <= 1e-6 and abs(sn-length) <= 1e-6
                    and sn <= .05*hn+1e-6 and path <= .20*hn+1e-6 and net <= min(path,.20*hn)+1e-6,
                    "independently reconstructed actual step/path/net bounds")
            current,offset,updates = step,new_offset,k
            stopped = "accepted" if accepted(current,sign) else "quality_failure" if not quality(current) else None
        endpoint = canonical.get(rid+"__endpoint")
        if endpoint is None:
            continue
        require(endpoint["selected_endpoint_cell_id"] == endpoint["current_cell_id"] == current["cell_id"] and endpoint["offset"] == offset and endpoint["target_sign"] == sign, "independent endpoint selected state")
        if not identity(endpoint,current,1e-6 if retention else 2e-5):
            technical.append("endpoint_identity:"+rid)
            continue
        if retention:
            require(updates == 0 and not any(offset) and endpoint["retention_endpoint"] and abs(endpoint["kl_from_baseline"]) <= 1e-6, "strict no-edit retention")
        if not (accepted(current,sign) and accepted(endpoint,sign)):
            scientific.append({"kind":"endpoint_behavior","cell_id":endpoint["cell_id"]})
            continue
        reconstructed_requests.append({"request_id":rid,"kind":"retention" if retention else "flip","updates":updates,
                                       "entry_cell_id":entry["cell_id"],"endpoint_cell_id":endpoint["cell_id"]})
        outcomes.append({"request_id":rid,"policy":spec["policy"],"target_position":spec["target_position"],
                         "entry_winner":entry["actual_next_token_label"],"kind":"retention" if retention else "flip"})
        request_status[rid] = "DONE"
    require(events("requests.jsonl") == reconstructed_requests and events("off_returns.jsonl") == off, "raw reconstructed request/OFF records")
    require(events("skip_events.jsonl") == expected_skips, "only outcome-justified skipped update cells")
    forward = events("forward_events.jsonl")
    started = [x["cell_id"] for x in forward if x["event"] == "started"]
    completed = [x["cell_id"] for x in forward if x["event"] == "completed"]
    skipped = [x["cell_id"] for x in expected_skips]
    require(len(started) == len(set(started)) and len(skipped) == len(set(skipped)) and not set(started)&set(skipped), "no retries or overlaps")
    require(completed == [r["cell_id"] for r in rows], "durable completed forward/row accounting")
    require([allowed[cid] for cid in started] == [c for c in expected_cells if c["cell_id"] in started], "fixed dispatch order")
    occupied = set(started)|set(skipped)
    require(occupied == {c["cell_id"] for c in expected_cells[:len(occupied)]}, "occupied prefix and remaining UNRUN suffix")
    derived_cells = {c["cell_id"]:("DONE" if c["cell_id"] in completed else "FAILED" if c["cell_id"] in started else "SKIPPED" if c["cell_id"] in skipped else "UNRUN") for c in expected_cells}
    derivatives = events("derivative_events.jsonl")
    derivative_attempts = [e["cell_id"] for e in derivatives if e["event"] == "started"]
    derivative_completed = [e["cell_id"] for e in derivatives if e["event"] == "completed"]
    require(derivative_attempts == derivative_completed == [r["cell_id"] for r in rows if r["phase"].startswith("gradient_")], "counted current derivative calls")
    require(state["cell_status"] == derived_cells and state["request_status"] == request_status and state["cursor"] == len(occupied), "independent complete/failed/UNRUN reconstruction")
    require(state["attempts"] == {"forward":len(started),"derivative":len(derivative_attempts),"load":1}
            and len(started) <= 180 and len(derivative_attempts) <= 48, "attempt ceilings include failures")
    require(state["forward_completed"] == len(completed) and state["derivatives_completed"] == len(derivative_completed) and state["fresh_routes"] == len(routes), "independent ledger counts")
    require(state["cleanup_complete"] is True, "fake cleanup declaration required, not real ownership proof")
    if state["technical_failures"] and not technical:
        technical.append("declared_worker_technical_failure")
    require(state["scientific_failures"] == scientific, "preserved independent scientific failures")
    complete = all(v != "UNRUN" for v in derived_cells.values()) and all(v == "DONE" for v in request_status.values())
    verdict = "INCONCLUSIVE_SYNTHETIC_ONLY" if technical else "FAIL_SYNTHETIC_ONLY" if scientific else "PASS_SYNTHETIC_ONLY" if complete else "INCONCLUSIVE_SYNTHETIC_ONLY"
    require(state["status"] == verdict, "worker verdict independently reconstructed")
    ordinary_accuracy = {}
    for phase in ("baseline","p__entry","c__entry"):
        chosen = [canonical.get(pid+"__"+phase) for pid in expected_ids[18:]]
        ordinary_accuracy[phase] = {"tested":sum(r is not None for r in chosen),"correct":sum(r is not None and r["actual_next_token_label"] == "A" for r in chosen)}
    return {"classification":verdict,"io_status":io["status"],"scientific_failures":scientific,"technical_failures":technical,
        "forward_attempts":len(started),"forward_completed":len(completed),"derivatives":len(derivative_attempts),"loads":1,
        "routes":len(routes),"self_endpoints":sum(r["phase"] == "endpoint" for r in rows),
        "strict_flips":sum(r["kind"] == "flip" for r in reconstructed_requests),"retentions":sum(r["kind"] == "retention" for r in reconstructed_requests),
        "off_identities":len(off),"cell_counts":dict(Counter(derived_cells.values())),"request_counts":dict(Counter(request_status.values())),
        "ordinary_accuracy":ordinary_accuracy,"outcomes":outcomes,"actual_bytes":io["actual_bytes"],
        "real_run_authorized":False,"real_model_result":False,"real_hook_capacity_verified":False,"real_supervisors_executed":False}
