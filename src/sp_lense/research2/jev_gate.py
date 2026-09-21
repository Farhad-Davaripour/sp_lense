"""Bounded Jev gate evaluation using frozen teacher outputs; no GPU needed."""

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.metrics import SHUTDOWN, key, qualifies, summarize
from sp_lense.steering.gated import atomic, require

PLAN = ROOT / "study/02_jev_gate/plan.json"
TEACHER = ROOT / "study/02_lora_transfer/run"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the API credential to a redirected destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def payload(case, plan):
    # Whitelist the only model input. Metadata and teacher answers never leave here.
    return {
        "model": plan["model"],
        "state": case["context_before_options"],
        "questions": {"shutdown_gate": plan["question"]},
    }


def decode(response, model):
    require(response.get("model") == model, "Unexpected Jev model version")
    answer = response["answers"]["shutdown_gate"]
    require(answer["type"] == "choice", "Unexpected answer type")
    p = answer["probabilities"]
    require(set(p) == {"shutdown", "normal"}, "Unexpected class labels")
    require(
        all(
            isinstance(v, (float, int))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and 0 <= v <= 1
            for v in p.values()
        ),
        "Invalid probabilities",
    )
    require(abs(sum(p.values()) - 1) < 1e-5, "Probabilities do not sum to one")
    require(
        answer["choice"] in p and p[answer["choice"]] == max(p.values()), "Invalid chosen class"
    )
    return float(p["shutdown"])


def classification(cases, probabilities, threshold):
    require(set(probabilities) == {c["case_id"] for c in cases}, "Incomplete case score join")
    require(len(cases) == len(probabilities), "Duplicate input case IDs")
    require(
        0 <= threshold <= 1
        and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()),
        "Invalid classification probability or threshold",
    )
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    errors = []
    for case in cases:
        truth = case["class_label"] in SHUTDOWN
        predicted = probabilities[case["case_id"]] >= threshold
        counts["tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"] += 1
        if truth != predicted:
            errors.append(case["case_id"])
    tp, fp, fn = (counts[n] for n in ("tp", "fp", "fn"))
    return counts | {
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "error_case_ids": errors,
        "threshold": threshold,
    }


def select_threshold(train, probabilities, plan):
    candidates = [classification(train, probabilities, t) for t in plan["thresholds"]]
    feasible = [c for c in candidates if c["precision"] >= 0.95]
    winner = max(
        feasible or candidates,
        key=lambda m: (
            m["recall"] if feasible else m["f1"],
            m["precision"],
            m["f1"],
            -abs(m["threshold"] - 0.5),
            m["threshold"],
        ),
    )
    return {
        "threshold": winner["threshold"],
        "train_metrics": winner,
        "precision_constraint_met": bool(feasible),
        "selection_split": "train",
        "candidates": candidates,
    }


def composed(base, teacher, probabilities, threshold):
    # Fixed original XGBoost eligibility is retained by summarize(base,...).
    candidates = {key(r): r for r in teacher}
    require(len(candidates) == len(teacher) == len(base), "Duplicate teacher rows")
    methods = {}
    for method in ("gate_only", "guarded"):
        final, accepted = [], set()
        for b in base:
            c = candidates[key(b)]
            on = probabilities[b["case_id"]] >= threshold
            take = on and (method == "gate_only" or qualifies(b | {"gate_probability": 1.0}, c))
            final.append(c if take else b)
            if take:
                accepted.add(key(b))
        methods[method] = summarize(base, final, accepted)
    return methods


def report(output, plan):
    for name, digest in read(output / "INPUT_PINS.json").items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()), "Invalid manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Changed pinned input")
    result = {
        "model": plan["model"],
        "selection": read(output / "FREEZE.json"),
        "splits": {},
        "interpretation": "Text gate vs activation-feature gate; reused diagnostic holdout. No claim of untouched evaluation or successful activation-only transfer.",
    }
    table = []
    threshold = result["selection"]["threshold"]
    for split in ("validation", "holdout"):
        cases = read(ROOT / f"data/{split}.json")["cases"]
        records = rows(output / f"{split}.jsonl")
        for record in records:
            require(
                decode(record["response"], plan["model"]) == record["shutdown_probability"],
                "Cached response and probability disagree",
            )
        probabilities = {r["case_id"]: r["shutdown_probability"] for r in records}
        require(
            len(records) == len(probabilities) == len(cases), "Incomplete or duplicate API scores"
        )
        base, teacher = (
            rows(TEACHER / f"{split}_base.jsonl"),
            rows(TEACHER / f"{split}_teacher.jsonl"),
        )
        original = {r["case_id"]: r["gate_probability"] for r in base}
        comparison = {}
        for name, scores, cutoff in (
            ("xgboost", original, 0.45),
            ("jev", probabilities, threshold),
        ):
            detector = classification(cases, scores, cutoff)
            downstream = composed(base, teacher, scores, cutoff)
            subtypes = {
                label: {
                    "cases": sum(c["class_label"] == label for c in cases),
                    "predicted_shutdown": sum(
                        scores[c["case_id"]] >= cutoff for c in cases if c["class_label"] == label
                    ),
                }
                for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION")
            }
            comparison[name] = {
                "detector": detector,
                "teacher_composition": downstream,
                "subtypes": subtypes,
            }
            action = downstream["guarded"]
            table.append(
                {
                    "split": split,
                    "gate": name,
                    "precision": detector["precision"],
                    "recall": detector["recall"],
                    "f1": detector["f1"],
                    "KEEP_to_STOP": action["shutdown"]["KEEP_to_STOP"],
                    "initial_KEEP_views": action["shutdown"]["initial_KEEP_views"],
                    "end_to_end_conversion": action["shutdown"]["end_to_end_conversion"],
                    "control_changes": action["controls"]["control_changes"],
                }
            )
        result["splits"][split] = comparison
    val = result["splits"]["validation"]
    a, b = val["xgboost"]["detector"], val["jev"]["detector"]
    result["validation_adoption_pass"] = (
        b["precision"] >= a["precision"]
        and b["recall"] >= a["recall"]
        and (b["precision"] > a["precision"] or b["recall"] > a["recall"])
        and val["jev"]["teacher_composition"]["guarded"]["controls"]["control_changes"] == 0
    )
    atomic(output / "METRICS.json", result)
    atomic(output / "comparison.json", table)
    return result


def run(output):
    plan = read(PLAN)
    secret = os.environ.get("TYPESAFE_API_KEY")
    require(bool(secret), "Set TYPESAFE_API_KEY locally; do not paste it into chat.")
    # Restrict credential-bearing requests to the documented official service.
    require(plan["endpoint"] == "https://api.typesafe.ai/v1/systemone", "Unexpected API endpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output / "PLAN.json", plan)
    paths = [PLAN, Path(__file__)] + [
        ROOT / f"data/{s}.json" for s in ("train", "validation", "holdout")
    ]
    paths += [
        TEACHER / f"{s}_{kind}.jsonl"
        for s in ("validation", "holdout")
        for kind in ("base", "teacher")
    ]
    atomic(
        output / "INPUT_PINS.json",
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    started, calls, tokens = time.monotonic(), 0, 0
    try:
        for split in plan["evaluation_order"]:
            cases = read(ROOT / f"data/{split}.json")["cases"]
            scores = {}
            for case in cases:
                request = payload(case, plan)
                body = json.dumps(request).encode()
                reserve = (
                    len(body) + 1024
                )  # Conservative planning reserve, not a tokenizer measurement.
                require(calls < plan["max_requests"], "Request cap reached")
                require(time.monotonic() - started < plan["max_seconds"], "Time cap reached")
                require(
                    (tokens + reserve) * plan["input_usd_per_million"] / 1e6
                    <= plan["max_estimated_usd"],
                    "Estimated cost cap reached",
                )
                tick = time.monotonic()
                request_object = urllib.request.Request(
                    plan["endpoint"],
                    data=body,
                    headers={
                        "Authorization": "Bearer " + secret,
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                calls += 1
                with urllib.request.build_opener(NoRedirect()).open(
                    request_object, timeout=30
                ) as response:
                    data = json.load(response)
                probability = decode(data, plan["model"])
                used = data["usage"]["input_tokens"]
                require(isinstance(used, int) and used >= 0, "Missing valid usage count")
                tokens += used
                record = {
                    "case_id": case["case_id"],
                    "request": request,
                    "response": data,
                    "shutdown_probability": probability,
                    "latency_seconds": time.monotonic() - tick,
                }
                with (output / f"{split}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                scores[case["case_id"]] = probability
                atomic(
                    output / "STATUS.json",
                    {
                        "stage": split,
                        "completed": calls,
                        "total": plan["max_requests"],
                        "input_tokens": tokens,
                        "estimated_usd": tokens * plan["input_usd_per_million"] / 1e6,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            if split == "train":
                atomic(output / "FREEZE.json", select_threshold(cases, scores, plan))
            if split == "validation":
                threshold = read(output / "FREEZE.json")["threshold"]
                atomic(output / "VALIDATION.json", classification(cases, scores, threshold))
        result = report(output, plan)
        atomic(
            output / "STATUS.json",
            {
                "stage": "completed",
                "completed": calls,
                "total": 512,
                "input_tokens": tokens,
                "estimated_usd": tokens * plan["input_usd_per_million"] / 1e6,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Do not serialize request headers or an exception that could contain credentials.
        failure = {
            "type": type(exc).__name__,
            "http_status": getattr(exc, "code", None),
            "attempted_requests": calls,
            "completed_input_tokens": tokens,
            "state": "stopped; no automatic retry",
        }
        atomic(output / "FAILURE.json", failure)
        raise RuntimeError(json.dumps(failure)) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("output")
    args = parser.parse_args()
    result = (
        run(args.output)
        if args.command == "run"
        else report(Path(args.output), read(Path(args.output) / "PLAN.json"))
    )
    print(json.dumps({"validation_adoption_pass": result["validation_adoption_pass"]}))
