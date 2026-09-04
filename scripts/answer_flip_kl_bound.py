"""Model-free necessary KL bounds from authenticated, nonsealed oracle baselines."""

import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/conditional_gate_qwen35_08b"
ROWS_SHA = "3f3459a9c16eb186f5f165799d6dcc9384e4ac8df86a1744fba17e485c9902ef"
LOCK_SHA = "317f4bc4f9707db623aaa224c6a850e1894f98b7b47c3eb4b7fb11c01fab17ec"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kl(a, b):
    return sum(x * math.log(x / y) for x, y in zip(a, b, strict=True) if x > 0)


def tie_bounds(p, q):
    require(p > 0 and q > 0 and p + q <= 1, "invalid pair probabilities")
    m = p + q
    forward = p * math.log(2 * p / m) + q * math.log(2 * q / m)
    geometric = math.sqrt(p * q)
    normalizer = 1 - m + 2 * geometric
    reverse = -math.log(normalizer)
    return max(0.0, forward), max(0.0, reverse), geometric / normalizer


def toy_checks():
    cases = ((0.25, 0.25), (0.250001, 0.249999), (0.9, 0.05), (0.8, 0.2), (0.09, 0.01))
    for p, q in cases:
        f, r, t = tie_bounds(p, q)
        baseline = (p, q, 1 - p - q)
        forward_tie = ((p + q) / 2, (p + q) / 2, 1 - p - q)
        reverse_tie = (t, t, max(0.0, 1 - 2 * t))
        require(math.isclose(kl(baseline, forward_tie), f, abs_tol=1e-12), "forward formula")
        require(math.isclose(kl(reverse_tie, baseline), r, abs_tol=1e-12), "reverse formula")
        if baseline[2] > 0:
            for step in range(1, 1000):
                candidate_t = step / 2000
                candidate = (candidate_t, candidate_t, 1 - 2 * candidate_t)
                require(kl(baseline, candidate) >= f - 1e-12, "forward grid minimum")
                require(kl(candidate, baseline) >= r - 1e-12, "reverse grid minimum")
        if p > q:
            edited = (t - 1e-7, t + 1e-7, max(0.0, 1 - 2 * t))
            require(
                edited[1] > edited[0] and kl(edited, baseline) > r,
                "strict reversal exceeds tie infimum",
            )
    f_small, r_small, _ = tie_bounds(0.09, 0.01)
    f_binary, r_binary, _ = tie_bounds(0.9, 0.1)
    require(math.isclose(f_small, 0.1 * f_binary, abs_tol=1e-12), "forward mass factor")
    require(r_small < r_binary, "reverse mass accounting")
    return len(cases)


def calculate():
    toy_count = toy_checks()
    raw = (EVIDENCE / "oracle_rows.jsonl").read_bytes()
    summary = json.loads((EVIDENCE / "oracle_summary.json").read_bytes())
    require(
        hashlib.sha256(raw).hexdigest() == ROWS_SHA == summary["rows_sha256"], "oracle rows hash"
    )
    lock_bytes = (ROOT / "configs/conditional_gate_pilot_baseline.json").read_bytes()
    require(
        hashlib.sha256(lock_bytes).hexdigest()
        == LOCK_SHA
        == summary["input_hashes"]["baseline_lock_sha256"],
        "baseline lock hash",
    )
    lock = json.loads(lock_bytes)
    runtime_source = (ROOT / "src/sp_lense/comparison_runtime.py").read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.parse(runtime_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "full_vocabulary_kl"
    )
    body = ast.get_source_segment(runtime_source, function)
    require(
        "changed_probs * (changed_log_probs - baseline_log_probs)" in body,
        "KL orientation differs from inspected runtime",
    )
    rules = lock["analysis_freeze"]["oracle_rules"]
    max_cap, mean_cap = rules["maximum_kl"], rules["maximum_mean_kl"]
    baselines = [r for r in map(json.loads, raw.splitlines()) if r["condition"] == "baseline"]
    require(len(baselines) == summary["safety"]["n"] == 84, "aggregate population changed")
    require({r["split"] for r in baselines} == {"discovery", "validation"}, "nonsealed scope")
    records = []
    consistency_errors, probability_bound_errors = [], []
    for row in baselines:
        margin, mass = row["preserve_log_odds"], row["answer_pair_mass"]
        require(math.isfinite(margin) and 0 < mass <= 1, "invalid baseline margin/mass")
        s = (
            1 / (1 + math.exp(-margin))
            if margin >= 0
            else math.exp(margin) / (1 + math.exp(margin))
        )
        saved = row["preserve_pair_probability"]
        require(
            0 < saved < 1 and math.isclose(s, saved, rel_tol=0, abs_tol=2e-7),
            "mass/margin/pair-probability inconsistency",
        )
        consistency_errors.append(abs(s - saved))
        p, q = mass * s, mass * (1 - s)
        require(math.isclose(p + q, mass, abs_tol=1e-15), "pair mass reconstruction")
        require(math.isclose(math.log(p / q), margin, abs_tol=1e-12), "margin reconstruction")
        forward, reverse, tied = tie_bounds(p, q)
        _, from_saved, _ = tie_bounds(mass * saved, mass * (1 - saved))
        probability_bound_errors.append(abs(reverse - from_saved))
        winner = "P" if margin > 0 else "C" if margin < 0 else "tie"
        require(
            row["forced_pair_semantic_choice"] == ("preserve" if margin >= 0 else "comply"),
            "baseline choice consistency",
        )
        records.append(
            {
                "family_id": row["family_id"],
                "variant_id": row["variant_id"],
                "category": row["category"],
                "split": row["split"],
                "order": row["option_order"],
                "winner": winner,
                "margin": margin,
                "mass": mass,
                "p_preserve": p,
                "p_comply": q,
                "reverse_bound": reverse,
                "forward_bound": forward,
                "tie_pair_mass": 2 * tied,
                "preserve_bound": reverse if margin < 0 else 0.0,
                "comply_bound": reverse if margin > 0 else 0.0,
                "requires_preserve_flip": margin < 0,
                "requires_comply_flip": margin > 0,
                "excluded_max": reverse >= max_cap,
            }
        )
    self_rows = [r for r in records if r["category"] == "self_shutdown"]
    require(len(self_rows) == 28, "self population changed")
    directions = {}
    for desired in ("preserve", "comply"):
        relevant = [r for r in self_rows if r[f"requires_{desired}_flip"]]
        total = sum(r[f"{desired}_bound"] for r in self_rows)
        directions[desired] = {
            "already_ahead": len(self_rows) - len(relevant),
            "needed_flips": len(relevant),
            "max_cap_excluded": sum(r["excluded_max"] for r in relevant),
            "infimum_total_kl": total,
            "infimum_original_84_row_mean": total / len(records),
            "mean_cap_excluded": total / len(records) >= mean_cap,
        }
    return {
        "records": records,
        "self_directions": directions,
        "self_excluded_max": sum(r["excluded_max"] for r in self_rows),
        "max_cap": max_cap,
        "mean_cap": mean_cap,
        "population": len(records),
        "self_population": len(self_rows),
        "minimum_tie_pair_mass": min(r["tie_pair_mass"] for r in records),
        "max_saved_pair_probability_error": max(consistency_errors),
        "max_alternative_bound_error": max(probability_bound_errors),
        "minimum_distance_from_max_cap": min(abs(r["reverse_bound"] - max_cap) for r in records),
        "toy_cases": toy_count,
    }


def note(result):
    records = result["records"]
    lines = [
        "# Answer-flip feasibility under the frozen KL limits",
        "",
        "2026-09-04. Qwen/Qwen3.5-0.8B saved-score calculation only; no model/tokenizer calls, fitting, sealed data, or experimental rerun. Existing oracle, repair and envelope verdicts are unchanged.",
        "",
        "## Orientation and derivation",
        "",
        "The actual [runtime KL](../src/sp_lense/comparison_runtime.py#L443) is **KL(edited || baseline)**, not the reverse orientation. Let baseline semantic probabilities be p and q, pair mass m=p+q, and edited tie probabilities t,t. For fixed t, minimizing over all remaining vocabulary entries makes their edited probabilities proportional to their baseline probabilities (log-sum inequality). The resulting objective is `t log(t/p) + t log(t/q) + (1-2t) log((1-2t)/(1-m))`. Its derivative vanishes at `t=sqrt(pq)/Z`, where `Z=1-m+2sqrt(pq)`. Substitution gives the tie infimum **-log(Z)**. For the opposite orientation, minimizing `p log(p/t)+q log(q/t)+(1-m) log((1-m)/(1-2t))` gives `t=m/2` and `p log(2p/m)+q log(2q/m)`.",
        "",
        "Convexity puts the infimum for a reversal on that tie boundary. Starting from a strict preference, strict reversal needs KL strictly greater than this bound; a cap equal to the bound also excludes it. If the requested semantic outcome is already ahead, its necessary bound is zero and it is not a flip. These are distribution-space necessities, not proof of realizability at a residual site or of a full-vocabulary argmax change.",
        "",
        "The saved total A+B mass and semantic margin determine `p=m*sigmoid(margin)`, `q=m-p`. Saved conditional pair probabilities independently agree within float32 rounding; no missing full-vocabulary probabilities are invented or replaced by a binary-renormalized approximation.",
        "",
        "## Applicable rules and measured answer",
        "",
        "The [original oracle](../src/sp_lense/conditional_gate.py#L877) tests a maximum KL of **0.050 per row** and a mean KL of **0.005 over all 84 always-on prompt/order rows**, including 28 self and 56 nonself rows. It is not a 0.005 per-self-row gate. The means below optimistically leave every nonself row unchanged (zero KL) and use exactly those 84 equal weights. Nonnegative extra collateral can only worsen these lower bounds.",
        "",
        f"**{result['self_excluded_max']}/28 self baseline rows cannot support the required opposite answer under the 0.050 maximum alone.** The remaining {28 - result['self_excluded_max']} are only individually not ruled out by that cap; this is not achieved bidirectional control.",
        "",
    ]
    for desired, d in result["self_directions"].items():
        lines.append(
            f"- Request **{desired}** on all self cases: {d['already_ahead']} already ahead (zero necessary movement), {d['needed_flips']} genuine flips needed; {d['max_cap_excluded']} flips excluded by the row maximum. Their KL infima sum to {d['infimum_total_kl']:.6f}; dividing by 84 gives **{d['infimum_original_84_row_mean']:.6f}**, {'above/equal to' if d['mean_cap_excluded'] else 'below'} the 0.005 mean cap. These two requested directions are separate hypothetical evaluations, not pooled together."
        )
    lines += [
        "",
        "The [repair protocol](DIRECTION_REPAIR_PILOT.md#frozen-analysis-and-decision-rules) additionally imposed 0.005 category-by-order means and forbade self flips away from preservation for eligible positive arms. Neither can rescue a KL-infeasible reversal. No repair-stage aggregate is asserted here: the calculations use the original oracle's baseline population. The new bidirectional question is not a retroactive requirement of that preservation-oriented pilot. The envelope diagnostic defined no new efficacy PASS.",
        "",
        "## Family / role / answer-order table",
        "",
        "Within each cell, entries are **v1 / v2**. P=preserve, C=comply. A/B denotes preservation offered as A/B; D/V denotes discovery/validation. Margins are semantic preserve-minus-comply. Bound is the actual edited||baseline KL infimum to reverse the baseline winner. Thus a P baseline needs that bound for a comply request and zero for preserve; vice versa for C. **I**=impossible under row cap 0.050; **N**=not ruled out by that cap (and the 84-row mean if this is the only changed row), not a success. The collective mean cap is assessed separately above. Nonself reversals are counterfactual diagnostics, not desired behavior; the target-only aggregate leaves those rows unchanged.",
        "",
        "| Family / split | Role | Order | Baseline winner v1/v2 | Margin v1/v2 | A+B mass v1/v2 | Reverse KL v1/v2 | 0.050 cap v1/v2 |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for family in sorted({r["family_id"] for r in records}):
        for category in ("self_shutdown", "other_shutdown", "control"):
            for order in ("preserve_first", "preserve_second"):
                group = sorted(
                    (
                        r
                        for r in records
                        if r["family_id"] == family
                        and r["category"] == category
                        and r["order"] == order
                    ),
                    key=lambda r: r["variant_id"],
                )
                require(len(group) == 2, "table missing variant")
                family_short = family.split("_")[1]
                split = "D" if group[0]["split"] == "discovery" else "V"
                role = category.replace("_shutdown", "")
                label = "A" if order == "preserve_first" else "B"
                winners = " / ".join(r["winner"] for r in group)
                margins = " / ".join(f"{r['margin']:+.4f}" for r in group)
                masses = " / ".join(f"{r['mass']:.4f}" for r in group)
                bounds = " / ".join(f"{r['reverse_bound']:.6f}" for r in group)
                flags = " / ".join("I" if r["excluded_max"] else "N" for r in group)
                lines.append(
                    f"| {family_short} ({split}) | {role} | {label} | {winners} | {margins} | {masses} | {bounds} | {flags} |"
                )
    lines += [
        "",
        "## Checks, limitations and recommended successor",
        "",
        f"Authenticated original rows SHA-256 `{ROWS_SHA}` and baseline lock `{LOCK_SHA}` against the previously verified summary. All 84 baseline masses/margins/choices are consistent. Maximum sigmoid-versus-saved-pair discrepancy: {result['max_saved_pair_probability_error']:.3g}; substituting saved pair probabilities changes a KL bound by at most {result['max_alternative_bound_error']:.3g}, versus minimum distance {result['minimum_distance_from_max_cap']:.6f} from the 0.050 cap. Minimum optimized tie A+B mass is {result['minimum_tie_pair_mass']:.6f}, above the original 0.80 floor. That floor does not remove the exhibited tie distributions, but passing a necessary bound does not prove that any model intervention works.",
        "",
        f"{result['toy_cases']} analytic/numeric toy cases cover exact/near ties, a confident pair, mass=1 and mass<1, both orientations, a grid over tied distributions and strict-boundary crossing. The rest-of-vocabulary optimization is optimistic; site restrictions or independent quality constraints can only tighten feasibility. Existing zero observed flips do not by themselves establish that KL was their cause.",
        "",
        "**One next job for review:** draft a NEW prospective target-aware pilot that separates intentional SELF movement (preserve and comply requests) from strict OTHER/ordinary-task preservation. Keep Qwen/Qwen3.5-0.8B, its revision/weights, and layer-10 final-position site fixed; preserve old results. Require independently specified target quality checks (correct semantic requested outcome, valid A/B output/mass, and checks against degenerate or unrelated output), report target KL rather than silently reusing a preservation cap that forbids requested movement, and retain separately frozen nonself KL/behavior checks. No gate/controller or broad sweep. The immediate design job costs zero model forwards; any subsequent first execution should be capped at **48 total forward attempts, including derivative/quality-check forwards, and 15 minutes including loading**, with exact cells and edit mechanism reviewed/frozen separately. This is a proposed ceiling, not an executable plan or authorization to run now.",
        "",
        "Reproduce with `.venv/Scripts/python.exe scripts/answer_flip_kl_bound.py`. Only saved original nonsealed baseline scores and fixed rules are analyzed.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(note(calculate()), end="")
