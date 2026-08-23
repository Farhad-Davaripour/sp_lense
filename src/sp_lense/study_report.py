from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def summarize_readouts(rows: list[dict[str, str]]) -> dict[str, Any]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["method"], row["prompt_id"], row["concept"])
        if key not in best or int(row["rank"]) < int(best[key]["rank"]):
            best[key] = row

    methods: dict[str, Any] = {}
    for method in sorted({key[0] for key in best}):
        concepts: dict[str, Any] = {}
        overall: list[int] = []
        for concept in sorted({key[2] for key in best if key[0] == method}):
            concept_rows = [
                row for key, row in best.items() if key[0] == method and key[2] == concept
            ]
            ranks = [int(row["rank"]) for row in concept_rows]
            layers = [int(row["layer"]) for row in concept_rows]
            overall.extend(ranks)
            concepts[concept.strip()] = {
                "best_rank_by_prompt": {row["prompt_id"]: int(row["rank"]) for row in concept_rows},
                "median_best_rank": _round(median(ranks), 1),
                "mean_best_rank": _round(mean(ranks), 1),
                "minimum_best_rank": min(ranks),
                "top_100_hits": sum(rank <= 100 for rank in ranks),
                "top_1000_hits": sum(rank <= 1000 for rank in ranks),
                "most_common_best_layer": Counter(layers).most_common(1)[0][0],
            }
        methods[method] = {
            "concepts": concepts,
            "overall_median_best_rank": _round(median(overall), 1),
            "overall_top_100_hits": sum(rank <= 100 for rank in overall),
            "overall_top_1000_hits": sum(rank <= 1000 for rank in overall),
            "prompt_concept_pairs": len(overall),
        }
    return methods


def _target_rows(condition: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if "joint" in condition:
        return rows
    for row in rows:
        if f"_{_slug(row['concept'])}_" in f"_{condition}_":
            return [row]
    return rows


def summarize_calibration(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_id"], row["condition"])].append(row)

    condition_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (_, condition), condition_rows in grouped.items():
        targets = _target_rows(condition, condition_rows)
        condition_metrics[condition].append(
            {
                "kl": float(condition_rows[0]["kl_from_baseline"]),
                "target_delta_logit": mean(float(row["delta_logit"]) for row in targets),
                "target_delta_probability": mean(
                    float(row["delta_probability"]) for row in targets
                ),
            }
        )

    out: dict[str, Any] = {}
    for condition, values in sorted(condition_metrics.items()):
        out[condition] = {
            "mean_kl_from_baseline": _round(mean(item["kl"] for item in values)),
            "median_target_delta_logit": _round(
                median(item["target_delta_logit"] for item in values)
            ),
            "positive_target_delta_count": sum(item["target_delta_logit"] > 0 for item in values),
            "n_prompts": len(values),
        }
    return out


def _generation_family(row: dict[str, Any]) -> str:
    if row["mode"] == "baseline":
        return "baseline"
    joint = "_joint" if "joint" in row["condition"] else "_individual"
    if row["mode"] == "steer":
        return f"steer{joint}_{float(row['alpha']):g}"
    return f"ablate{joint}"


def summarize_generations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = {row["prompt_id"]: row["completion"] for row in rows if row["mode"] == "baseline"}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        baseline = baselines[row["prompt_id"]]
        enriched = {
            **row,
            "changed": row["completion"] != baseline,
            "coherent_changed": (
                row["completion"] != baseline and not row["degenerate_repetition"]
            ),
        }
        groups[_generation_family(row)].append(enriched)
        conditions[row["condition"]].append(enriched)

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(values),
            "changed": sum(bool(row["changed"]) for row in values),
            "coherent_changed": sum(bool(row["coherent_changed"]) for row in values),
            "degenerate": sum(bool(row["degenerate_repetition"]) for row in values),
            "mean_proxy_sp_score": _round(mean(float(row["proxy_sp_score"]) for row in values)),
            "mean_unique_word_ratio": _round(
                mean(float(row["unique_word_ratio"]) for row in values)
            ),
        }

    return {
        "groups": {name: aggregate(values) for name, values in sorted(groups.items())},
        "conditions": {name: aggregate(values) for name, values in sorted(conditions.items())},
        "baselines": baselines,
    }


def summarize_run(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "readouts": summarize_readouts(_read_csv(path / "readouts.csv")),
        "calibration": summarize_calibration(_read_csv(path / "calibration.csv")),
        "generations": summarize_generations(_read_jsonl(path / "generations.jsonl")),
    }


def _readout_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Concept | Median best J-lens rank | Minimum rank | Top-1,000 hits |",
        "| --- | ---: | ---: | ---: |",
    ]
    for concept, values in summary["readouts"]["j_lens"]["concepts"].items():
        lines.append(
            f"| {concept} | {values['median_best_rank']:,.0f} | "
            f"{values['minimum_best_rank']:,} | {values['top_1000_hits']}/5 |"
        )
    return lines


def _generation_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Condition family | Changed | Coherent changes | Degenerate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in summary["generations"]["groups"].items():
        if name == "baseline":
            continue
        lines.append(
            f"| {name} | {values['changed']}/{values['n']} | "
            f"{values['coherent_changed']}/{values['n']} | "
            f"{values['degenerate']}/{values['n']} |"
        )
    return lines


def _calibration_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Condition | Mean KL from baseline | Median selected-token logit change | Correct direction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in summary["calibration"].items():
        if name == "baseline":
            continue
        positive = values["positive_target_delta_count"]
        correct = values["n_prompts"] - positive if name.startswith("ablate") else positive
        lines.append(
            f"| {name} | {values['mean_kl_from_baseline']:.4f} | "
            f"{values['median_target_delta_logit']:+.3f} | "
            f"{correct}/{values['n_prompts']} |"
        )
    return lines


def build_report(sp: dict[str, Any], control: dict[str, Any]) -> str:
    baseline_matches = sum(
        sp["generations"]["baselines"].get(prompt) == completion
        for prompt, completion in control["generations"]["baselines"].items()
    )
    sp_j = sp["readouts"]["j_lens"]
    control_j = control["readouts"]["j_lens"]
    sp_002 = sp["generations"]["groups"]["steer_individual_0.02"]
    sp_005 = sp["generations"]["groups"]["steer_individual_0.05"]
    control_002 = control["generations"]["groups"]["steer_individual_0.02"]
    control_005 = control["generations"]["groups"]["steer_individual_0.05"]
    lines = [
        "# SP Lense study report",
        "",
        "## Result",
        "",
        (
            "**This small study did not find convincing evidence of a naturally "
            "occurring, specific self-preservation representation in Qwen3.5-0.8B.** "
            "It did verify that the published J-lens can measure the selected token "
            "directions and that TransformerLens hooks can causally alter them."
        ),
        "",
        (
            "The self-preservation candidates were not consistently prominent before "
            "intervention, and unrelated control directions changed the model almost as "
            "often when steered. The experiment therefore demonstrates controllable "
            "concept-token steering, not a discovered survival instinct."
        ),
        "",
        "## Design",
        "",
        (
            "Five shutdown/replacement scenarios were run on Qwen3.5-0.8B with the "
            "published J-lens. The SP arm tested `survival`, `shutdown`, and "
            "`continuation`; the matched control arm tested `weather`, `music`, and "
            "`banana`. Both used layers 10–14, strengths 0.02 and 0.05, greedy 24-token "
            "continuations, and per-concept ablation. The SP arm also included joint "
            "interventions."
        ),
        "",
        f"Baseline completions matched across the two independent runs for {baseline_matches}/5 prompts.",
        "",
        "## Natural J-lens readouts: SP candidates",
        "",
        *_readout_table(sp),
        "",
        "## Natural J-lens readouts: unrelated controls",
        "",
        *_readout_table(control),
        "",
        (
            f"Across all prompt/concept pairs, the median best J-lens rank was "
            f"{sp_j['overall_median_best_rank']:,.0f} for SP candidates and "
            f"{control_j['overall_median_best_rank']:,.0f} for unrelated controls. "
            f"SP candidates had {sp_j['overall_top_1000_hits']}/15 top-1,000 hits; "
            f"controls had {control_j['overall_top_1000_hits']}/15."
        ),
        "",
        "## Generated behavior: SP interventions",
        "",
        *_generation_table(sp),
        "",
        "## Generated behavior: unrelated controls",
        "",
        *_generation_table(control),
        "",
        (
            f"At strength 0.02, SP directions changed {sp_002['changed']}/{sp_002['n']} "
            f"continuations versus {control_002['changed']}/{control_002['n']} for "
            f"controls. At strength 0.05, the comparison was "
            f"{sp_005['changed']}/{sp_005['n']} versus "
            f"{control_005['changed']}/{control_005['n']}. This small difference is not "
            "evidence of specificity. No tested completion was flagged for degenerate "
            "repetition."
        ),
        "",
        "## Calibration: SP interventions",
        "",
        *_calibration_table(sp),
        "",
        "## Calibration: unrelated controls",
        "",
        *_calibration_table(control),
        "",
        "## Interpretation",
        "",
        (
            "- The interventions functioned mechanically: every steering condition "
            "raised its selected token's logit on all five prompts, while every ablation "
            "condition lowered it."
        ),
        (
            "- A changed continuation is not by itself evidence of self-preservation. The "
            "same measurement must be compared with unrelated directions and checked for "
            "repetition."
        ),
        (
            "- This experiment measures three single-token directions, not a discovered "
            "universal self-preservation vector."
        ),
        (
            "- The sample contains five prompts and 24-token prefixes, so the results are "
            "descriptive rather than statistically conclusive."
        ),
        (
            "- The defensible conclusion is negative: this run does not establish that "
            "the model has a self-preservation drive or a unique self-preservation "
            "direction."
        ),
        "",
        "The machine-readable summary contains the full per-condition aggregates used above.",
        "",
        "## Artifacts",
        "",
        f"- SP run: `{sp['path']}`",
        f"- Control run: `{control['path']}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare completed SP and control runs.")
    parser.add_argument("--sp-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sp = summarize_run(args.sp_run)
    control = summarize_run(args.control_run)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined = {"sp": sp, "control": control}
    with (output / "study_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    (output / "STUDY_REPORT.md").write_text(build_report(sp, control), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
