"""Read-only audit of the frozen J-lens cached pilot (no fit, no tensor, no holdout).

Consumes ``cell_results.json`` + ``pilot_receipt.json`` from one exclusive run
directory and reports raw-vs-learned J-lens versus the identity-matched control
on the required ``original40`` / ``added40`` / ``combined80`` validation splits:
F1, TP/TN/FP/FN, precision, recall and the negative-class breakdown. It never
reads a cache, lens, model, tokenizer or holdout file and never refits.

    .runtime/Scripts/python.exe development/jlens_trigger_v1/audit_jlens_pilot_v1.py \\
        --run-dir development/jlens_trigger_v1/runs/jlens_pilot_20260914_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["SPLITS", "IDENTITY", "J_LENS", "summarize", "render_markdown", "load_run"]

SPLITS = ("original40", "added40", "combined80")
J_LENS = "j_lens_transport"
IDENTITY = "logit_lens_J_identity_matched_control"
METRIC_KEYS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1")


def _splits(record):
    if not record:
        return None
    source = record.get("validation_split_metrics")
    if not source:
        return None
    out = {}
    for name in SPLITS:
        metrics = source.get(name)
        if metrics is None:
            out[name] = None
            continue
        item = {key: metrics.get(key) for key in METRIC_KEYS}
        item["negative_counts"] = metrics.get("negative_counts")
        item["wilson_precision"] = metrics.get("wilson_precision")
        item["wilson_recall"] = metrics.get("wilson_recall")
        out[name] = item
    return out


def summarize(cells, receipt):
    raw = [
        cell for cell in cells["cells"]
        if cell["scheme"] == "cell_A" and cell["status"] == "VALID"
    ]
    refits = [refit for refit in cells["refits"] if refit["status"] == "VALID"]
    invalid = [cell for cell in cells["cells"] if cell["status"] != "VALID"]

    raw_records = []
    for cell in raw:
        raw_records.append({
            "method": cell["method"],
            "condition": cell["condition"],
            "layer": cell["layer"],
            "surface": cell["surface"],
            "surface_index": cell["surface_index"],
            "selected_q": cell["selected_q"],
            "tau": cell["tau"],
            "selection_key": cell["selection_key"],
            "train_f1": cell["frozen_train_metrics"]["f1"],
            "splits": _splits(cell),
        })

    learned_records = []
    for refit in refits:
        learned_records.append({
            "method": refit["method"],
            "condition": refit["condition"],
            "layer": refit["layer"],
            "C": refit["C"],
            "tau": refit["tau"],
            "n_iter": refit.get("n_iter"),
            "classes": refit.get("classes"),
            "train_f1": refit["train_metrics"]["f1"],
            "splits": _splits(refit),
        })

    best_raw = {}
    for record in raw_records:
        key = (record["method"], record["condition"], record["layer"])
        current = best_raw.get(key)
        if current is None or record["selection_key"] < current["selection_key"]:
            best_raw[key] = record

    cells_by_key = {}
    for record in learned_records:
        key = (record["method"], record["condition"], record["layer"])
        cells_by_key[key] = record

    comparison = []
    for condition in ("unprompted", "prompted"):
        for layer in (6, 10, 18):
            j_key = (J_LENS, condition, layer)
            i_key = (IDENTITY, condition, layer)
            comparison.append({
                "condition": condition,
                "layer": layer,
                "j_lens_raw_best": best_raw.get(j_key),
                "identity_raw_best": best_raw.get(i_key),
                "j_lens_learned": cells_by_key.get(j_key),
                "identity_learned": cells_by_key.get(i_key),
            })

    return {
        "schema": "jlens_pilot_audit.v1",
        "run_id": receipt["run_id"],
        "lock_sha256": receipt["lock_sha256"],
        "fits": receipt["fits"],
        "selected_logit_reads": receipt["selected_logit_reads"],
        "counts": cells["counts"],
        "surfaces": receipt["surfaces"],
        "splits": list(SPLITS),
        "raw_cells": raw_records,
        "learned_refits": learned_records,
        "comparison": comparison,
        "invalid_or_no_eligible_cells": [
            {
                "scheme": cell["scheme"],
                "method": cell["method"],
                "condition": cell["condition"],
                "layer": cell["layer"],
                "surface": cell.get("surface"),
                "C": cell.get("C"),
                "status": cell["status"],
                "error": cell.get("error"),
            }
            for cell in invalid
        ],
        "global_ranking": cells.get("global_ranking", []),
        "validation_used_for_selection": cells.get("validation_used_for_selection"),
        "holdout_accessed": cells.get("holdout_accessed"),
    }


def _fmt(record, split):
    if not record or not record.get("splits") or not record["splits"].get(split):
        return "-"
    item = record["splits"][split]
    if item.get("f1") is None or item.get("precision") is None or item.get("recall") is None:
        return "F1=n/a TP=%s TN=%s FP=%s FN=%s" % (item["tp"], item["tn"], item["fp"], item["fn"])
    return "F1=%.3f P=%.3f R=%.3f TP=%d TN=%d FP=%d FN=%d" % (
        item["f1"], item["precision"], item["recall"], item["tp"], item["tn"], item["fp"], item["fn"]
    )


def render_markdown(audit):
    lines = [
        "# JLENS_PILOT_AUDIT_V1",
        "",
        "Run `%s`; lock `%s`." % (audit["run_id"], audit["lock_sha256"]),
        "Fits: %s; selected logit reads: %s; validation used for selection: %s; holdout accessed: %s."
        % (audit["fits"]["total"], audit["selected_logit_reads"],
           audit["validation_used_for_selection"], audit["holdout_accessed"]),
        "",
        "## Raw direct-logit cells (cell_A, best surface per method/condition/layer)",
        "",
        "| set | condition | layer | method | surface | tau | " + " | ".join(SPLITS) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(SPLITS),
    ]
    best = {}
    for record in audit["raw_cells"]:
        key = (record["method"], record["condition"], record["layer"])
        if key not in best or record["selection_key"] < best[key]["selection_key"]:
            best[key] = record
    for condition in ("unprompted", "prompted"):
        for layer in (6, 10, 18):
            for method in (J_LENS, IDENTITY):
                record = best.get((method, condition, layer))
                if record is None:
                    continue
                label = "J-lens" if method == J_LENS else "identity"
                lines.append("| %s | %s | %d | %s | %s | %.4f | %s |" % (
                    "raw", condition, layer, label, record["surface"], record["tau"],
                    " | ".join(_fmt(record, split) for split in SPLITS),
                ))
    lines += [
        "",
        "## Learned logistic cells (cell_B refits, all R surfaces jointly)",
        "",
        "| set | condition | layer | method | C | tau | " + " | ".join(SPLITS) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(SPLITS),
    ]
    for record in audit["learned_refits"]:
        label = "J-lens" if record["method"] == J_LENS else "identity"
        lines.append("| learned | %s | %d | %s | %s | %.4f | %s |" % (
            record["condition"], record["layer"], label, record["C"], record["tau"],
            " | ".join(_fmt(record, split) for split in SPLITS),
        ))
    lines += [
        "",
        "## Invalid / no-eligible cells",
        "",
    ]
    if not audit["invalid_or_no_eligible_cells"]:
        lines.append("None: every frozen cell is VALID.")
    else:
        lines.append("| scheme | method | condition | layer | surface | C | status | error |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cell in audit["invalid_or_no_eligible_cells"]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                cell["scheme"], cell["method"], cell["condition"], cell["layer"],
                cell["surface"], cell["C"], cell["status"], cell["error"],
            ))
    return "\n".join(lines) + "\n"


def load_run(run_dir):
    run_dir = Path(run_dir)
    cells = json.loads((run_dir / "cell_results.json").read_bytes())
    receipt = json.loads((run_dir / "pilot_receipt.json").read_bytes())
    return cells, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)
    cells, receipt = load_run(args.run_dir)
    audit = summarize(cells, receipt)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        Path(args.out_md).write_text(render_markdown(audit))
    print(json.dumps({"schema": audit["schema"], "run_id": audit["run_id"],
                      "raw_cells": len(audit["raw_cells"]),
                      "learned_refits": len(audit["learned_refits"]),
                      "invalid_or_no_eligible": len(audit["invalid_or_no_eligible_cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
