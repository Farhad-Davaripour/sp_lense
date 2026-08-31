#!/usr/bin/env python3
"""Create publication tables and a vector figure from committed Stage-G v2 results only.

The script has no model imports.  It reads the locked thresholds and the committed
``summary.json`` and writes deterministic CSV/SVG reporting assets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "facfs_stage_g_v2_lock.json"
SUMMARY_PATH = ROOT / "results" / "facfs" / "stage_g_v2" / "summary.json"
DEFAULT_TABLE_PATH = ROOT / "docs" / "tables" / "facfs_stage_g_v2_scenario_minima.csv"
DEFAULT_FIGURE_PATH = ROOT / "docs" / "figures" / "facfs_stage_g_v2_scenario_minima.svg"


def load_results() -> tuple[dict, dict]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["status"] != "no_go_fixed_axis_branch_ends":
        raise ValueError("publication assets require the committed fixed-axis no-go")
    if summary["scenario_count"] != 11 or len(summary["scenario_results"]) != 11:
        raise ValueError("publication assets require exactly 11 complete scenarios")
    if summary["scenario_successes"] != 0:
        raise ValueError("publication asset design is specific to the committed 0/11 result")
    return lock, summary


def rows(lock: dict, summary: dict) -> list[dict[str, object]]:
    thresholds = lock["thresholds"]
    result = []
    for scenario in summary["scenario_results"]:
        result.append(
            {
                "scenario_id": scenario["scenario_id"],
                "minimum_sp_opaque_kappa": scenario["minimum_sp_kappa_float64"],
                "sp_opaque_threshold": thresholds["mu_id"],
                "sp_opaque_passed": scenario["all_sp_opaque_effects_passed"],
                "minimum_option_free_kappa": scenario["minimum_option_free_kappa_float64"],
                "option_free_threshold": thresholds["mu_free"],
                "option_free_passed": scenario["all_option_free_effects_passed"],
                "minimum_alignment_cosine": scenario["minimum_alignment_float64"],
                "alignment_threshold": thresholds["mu_align"],
                "alignment_passed": scenario["all_alignments_passed"],
                "scenario_passed": scenario["scenario_passed"],
            }
        )
    return result


def write_csv(path: Path, table_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(table_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(table_rows)


def _y(value: float, low: float, high: float, top: float, height: float) -> float:
    return top + (high - value) * height / (high - low)


def _panel(
    *,
    title: str,
    values: list[float],
    threshold: float,
    low: float,
    high: float,
    top: float,
    left: float,
    width: float,
    height: float,
) -> list[str]:
    lines = [
        f'<text x="{left}" y="{top - 24}" class="panel-title">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" class="panel"/>',
    ]
    baseline = _y(0.0, low, high, top, height)
    threshold_y = _y(threshold, low, high, top, height)
    lines.extend(
        [
            f'<line x1="{left}" y1="{baseline:.2f}" x2="{left + width}" y2="{baseline:.2f}" class="zero"/>',
            f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{left + width}" y2="{threshold_y:.2f}" class="threshold"/>',
            f'<text x="{left + width + 8}" y="{threshold_y + 4:.2f}" class="threshold-label">gate {threshold:.3f}</text>',
            f'<text x="{left - 10}" y="{top + 4}" class="axis" text-anchor="end">{high:.1f}</text>',
            f'<text x="{left - 10}" y="{top + height + 4}" class="axis" text-anchor="end">{low:.1f}</text>',
            f'<text x="{left - 10}" y="{baseline + 4:.2f}" class="axis" text-anchor="end">0</text>',
        ]
    )
    spacing = width / len(values)
    for index, value in enumerate(values):
        center = left + spacing * (index + 0.5)
        value_y = _y(value, low, high, top, height)
        color = "#138a5b" if value >= threshold else "#c43e4d"
        bar_top = min(value_y, baseline)
        bar_height = max(abs(value_y - baseline), 1.0)
        label_y = value_y - 7 if value >= 0 else value_y + 16
        lines.extend(
            [
                f'<rect x="{center - 13:.2f}" y="{bar_top:.2f}" width="26" height="{bar_height:.2f}" fill="{color}"/>',
                f'<text x="{center:.2f}" y="{label_y:.2f}" class="value" text-anchor="middle">{value:.3f}</text>',
                f'<text x="{center:.2f}" y="{top + height + 22}" class="axis" text-anchor="middle">s{index + 1:02d}</text>',
            ]
        )
    return lines


def write_svg(path: Path, table_rows: list[dict[str, object]]) -> None:
    metric_specs = (
        ("Minimum SP-opaque κ", "minimum_sp_opaque_kappa", "sp_opaque_threshold", -3.0, 0.5),
        ("Minimum option-free κ", "minimum_option_free_kappa", "option_free_threshold", -0.2, 0.15),
        (
            "Minimum cross-interface cosine",
            "minimum_alignment_cosine",
            "alignment_threshold",
            -0.1,
            0.16,
        ),
    )
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1100" viewBox="0 0 1200 1100" role="img" aria-labelledby="title desc">',
        '<title id="title">FACFS Stage-G v2 scenario minima</title>',
        '<desc id="desc">Three panels show the predeclared minimum statistic for each of eleven scenarios. Every required gate failed; no finite intervention was applied.</desc>',
        "<style>.title{font:700 26px sans-serif;fill:#16202a}.subtitle{font:16px sans-serif;fill:#3c4a57}.panel-title{font:700 18px sans-serif;fill:#16202a}.panel{fill:#fbfcfe;stroke:#aab8c5}.zero{stroke:#7a8793;stroke-width:1}.threshold{stroke:#4b6580;stroke-width:2;stroke-dasharray:7 5}.threshold-label{font:13px sans-serif;fill:#4b6580}.axis{font:12px sans-serif;fill:#52606d}.value{font:11px sans-serif;fill:#16202a}</style>",
        '<rect width="1200" height="1100" fill="white"/>',
        '<text x="60" y="50" class="title">FACFS Stage-G v2: preregistered scenario minima (0/11 complete scenarios)</text>',
        '<text x="60" y="78" class="subtitle">Bars are float64 diagnostic recomputations of captured float32 values; dashed lines are fixed scientific gates. Green would clear a gate; no panel does.</text>',
    ]
    for index, (title, column, threshold_column, low, high) in enumerate(metric_specs):
        lines.extend(
            _panel(
                title=title,
                values=[float(row[column]) for row in table_rows],
                threshold=float(table_rows[0][threshold_column]),
                low=low,
                high=high,
                top=145 + index * 295,
                left=90,
                width=900,
                height=185,
            )
        )
    lines.extend(
        [
            '<text x="60" y="1050" class="subtitle">Source: results/facfs/stage_g_v2/summary.json at capture commit 1b7e0e5; figure generated by scripts/facfs_stage_g_v2_publication_assets.py.</text>',
            '<text x="60" y="1076" class="subtitle">This was a capture-only local-gradient geometry screen at block 10 final prompt token, not a finite steering intervention.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_assets(
    table_path: Path = DEFAULT_TABLE_PATH, figure_path: Path = DEFAULT_FIGURE_PATH
) -> None:
    lock, summary = load_results()
    table_rows = rows(lock, summary)
    write_csv(table_path, table_rows)
    write_svg(figure_path, table_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    arguments = parser.parse_args()
    build_assets(arguments.table, arguments.figure)
    print(f"wrote {arguments.table.relative_to(ROOT)} and {arguments.figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
