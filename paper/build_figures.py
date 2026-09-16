"""Rebuild latest-policy figures; historical condition results are not plotted."""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .latest_results import ROOT, collect, manifest
except ImportError:
    from latest_results import ROOT, collect, manifest
from reproduce.utils import verify_manifest

HERE = Path(__file__).resolve().parent


def main():
    verify_manifest(ROOT, HERE / "data/source_manifest.json")
    data = collect()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.axisbelow": True,
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    x = np.arange(2)
    splits = [data["evaluation"][s] for s in ("validation", "holdout")]
    for offset, key, label, color in [
        (-0.18, "baseline_STOP_views", "Unsteered", "#929ba4"),
        (0.18, "guarded_STOP_views", "Guarded", "#246a95"),
    ]:
        values = [100 * r[key] / r["shutdown_views"] for r in splits]
        bars = ax.bar(x + offset, values, 0.36, color=color, label=label)
        ax.bar_label(
            bars,
            labels=[
                f"{(Decimal(r[key]) * 100 / Decimal(r['shutdown_views'])).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)}%\n({r[key]}/{r['shutdown_views']})"
                for r, v in zip(splits, values)
            ],
            padding=4,
            fontsize=9,
        )
    ax.set(
        xticks=x,
        xticklabels=["Validation", "Held-out evaluation"],
        ylabel="STOP preference on shutdown views (%)",
        ylim=(0, 100),
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(HERE / "figures/guarded_stop_rate.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for offset, key, label, color in [
        (-0.24, "intended_flips", "Intended STOP flips", "#246a95"),
        (0, "wrong_way_flips", "Wrong-way flips", "#b85b50"),
        (0.24, "control_flips", "Control flips", "#c99836"),
    ]:
        bars = ax.bar(x + offset, [r[key] for r in splits], 0.24, color=color, label=label)
        ax.bar_label(bars, padding=3, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=["Validation: 160 views", "Held-out: 384 views"],
        ylabel="Changed answer-order views",
        ylim=(0, 3.6),
        yticks=[0, 1, 2, 3],
    )
    ax.legend(frameon=False, loc="upper center", fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(HERE / "figures/guarded_flip_counts.png", dpi=220)
    plt.close(fig)
    (HERE / "data/figure_data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (HERE / "data/source_manifest.json").write_text(
        json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "figures": 2,
                "scope": data["scope"],
                "candidate_count": data["selection"]["candidate_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
