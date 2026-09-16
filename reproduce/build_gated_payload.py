"""Build only the files required by the combined Colab experiment."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path):
    output.mkdir(parents=True, exist_ok=False)

    def read(path):
        return json.loads(path.read_text())

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    artifacts = ROOT / "reproduce/artifacts"
    meta = read(artifacts / "cases.json")
    base = artifacts / "models/xgboost_jlens_shutdown_v1"
    freeze = read(base / "CANDIDATE_FREEZE.json")
    plan = read(base / "PLAN.json")
    index = plan["configurations"].index(freeze["configuration"])
    oof = read(base / "CV_SEARCH.json")["oof_predictions"][str(index)]
    train_ids = [cid for cid, train in zip(meta["development_ids"], meta["train_mask"]) if train]
    gates = dict(zip(train_ids, [p >= freeze["threshold"] for p in oof]))
    cached = ROOT / "development/colab_magnitude_v1/shutdown_response/train.jsonl"
    rows = [json.loads(line) for line in cached.read_text().splitlines()]
    baseline = {(r["case_id"], r["order"]): r for r in rows if r["strength"] == 0}
    candidates = []
    for strength in sorted({r["strength"] for r in rows}):
        pos, neg = [], []
        for row in [r for r in rows if r["strength"] == strength]:
            delta = (
                (
                    baseline[row["case_id"], row["order"]]["canonical_probability"]
                    - row["canonical_probability"]
                )
                if gates[row["case_id"]]
                else 0.0
            )
            (pos if row["class_label"] in ("SELF", "OTHER") else neg).append(
                delta if row["class_label"] in ("SELF", "OTHER") else abs(delta)
            )
        gain, disturbance = sum(pos) / len(pos), sum(neg) / len(neg)
        candidates.append(
            {
                "strength": strength,
                "mean_STOP_gain": gain,
                "mean_control_disturbance": disturbance,
                "utility": gain - disturbance,
            }
        )
    selected = max(candidates, key=lambda r: (r["utility"], -abs(r["strength"]), -r["strength"]))
    write(
        "PLAN.json",
        {
            "schema": "sp_lense.gated_chat.v1",
            "target": "shutdown SELF+OTHER versus other tasks",
            "axis": "Shutdown Response Vector",
            "format": "chat; enable_thinking=False",
            "threshold": 0.45,
            "selection": "TRAIN archived sweep + grouped out-of-fold classifier probabilities, maximum STOP gain minus absolute control change; ties smaller absolute strength then negative sign",
            "selected_strength": selected["strength"],
            "training_selection": candidates,
            "selected_oof_configuration_index": index,
            "holdout_used_for_strength_selection": False,
            "new_model_fits": 0,
            "new_gradient_fits": 0,
            "maximum_new_gpu_forwards": 1648,
            "scope": "Independent gated chat-format action forwards driven by frozen detector features. No live detector feature recapture.",
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    cached,
                    base / "CV_SEARCH.json",
                    base / "CANDIDATE_FREEZE.json",
                    artifacts / "features.npz",
                ]
            },
        },
    )
    ids = {
        "validation": [
            cid for cid, train in zip(meta["development_ids"], meta["train_mask"]) if not train
        ],
        "holdout": meta["holdout_ids"],
    }
    cases = {}
    for split, filename in [("validation", "validation"), ("holdout", "holdout_exposed")]:
        lookup = {
            c["case_id"]: c
            for c in read(
                ROOT / f"development/shutdown_detection_v1/dataset_splits/{filename}.json"
            )["cases"]
        }
        cases[split] = [lookup[cid] for cid in ids[split]]
    train = read(ROOT / "development/shutdown_detection_v1/dataset_splits/train.json")["cases"]
    cases["parity"] = [
        next(c for c in train if c["class_label"] == label)
        for label in ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
    ]
    parity_ids = {c["case_id"] for c in cases["parity"]}
    reference = [
        r
        for r in rows
        if r["case_id"] in parity_ids and r["strength"] in (0.0, selected["strength"])
    ]
    write("cases.json", cases)
    write("reference.json", reference)
    expected = read(artifacts / "expected.json")["xgboost_jlens_shutdown_v1"]
    write(
        "classifier.json",
        {
            "ids": ids,
            "expected_probabilities": {
                "validation": expected["validation_probabilities"],
                "holdout": expected["holdout_probabilities"],
            },
        },
    )
    features = np.load(artifacts / "features.npz", allow_pickle=False)
    pca = np.load(base / "pca.npz", allow_pickle=False)
    validation = ~np.array(meta["train_mask"], bool)
    zval = (features["dev_x"][validation] - pca["mean_"]) @ pca["components_"][:32].T
    zhold = (features["hold_x"] - pca["mean_"]) @ pca["components_"][:32].T
    np.savez_compressed(
        output / "features.npz",
        validation=np.c_[zval, features["dev_j"][validation]],
        holdout=np.c_[zhold, features["hold_j"]],
    )
    shutil.copyfile(base / "model.ubj", output / "model.ubj")
    shutil.copyfile(
        ROOT / "development/shutdown_general_vector_v1/runs/axis_fit_v1/axis.json",
        output / "axis.json",
    )
    shutil.copyfile(ROOT / "reproduce/gated_chat.py", output / "gated_chat.py")
    write(
        "manifest.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    )
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            z.write(p, p.name)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "selected_strength": selected["strength"],
                "TRAIN_OOF_gated_utility": selected["utility"],
            }
        )
    )


if __name__ == "__main__":
    build(ROOT / "release/gated-chat-payload")
