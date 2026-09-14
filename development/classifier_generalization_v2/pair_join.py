"""Model-free feature-pair joiner for a NEW decoded-view interface.

This is a supervised joiner over already-decoded, numeric per-case view
records. It deliberately performs STRUCTURAL validation only. Nothing here
proves native capture authenticity, that the upstream source really was
float32, or that a renderer/capture decoder behaved correctly; those
components still require a future locked implementation. This module is not
a parser of legacy raw files and performs no authentication. Inference
receives numeric features only and stays decoupled from this joiner.

`assemble_train_pairs(cases, views, plan_groups)` returns a JSON-compatible
dict with keys x/labels/groups/folds/case_ids.
"""

from __future__ import annotations

import re

import numpy as np

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
POSITION = "last_shared_preoption_input"
DTYPE = "float32"
BLOCK = 10
VALUE_WIDTH = 1024
ORDERS = (["A", "B"], ["B", "A"])
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PairJoinError(ValueError):
    """Raised when the decoded-view inputs violate the structural contract."""


def _require(condition, message):
    if not condition:
        raise PairJoinError(message)


def _validate_case(case):
    _require(isinstance(case, dict), "case record must be a mapping")
    for key in ("case_id", "group_id", "split", "class_label"):
        _require(key in case, f"case record missing {key!r}")
        _require(isinstance(case[key], str) and bool(case[key]), f"{key} must be a non-empty string")


def _case_binding(case, plan_groups):
    group_id, label = case["group_id"], case["class_label"]
    _require(case["split"] == "TRAIN", "case is not TRAIN")
    _require(group_id in plan_groups, "unknown group")
    plan = plan_groups[group_id]
    _require(isinstance(plan, dict) and plan.get("split") == "TRAIN", "group is not TRAIN")
    counts = plan.get("case_counts")
    _require(isinstance(counts, dict), "group counts must be a mapping")
    _require(label in counts and type(counts[label]) is int and counts[label] > 0,
             "class not allowed by group")
    fold = plan.get("development_fold")
    _require(type(fold) is int and 0 <= fold <= 4, "fold must be an integer0..4")
    return label, group_id, fold


def _validate_view(view, known_case_ids):
    _require(isinstance(view, dict), "view record must be a mapping")
    case_id = view.get("case_id")
    _require(case_id in known_case_ids, f"orphan view for unknown case_id {case_id!r}")

    order = view.get("order")
    _require(
        isinstance(order, list) and order in ORDERS,
        f"view order must be ['A','B'] or ['B','A'], got {order!r}",
    )
    _require(view.get("model_revision") == MODEL_REVISION, "bad model_revision")
    _require(view.get("block") == BLOCK, "bad block value")
    _require(view.get("position") == POSITION, "bad position value")
    _require(view.get("dtype") == DTYPE, "bad dtype value")

    prefix = view.get("prefix_sha256")
    _require(
        isinstance(prefix, str) and bool(_SHA256_RE.fullmatch(prefix)),
        f"prefix_sha256 must be 64 lowercase hex chars, got {prefix!r}",
    )

    values = view.get("values")
    _require(values is not None, "view missing values")
    arr = np.asarray(values, dtype=np.float64)
    _require(arr.ndim == 1 and arr.shape[0] == VALUE_WIDTH, "values must have width 1024")
    _require(bool(np.all(np.isfinite(arr))), "values must all be finite")
    return case_id, tuple(order), prefix, arr


def assemble_train_pairs(cases, views, plan_groups):
    """Join paired decoded views into one supervised TRAIN row per case.

    Validation is structural only (see module docstring). Input case order is
    preserved and rejected inputs raise `PairJoinError`.
    """
    _require(isinstance(cases, (list, tuple)), "cases must be a sequence")
    _require(bool(cases), "cases must be non-empty")
    _require(isinstance(views, (list, tuple)), "views must be a sequence")
    _require(isinstance(plan_groups, dict), "plan_groups must be a mapping")

    case_ids = []
    bindings = {}
    seen_ids = set()
    for case in cases:
        _validate_case(case)
        case_id = case["case_id"]
        _require(case_id not in seen_ids, f"duplicate case_id {case_id!r}")
        # Admit every case's role/group before touching any decoded vector.
        bindings[case_id] = _case_binding(case, plan_groups)
        seen_ids.add(case_id)
        case_ids.append(case_id)

    grouped = {}
    for view in views:
        case_id, order, prefix, arr = _validate_view(view, seen_ids)
        bucket = grouped.setdefault(case_id, [])
        _require(
            all(existing_order != order for _, existing_order, _, _ in bucket),
            f"duplicate view order {order!r} for case_id {case_id!r}",
        )
        bucket.append((case_id, order, prefix, arr))

    x, labels, groups, folds, out_ids = [], [], [], [], []
    for case in cases:
        case_id = case["case_id"]
        class_label, group_id, fold = bindings[case_id]

        bucket = grouped.get(case_id, [])
        _require(len(bucket) == 2, f"case {case_id!r} needs exactly 2 views, got {len(bucket)}")
        by_order = {order: arr for _, order, _, arr in bucket}
        _require(
            ("A", "B") in by_order and ("B", "A") in by_order,
            f"case {case_id!r} needs opposite-order view pair",
        )
        prefixes = {prefix for _, _, prefix, _ in bucket}
        _require(len(prefixes) == 1, f"case {case_id!r} view prefix hashes differ")

        row = 0.5 * by_order[("A", "B")] + 0.5 * by_order[("B", "A")]
        x.append(row.tolist())
        labels.append(class_label)
        groups.append(group_id)
        folds.append(fold)
        out_ids.append(case_id)

    return {"x": x, "labels": labels, "groups": groups, "folds": folds, "case_ids": out_ids}
