# Pair-Join Independent Review — V1

- Job ID: `classifier_pair_join_review_20260913_1234`
- Scope: `pair_join.py`, `test_pair_join.py`, `test_pair_preflight.py` (new decoded-view structural joiner only)
- Verdict: **ACCEPTED_STRUCTURAL_HELPER**

## Artifact hashes (native SHA256)

| File | SHA256 |
|---|---|
| `pair_join.py` | `2598226122bcd503e52f5d2b744812483a9f53eee6d18b4bb3f57615223b9881` |
| `test_pair_join.py` | `2e89179bc9a1f59b5c20e8ff264b9e419cfdc6fabdd6ee23f582e5cb4c4678c1` |
| `test_pair_preflight.py` | `cb748986d6287b3eda71d04c8d4f46e3a60b732d73e0184a9c72c949557774b0` |

Runtime: `.runtime/Scripts/python.exe` = CPython 3.12.14, NumPy 2.5.3.

## Test evidence (only the six fake unit tests)

`python -m unittest test_pair_join test_pair_preflight -v` → **Ran 6 tests, OK (0.082s)**. All six passed; no real estimator, capture, model, tokenizer, cache, fit or holdout touched.

## Contract verification

- **Admission before vector conversion** — `assemble_train_pairs` completes the full TRAIN/group/class/fold pass (`_validate_case` + `_case_binding`) over every case before the views loop. Probe (adversarial `__array__` poison): wrong-role case (`split="HOLDOUT"`) raises `PairJoinError: case is not TRAIN` with the poison object untouched (preflight test, PASS). Additionally a *bad-revision* and an *orphan* view were rejected with the poison array never inspected.
- **Two opposite orders per case** — `order in ORDERS`, duplicate same-order view rejected, `("A","B")` + `("B","A")` enforced; reversed input order yields byte-identical rows.
- **Matching prefix hash** — 64 lowercase hex enforced; differing pair prefixes rejected.
- **Frozen revision / block10 / preoption position** — each independently rejected when wrong (`bad model_revision`, `bad block value`, `bad position value`).
- **float32 declared source** — `dtype="float64"` rejected.
- **Width 1024, all finite** — 1023-wide, `nan`, `inf` all rejected.
- **Missing / duplicate / orphan views** — each rejected.
- **Canonical string class in test data** — root's correction verified: numeric `class_label=0` now raises `class_label must be a non-empty string`; tests use canonical `"SELF"`/`"OTHER"`.
- **Output** — one float64 average row per case (convex `0.5*a + 0.5*b`, exact to `atol=0`), Python-float (float64) elements, length 1024, keys `x/labels/groups/folds/case_ids`; class/group/fold are canonical `str`/`str`/`int`; no normalization applied.

## Non-blocking observations (not contract violations)

1. Malformed, non-numeric `values` (e.g. `"abc"`) surfaces NumPy's raw `ValueError` rather than `PairJoinError`; PairJoinError subclasses ValueError but the raw type is not the joiner's own. Outside the stated reject list (missing/duplicate/orphan/wrong role).
2. An unhashable `case_id` (e.g. a list) raises `TypeError` from set membership instead of `PairJoinError`.
3. `block` and `position` rejections are implemented but not covered by any of the six tests (verified only by my external probe).

Neither affects the six-test contract outcome.

## Out of scope — missing upstream authenticity (not a joiner defect)

Native capture authenticity, true float32 provenance, renderer/decoder correctness, and legacy raw-file parsing are deliberately absent per the module docstring. The `dtype`/`prefix_sha256`/`model_revision` fields are *declared* structural claims only; no authentication is performed and none is required of this helper. This is separate from, and must not be reported as, a joiner defect.

## Scope compliance

No files written beyond this report; no Git changes, configuration, installation, network, coordination-state edits, subagents, real model/cache/fit/holdout, or old checkout. Native SHA256 via `Get-FileHash -Algorithm SHA256`.
