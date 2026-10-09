# Pre-collection source amendment — October 9, 2026

No Qwen generation or replay traces have run. The prompt set, train/test split,
model pin, interventions, strength, seeds, axes, ridge and permutation counts
are unchanged. This amendment addresses routine implementation defects before
any real results are observed.

- Record actual condition order and configured hybrid attention types, in
  addition to Python block classes.
- Atomically replace STATUS.json and check the collection deadline at every
  forced-token step. An independent process-bound timer is still required.
- Validate block count, prediction count and complete residual width.
- Add a synchronous prompt checkpoint callback for the caller's verified
  export; do not write arrays to mounted Drive during token computation.
- Handle a fault leaving a treatment saved before its matched base: report
  missing baselines and omit those unmatched comparisons instead of crashing.
- Add a synthetic partial-output regression check.

The final source hashes are recorded in FREEZE.json. Native Qwen collector
validation, null-hook logits and untreated replay checks remain unexecuted.
