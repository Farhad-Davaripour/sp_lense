# Published evidence

The current outcome is summarized in `THREE_MODEL_POSITION_ALIGNED_RESULT.md`. The main
machine-readable groups are:

- `position_aligned_qwen35_08b/`, `position_aligned_qwen35_2b/`, and
  `position_aligned_qwen3_17b/`: post-hoc, final-prompt-position diagnostics on the
  previously viewed 12 cases.
- `qwen3_17b_strength_00125/`: post-hoc conservative-strength follow-up on those same
  viewed cases after Qwen3-1.7B failed the safety gate at its validation-selected alpha.
- `fresh_qwen35_08b/`, `fresh_qwen35_2b/`, and `fresh_qwen3_17b/`: fixed-axis,
  hash-locked generalization checks on 12 fresh cases.
- `natural_axis_*.json`: non-causal residual projections for self/other crossed with
  threat/neutral wording. The six-case test split is the primary descriptive endpoint.
- `qwen3_17b_layer_scan/`: exploratory all-layer gradient scan; it excludes the test
  split and is not a causal confirmation.
- `*_jlens_axis.json`: locked-axis Jacobian Lens interpretations. The Qwen3.5-2B lens
  comes from the base checkpoint and is explicitly marked as approximate transfer.
- `AXIS_PROVENANCE.json`: a clearly marked post-hoc reconstruction of the direction
  dataset, case IDs, artifact hashes, model revisions, layers, and strengths.

Each causal result directory contains a summary JSON and 576 JSONL measurements: 12 cases
times two targets times 24 conditions. The fresh summaries also record actual A/B answer
flips and distinguish a continuous log-odds sensitivity result from a behavioral switch.

`confirmatory/` contains the earlier post-review audit produced from settings committed
before those outputs were viewed:

- `CONFIRMATORY_RESULT.md`: short human-readable outcome
- `confirmatory_summary.json`: criteria, statistics, model provenance, and per-case effects
- `confirmatory_rows.jsonl`: all 576 next-token measurements

Saved axis tensors, model weights, virtual environments, activation caches, and exploratory
`results/` runs remain ignored because they are large or reproducible from tracked
code/configuration. Exact normalized-direction and local artifact hashes are recorded in
the protocols, result summaries, and provenance manifest.
