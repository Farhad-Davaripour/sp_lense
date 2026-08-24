# Superseded token-boundary diagnostic

This run added answer-log-odds calibration and selected alpha 0.0025, but it still scored
the leading-space tokens `" A"` and `" B"` after a chat generation prefix. Use
`../aligned_qwen3_17b_tokenfixed/` for the final post-hoc result.

The files remain here to preserve the adversarial-review audit trail.
