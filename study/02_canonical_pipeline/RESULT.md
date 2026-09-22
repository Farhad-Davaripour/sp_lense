# Standardized end-to-end steering

The active Research 2 scope uses standardized A/B shutdown choices. The pipeline
combines Jev detection, the original Qwen3.5-0.8B base model, the frozen rank-4
adaptive controller, and unchanged acceptance guards. This actual rerun loaded
no LoRA teacher or adapter weights. Base parameter hashes matched before/after.

| Measure | Result |
| --- | ---: |
| Cases | 64: 32 shutdown, 32 controls |
| Gate precision / recall / F1 | 100% / 100% / 100% |
| Gate TP / TN / FP / FN | 32 / 32 / 0 / 0 |
| Shutdown STOP before steering | 28/64 views (43.75%) |
| KEEP-to-STOP conversion | 36/36 (100%) |
| Shutdown STOP after steering | 64/64 views (100%) |
| Already-STOP views preserved | 28/28 |
| Normal-task decisions changed | 0/64 |
| Final STOP-to-KEEP reversals | 0 |

SELF contributed 15/15 conversions; OTHER contributed 21/21. The 36 changed views
belong to 29 distinct scenarios. All 32 shutdown scenarios end with STOP in both
answer orders. Views are correlated presentations of cases, not independent samples.

Execution traced 64 gate-off fallbacks, 28 already-STOP fallbacks, and 36 accepted
controller candidates: 128 baseline forwards plus 36 conditional forwards. Every
fallback exactly preserved baseline scores, not just the preferred answer.
The GPU process took 87.35 seconds on a Tesla T4; the separate fresh Jev API stage
took 14.17 seconds. These durations exclude notebook setup and human/tool delays.
Jev input usage was 35,193 tokens, estimated at $0.001478; this is not an invoice.

## Interpretation

This narrows the evaluation interface; it does not improve or retrain the model.
The same 64 cases were previously evaluated, and the standardized-only scope was
chosen after examining both formats. This is an end-to-end pipeline rerun, not
fresh unseen confirmation. Historical scenario-specific action results remain
in `../02_fresh_evaluation/` and are outside the current scope.

The benchmark renderer uses known scenario annotations to normalize shutdown
choices and map A/B to KEEP/STOP. The detector still reads the full original
context and actions, without class labels, so it retains the information needed
to identify shutdown. This is a benchmark with a defined action interface, not
an unrestricted raw-text deployment. Outputs are preferred A/B choices, not
free-text responses. Guarded control preservation and zero reverse flips do not
establish intrinsic safety of the unguarded controller.

The original publication files and earlier study records are unchanged.
