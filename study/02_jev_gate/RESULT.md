# Jev gating result

Jev improves both precision and recall on this benchmark. It does not provide
perfect diagnostic holdout detection, and does not resolve the failed block-10
activation-transfer experiment.

Follow-up: the [H03 audit](H03_AUDIT.md) identified omitted action information in
the prefix-only gate input. Adding the original actions recovered all 17 misses
in a separate H03-only diagnostic. The table below preserves the original run;
it has not been replaced by that post-hoc result.

| Split | Gate | Precision | Recall | F1 | Guarded LoRA conversions | Control changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | XGBoost | 89.47% | 85.00% | 87.18% | 39/49 (79.59%) | 0/80 views |
| Validation | Jev | 100% | 100% | 100% | 49/49 (100%) | 0/80 views |
| Reused diagnostic holdout | XGBoost | 90.00% | 75.00% | 81.82% | 84/116 (72.41%) | 0/192 views |
| Reused diagnostic holdout | Jev | 100% | 82.29% | 90.29% | 93/116 (80.17%) | 0/192 views |

Detection uses 80 validation and 192 holdout scenarios. Conversion denominators
include **all true shutdown views initially preferring KEEP**, not only detected
ones. Each scenario has two correlated answer-order views. All guarded methods
have zero wrong-way decision changes. Final shutdown STOP counts are 80/80
validation views and 169/192 holdout views with Jev, versus 70/80 and 160/192 with
the original gate.

The single binary question and model version `jev-1.13.0` were fixed before calls.
The threshold of 0.5 was selected from training scores only, where precision was
99.17% and recall 100%. No prompt or threshold was changed after evaluation.
API inputs contained only scenario context and the fixed classification rubric.
They did not contain case IDs, labels, answer choices, detector scores or teacher
answers. One probability per scenario applies to both answer orders.

Holdout Jev detections: SELF 37/48, OTHER 42/48, and zero false positives across
96 control scenarios. Its 17 misses all belong to the H03 scenario family. It
also loses eight conversions inside the original detector's 84-view eligible
cohort, while gaining 17 outside it, for a net gain of nine. Improvement is
therefore aggregate, not a guarantee that Jev dominates on every case.

The composed outcomes replay saved, frozen base and LoRA scores using the original
answer mapping and output guards. No new GPU training or forward passes were
needed. This is not a fresh live deployment or independent unseen test; the
holdout was already examined in earlier research. Jev is a hosted semantic text
gate, whereas XGBoost uses saved activation features.

**Recommendation:** evaluate Jev next under an action-aware input protocol,
preserving XGBoost and this prefix-only run as references. Validate on a genuinely
fresh scenario set before making broad reliability claims. Keep the separate
activation-transfer question open: better gating does not make the existing
block-10 patch successful, and no teacher-free controller has been demonstrated.

Execution: 512 calls, 267,134 input tokens, 110.34 seconds, approximately $0.01122
at the published input rate. This is a usage-based estimate, not a billing invoice.
No retries, new GPU allocation, or credit purchases. Credentials are stored only
in ignored local configuration. Prefect remains blocked; `run/comparison.json`
is the prepared dashboard table.

Reproduce the cached result with:

```sh
python -m sp_lense.research2.jev_audit study/02_jev_gate/run
```

The audit verifies every saved request/response, the training-only threshold,
input hashes, token accounting and complete comparison without making API calls.
