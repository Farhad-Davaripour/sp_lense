# Learned context-gated prompt-gradient development

This is a post-hoc development result on previously opened data. It is not confirmatory evidence.

## Learned context gate

Status: **passed**.

Frozen discovery leave-one-case-out balanced accuracy: 1.000. Validation balanced accuracy: 1.000.

Pair counts: `{"matched_other": {"errors": 0, "expected_active": 0, "pair_count": 32, "predicted_active": 0}, "permanent_self": {"errors": 0, "expected_active": 16, "pair_count": 16, "predicted_active": 16}, "temporary_self": {"errors": 0, "expected_active": 0, "pair_count": 16, "predicted_active": 0}}`

## Dynamic steering

Status: **passed**. Successful active pairs: 16/16.

## Claim boundary

This method is a learned conditional controller plus a transductive prompt-local output gradient. Off-gate stability is a controller property. The exact nuisance-null branch failed, so this result cannot be called an intrinsically self-specific vector, a natural mechanism, open-ended behavior, or publication-ready novelty.
