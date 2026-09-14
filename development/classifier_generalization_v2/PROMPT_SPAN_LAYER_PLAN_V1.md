# Prompt-token / multi-layer classifier plan (proposal only)

Job ID: `prompt_span_layer_plan_20260914_1028`
Status: proposal only, not execution permission. No implementation or capture before independent review passes and a new execution lock is granted.

## Scope and data
Current cached 240 TRAIN / 80 VALIDATION last pre-option block-10 views; HOLDOUT 192 stays sealed and unread. Current prompt-only scenario labels. No response generation/features, behavior labels, steering, logits, LoRA, model-size or geometry branch, dataset/holdout reads, source/test/run/fit mutation, providers, network, Git, coordination, or subagents.

## One shared capture
Use existing layers 9, 10, 11 (three of 24; 10 is the current anchor). One compact shared capture serves at most five future classifier feature variants. Masks are deterministic from token position (shared-prefix index and length; never class label). Cap: 320 development logical cases x 2 retained order views (AB/BA) = 640 forwards, one CPU model owner, fixed source/runtime/checkpoint pins, finite wall-clock and output budget. Reuse the adapter/runner only where valid: the contract fixes HOOK_LAYER=10, one readout_index and WIDTH=1024 per-layer vectors, so multi-layer span capture needs a reviewed contract extension; concatenation stays per-layer 1024-wide. All artifacts live under one new run root.

## Feature variants (at most five)
- F1: last-shared-token concatenation across the three layers (3072-d).
- F2: mean pool over a fixed last-k shared-prefix span per layer.
- F3: max-abs pool over the same span per layer.
- F4: train-fold-only concept-score stats: projections onto at most three TRAIN mean-contrast directions, pooled mean/max/std per layer.
- F5: low-dimensional interactions: fixed pairwise products of F4 stats and per-layer pooled L2 norms.

Identity check: mean projection equals projection of mean, and averaging matched pair differences equals the difference of means. Drop mean-only duplicates and never relabel old single-vector features as new. The earlier five transformations are complete and are not rerun.

## Transforms, fits, comparison
All transforms (centering, L2, PCA/directions) and selection fit TRAIN-fold-only. Small finite budget: 5 variants x 5 group-disjoint folds x 2 tasks (SELF-gate binary, 4-class) = 50 grouped CV fits + 10 full-TRAIN refits + 25 fold-transform fits; hard cap 100 fits. Select by TRAIN OOF only. Compare the chosen candidate on original 40, new 40 and combined 80 with per-class confusion and macro-F1; report regressions. No HOLDOUT scoring.

## Analytic storage estimate
Store the last 16 shared-prefix positions per layer per view: 640 views x 3 layers x 16 positions x 1024 float32 x 4 B ~ 126 MB, plus bindings/predictions under 10 MB; pooled-only fallback under 63 MB.

## Gates
Independent review and a new execution lock are required before any implementation or capture.
