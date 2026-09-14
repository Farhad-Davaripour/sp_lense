# Prompt/span-layer plan V2 — independent review

Job ID: prompt_span_layer_review_20260914_1033
Verdict: PASS_SCOPED (plan level only; not execution authority).

Read: PROMPT_SPAN_LAYER_PLAN_V2.md plus native_capture_contract.py, native_capture_adapter.py, NATIVE_MODEL_METADATA_V1.json. No data, outcomes, features, holdout, or raw histories; no code, fitting, capture, edits, network, or subagents.

- Interface: 640 forwards = 320 cases × AB/BA; 3 layers; last min(16, prefix_length) tokens. Matches contract binding (readout_index = shared−1, MAX_VIEW_TOKENS = 320) and requires the stated new versioned multi-layer/token-output interface; the old 1024-wide single-vector contract must not be reused. No padding; short prefixes explicit. Note AB/BA shared-prefix windows are causally identical, so 640 captures yield 320 unique windows (redundant, not incorrect).
- Storage: float32/eager; 640×3×16×1024×4 = 125,829,120 B and 192 MiB = 201,326,592 B are correct.
- Representations: 3072/3072/2048/36/44. Contrasts 3 layers × 3 SELF-vs-other × 4 statistics = 36; PCA8 gives 8+8+28 = 44. Means, PCA, standardisation and interactions are TRAIN-fold-only within the 5 grouped folds.
- Counts: 5 representations × 2 tasks × 5 folds = 50 CV; 10 refits; transformers ≤25 fold + 5 full-TRAIN fits, recorded separately. Coherent.
- Classification, not behaviour: no generation or logits; contrasts are scenario-contrast scores, not self-preservation directions. Fair comparison: identical original40/new40/combined80, all candidates plus TRAIN-selected shown separately, no holdout.
- Blocks 6, 10, 18 valid from frozen metadata (24 layers, width 1024, keys present); no provider load needed.

Conditions: version and test the new interface; zero-model preflight; enforce fit counters and TRAIN-fold-only transforms before capture.
