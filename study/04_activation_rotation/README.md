# Research 4: narrow rotation pilot

Question: do Qwen residual activations contain reproducible rotational structure
across generated-token steps, and does a fixed steering intervention change it?
Analyze processing depth separately. This is a geometry pilot, not another
self-preservation success-rate study.

## Frozen scope

- One existing frozen Qwen checkpoint, revision
  `2fc06364715b967f1860aea9cf38778875588b17`, Qwen/Qwen3.5-0.8B in BF16 without quantization.
- Eight fixed short prompts; greedy baseline generation, at most 64 new tokens.
- Three conditions: no intervention; the existing Research 2 KEEP-minus-STOP direction
  at block 11 (zero-based 10), strength 0.5; one seeded random unit vector at the
  same block and strength. No vector search, fine-tuning or JEV calls.
- Generate each baseline once, then replay exactly the same input token IDs in
  all three conditions. This isolates activation changes from changed text.
  These are controlled replays, not free-running steered behavior tests.
- Twenty-four replay traces plus eight short baseline-generation passes.
- Replays use native cached generation with forced baseline token IDs, not a
  parallel teacher-forced prefill. The untreated replay must reproduce the
  original greedy decisions; the zero-strength hook must preserve logits.
- Save every residual coordinate at every block, indexed by the position that
  predicts each generated token. No sampling of hidden dimensions.
- Inference deadline: 45 minutes from entry, including setup; reserve the last
  five minutes for output preservation. Do not buy credits or automatically
  retry. Fresh account/resource admission and an authenticated session are
  required before execution. CPU analysis follows GPU release.

## Keep the two axes separate

**Token axis (primary):** block 16 held fixed; prediction steps advance with token
index. This is discrete autoregressive time, not biological milliseconds.

**Depth axis (secondary):** token prediction positions 0, 15, 31 and 47 held fixed;
advance through all blocks. These are different computations with different
weights, so a curve here is depth geometry, not evidence of temporal recurrence.
Record actual block count and block types; the pinned Qwen implementation has
hybrid attention/recurrent machinery and is not a homogeneous stack.

The injection block and last block receive descriptive temporal summaries only. If a
sequence ends early, preserve it; do not pad, regenerate or tune prompts. Skip
missing anchors and report their counts. Fewer than three usable train or test
prompts means an insufficient-data result for that analysis.
Each depth anchor is evaluated separately. The direct jump at the injection block
is an expected intervention effect, not by itself disrupted rotational dynamics.

## What counts as evidence

Use the full residual width, not a selected 2-D picture. Fit a regularized
skew-symmetric linear velocity model on the four preassigned training prompts;
evaluate prediction error in the original full-dimensional coordinates on the
four held-out prompts. Also report an unconstrained linear model and a constant
velocity reference. Regularization is fixed, not chosen on held-out data.

The implementation uses an orthonormal basis of the complete numerical training
span to solve the full-space problem efficiently. It discards no nonzero
principal components. Held-out directions outside that span remain in the error
denominator. This does not establish rotation in unobserved directions.

Report per-prompt results, activation norms, local angular changes, baseline
versus steering differences and the matched random-vector result. Refit on 19
within-trajectory order permutations and 19 block-order permutations (separate
axes). These are limited pilot checks, not a covariance-preserving neuroscience
null. A positive result requires a later temporal-covariance-preserving control.
Do not equate smooth curvature, a good training fit, or a nice projection with
rotation. Do not call any change a disruption unless baseline structure is
supported first. Two-dimensional plots may illustrate results but cannot decide
them. A null result is a valid outcome; no automatic expansion follows.

The methods are motivated by [Churchland et al., 2012](https://www.nature.com/articles/nature11129)
and the warnings about surrogate controls in
[Elsayed & Cunningham, 2017](https://www.nature.com/articles/nn.4617).
This pilot is not an exact reproduction of either method and makes no claim of
brain equivalence. Full residual width is not the model's entire state: KV caches,
recurrent caches, attention and MLP internals are outside the measurement.

## Files and execution

`pilot.json` fixes prompts and settings. `collect.py` accepts the already-loaded
frozen model/tokenizer; it does not download, allocate a runtime, mount Drive or
call services. `analyze.py` is CPU-only. `test_analysis.py` checks a known rotation,
nonrotating drift, and full-space error accounting before any GPU run.

Keep source in this study directory and private native arrays outside Git, under
`C:/Users/farha/Research4/artifacts/rotation_pilot_v1` and a verified private Drive
copy. Work locally in Colab and checkpoint after each prompt; do not repeatedly
write per-token logs to mounted Drive. Record the model pin, dependency versions,
vector hash, token IDs, exact sequence lengths, condition order, timing, null-hook
check, finite activation delivery and export hashes. Verify copies before runtime
release, including on faults. No scheduling or new dashboard is part of this pilot.
