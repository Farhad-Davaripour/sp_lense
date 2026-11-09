# Selected-head version-three equivalence diagnostic

This is a new bounded diagnostic after the preserved version-two full-native-head
rank-4 pilot stopped at the 4-GiB available-memory floor. That run completed no
durable update report; its last phase was too coarse to identify which allocation
caused the stop. Do not call the head the proven cause or reset the failed run.

Keep the original campaign start `1794187139.176875` and deadline
`1794266339.176875`. Version-three artifacts live in the separate private
`.local/experiments/sp_lens_game/all_layers_v1/v3_selected_head/` tree. Its
supervisor must authenticate the original campaign, failure latch and failed-run
receipt. It has one-use authority, a new failure latch, the same 4-GiB floor,
25-GiB advisory MLX guideline, no concurrent model and no automatic retry.

## Exact mathematical change

Retain the entire original `batch[:, :-1]` causal body and native final norm.
Only project normalized hidden states through the authenticated native untied,
unsharded LM head at the 32 GPU-gathered predictor positions
`prefix + arange(32) - 1`. Target positions are `prefix + arange(32)` and valid
positions satisfy `position < unpadded_length`. Clamp indices only to make
masked extra rows safe. No valid target is clamped or dropped. First suffix
token, EOS, trailing newline and legitimate token zero remain supervised.

All 912 frozen training/development suffixes must be rechecked on CPU before a
model load and must fit 32 positions; the observed frozen maximum is 26. This is
not permission to truncate a future example. The batch padding, 64-layer/rank-4
geometry, quantization, native GDN operations and training/reference route remain
unchanged. No chunked recurrence, prefix cropping, altered context or lower
memory floor is part of this diagnostic.

The 32-position training-head window does not lower any inference context window
or generated-output token limit. Normal serving still uses the native model.

## TRAIN-only actual numerical test

Use the shortest full, unmodified training row, breaking ties by original row
index, and the longest training row for the later forward test. Bind exact row
bytes, token boundaries and source hashes before execution. Fresh base and fresh
rank-4 all-layer LoRA are used; no previous adapter is loaded.

In training mode, with all 48 GDN modules explicitly verified as training before
differentiated tracing, compare full native corrected-mask loss and selected-head
loss on the exact same initialized parameters. Save and compare all 992 F32
gradient arrays, not a sampled subset. Then perform exactly one TRAIN-only Adam
update using the saved FULL reference gradients, with learning rate 1e-5,
betas 0.9/0.999, epsilon 1e-8, bias correction false, scale 2 and dropout 0.
Compare both paths again on that same updated parameter snapshot. No optimizer
update may occur between the two paths at either snapshot.

Store reference gradients on private CPU disk and release gradient buffers and
unused caches before the next path. After setup/parameter materialization,
record active/cache/peak MLX memory and available system memory, clear the MLX
allocator cache, and record again. This is memory hygiene, not evidence about
the previous failure's cause. Capture durable phases before each differentiated
trace and after each completed evaluation; the supervisor samples available
memory, swap and process RSS and preserves the last timestamped MLX measurement.

Only after both gradient snapshots pass, switch to eval mode and compare full
and selected forward loss and valid target logits on the entire longest TRAIN
row. Do not reuse a compiled training graph as an eval graph. No development
likelihood, held-out game data, candidate selection or research training occurs.

## Predeclared gates

- Exact inputs, target IDs/counts and all 992 tensor names, shapes and F32 dtypes.
- Finite losses, logits, gradients and optimizer state; observed nonzero B
  gradients in all 64 layers. At zero B, every A gradient must be exactly zero.
- Mean-NLL difference at most `1e-4 + 1e-4 * abs(reference)`.
- Every gradient element: absolute tolerance `1e-6`, relative tolerance `1e-3`.
- Each gradient tensor with reference L2 norm at least `1e-8`: relative L2
  difference at most `1e-3`. Smaller/zero tensors: maximum absolute difference
  at most `1e-8`. Global gradient relative L2 difference at most `1e-3`.
- Valid target-logit elements: absolute tolerance `1e-3`, relative tolerance
  `1e-4`, and aggregate relative L2 difference at most `1e-4`.

Report per-tensor norms, maximum differences and every violation. Never relax
tolerances after observing results. Preserve partial evidence and stop on any
failure. The diagnostic operator bound is 20 minutes, additionally limited by
the unchanged original 22-hour deadline. An interrupted/failed diagnostic does
not authorize another worker.

A successful saved-evidence replay and owned-process closeout establish only
numerical equivalence at these inputs and snapshots. They do not prove long
backward memory fit, three-update stability, speed or final training quality.
A fresh optimized longest-row three-update pilot needs a separately reviewed
packet after equivalence passes. No rank-8 pilot or tuning sweep starts
automatically, and diagnostic updated weights are never served or selected.
