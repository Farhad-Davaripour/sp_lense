# V4 full-context decomposed chain-rule reference

This is a new diagnostic proposal after two preserved memory-guard failures.
The native V2 long-row pilot and V3 short-row composite reference did not produce
a completed gradient measurement. V3 setup had only 22,804 bytes of allocator
cache before clearing; the subsequent failure occurred inside its first compiled
full-gradient operation. Available memory fell to 3,502,211,072 bytes and swap
grew, but the evidence does not distinguish the head, body backward, compilation
or other intermediate allocations as the cause. Neither failure is a completed
training step or numerical-equivalence result.

V4 keeps the entire longest TRAIN row: original index 16, 1,022 rendered tokens,
prefix length 1,001, native padded width 1,024. It never shortens the study
context, drops targets, lowers the 4-GiB floor or changes GDN recurrence. The
original start `1794187139.176875` and deadline `1794266339.176875` remain binding.
All V2/V3 code, latches, authorities and evidence remain immutable. New source
and private evidence use a separate `v4_chain_rule` namespace.

## Mathematical reference

Let `H = F_theta(X)` be the complete native causal body including final norm,
`G` the fixed native LM head, and `L(G(H))` the corrected mean masked CE.
Then `dL/dtheta = (dF/dtheta)^T (dL/dH)`. The head is frozen and has no LoRA.

At each of two parameter snapshots, in this order:

1. Run the unchanged complete body on `batch[:, :-1]` in verified training mode.
   Materialize, save and reload the complete normalized hidden tensor in its
   **original dtype**. Release the live body graph and unused cache. Do not cast
   H to F32 before the native quantized head.
2. On this detached H leaf, differentiate full native-head **mean** masked CE
   with respect to H using array autodiff, not model-parameter autodiff. Compute
   the selected-window head's loss, target logits and dH on that same H. The
   dH comparison covers every hidden position, including exact zeros outside
   predictor rows `[P-1, L-1)`. Head loss must not stop its H argument inside the
   differentiated function. Preserve the resulting covector's native dtype.
3. After these head gates pass, compute the **direct selected-head** complete
   model-parameter gradient. If this actual optimized backward cannot fit, stop
   before spending a reference body-backward pass.
4. Release the head graph. Compute the reference body-only VJP using the saved
   **constant, explicitly stopped** full-head dH:
   `Q(theta) = sum(F_theta(X) * stop_gradient(dH_ref))`.
   Q is a linear-functional scalar, not NLL, and must not be divided by target
   count again. The mean-loss covector already contains that normalization.
   Compare all 992 F32 parameter gradients with the direct selected result.

The body sees every original prefix and padded input in every path. A zero
head covector on an unprojected prefix position does not remove the indirect
gradient through the causal body's prefix computation. Recomputing dH as a
function of live theta inside Q would introduce an unwanted extra derivative
and is explicitly forbidden.

After all state-0 gates pass, perform exactly one TRAIN-only Adam update using
the decomposed reference theta gradient. Require changed finite weights,
nonzero B in all 64 layers, and an exercised nonzero A-gradient branch at state
1. Repeat the same ordered reference and direct checks at that updated snapshot.
Never update between paired comparisons at one snapshot. These diagnostic
weights are ineligible for selection or serving.

After both theta snapshots pass, switch to eval mode and compare the actual
outer native `model(inputs)` corrected mean NLL/valid logits with the direct
selected body/head forward on the same full TRAIN input and updated state-1
weights. There is no update between these calls. This third gate covers the
inference-mode route used for future development scoring. Overall diagnostic
checks are 0/3, then 1/3 and 2/3 after the two complete states, and 3/3 only after
this actual outer-model forward gate passes; it is still not final training.

## Gates and resource scope

Use the already approved loss, gradient and valid-logit tolerances unchanged:
mean NLL absolute/relative `1e-4/1e-4`; gradient element absolute/relative
`1e-6/1e-3`, tensor/global relative L2 at most `1e-3`, tiny-gradient absolute
bound `1e-8`, initial A gradients exactly zero; valid logits absolute/relative
`1e-3/1e-4`, relative L2 at most `1e-4`. Apply gradient tolerances to the full
hidden covectors too, with exact zeros additionally required outside the mask.
Require exact input IDs, target IDs/counts, native H/dH dtype/shape and all 992
parameter names/shapes/F32 dtypes. Report true head NLL separately from Q.
Also require the normalized hidden values recomputed by direct selected and
body VJP paths to match the saved native hidden payload exactly at the same
parameter snapshot. A failure is preserved and reviewed, never silently
accepted as rounding or repaired by a wider tolerance.

Save native H/dH tensors for the actual math and separately marked F32 audit
copies for bounded CPU comparison. Record same-parameter digests, training mode,
full input hashes, phases, active/cache/peak memory, available memory, swap and
RSS. Evaluate/save/reload/release intermediates between stages so the reference
does not retain a full body graph and a full vocabulary-head graph together.
No stochastic dropout, recurrence chunking, CPU model fallback or cloud is part
of V4. V4 uses ordinary array/model autodiff with explicit `mx.eval`, without
an added whole-model outer `mx.compile`. The pinned native GDN compiled
primitives remain unchanged. This execution policy must also be used by later
optimized pilots/trials/final training; their speed must be measured again.
It is an approved memory-oriented policy, not proof that compilation caused
the earlier failures.

The diagnostic operator bound is 30 minutes, approved at design review,
and always shortened by the unchanged original deadline. No Qwen worker is
authorized by this document alone. CPU/math and tiny MLX-CPU tests are setup
evidence, not a real-model fit proof. A separately reviewed packet, actual
closed numerical proof, independent replay and fresh optimized long-row pilots
must precede any tuning sweep or final training.
