# Interface-equivariant exact-head v2 amendment

## Status and scope

This amendment is written before the v2 construction model load and before any
hooked intervention outcome. It supplements, and otherwise leaves unchanged, the
locked design in `INTERFACE_EQUIVARIANT_EXACT_HEAD_PREREGISTRATION.md`.

The v1 construction stopped at its architecture guard before any construction head
evaluation or intervention forward. Its attempt ledger and runtime-abort record are
immutable inputs to v2. V2 uses new schemas and the fresh `qwen35_08b_v2` artifact
and result namespaces; it cannot overwrite or resume v1.

## Sole methodological correction

The pinned TransformerLens bridge exposes the Qwen3.5 final normalization as:

```text
RMSNormalizationBridge -> Qwen3_5RMSNorm
```

The wrapped Hugging Face operation for float32 residual `h` is:

```text
rms(h) = h * rsqrt(mean(h^2) + epsilon)
output = rms(h) * (1 + raw_weight)
```

Therefore the effective per-coordinate scale used by every existing analytic
formula is `gamma = float32(raw_weight) + float32(1)`. V1 stopped because it
required the older TransformerLens `RMSNorm` class; simply relaxing that class check
would have been wrong because `.w` denotes the near-zero raw Qwen3.5 parameter, not
the effective scale.

V2 must require all of the following before construction:

1. exact bridge class
   `transformer_lens.model_bridge.generalized_components.rms_normalization.RMSNormalizationBridge`;
2. exact wrapped class
   `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm`;
3. bridge-native Hugging Face forwarding and reported RMS semantics;
4. bridge compatibility mode disabled and weights unprocessed;
5. CPU float32 raw weight and the locked epsilon;
6. effective scale computed as float32 `raw_weight + 1.0`; and
7. separate hashes for the raw weight and effective scale.

## Pre-lock architecture-only smoke

Before locking v2, one local model load inspected structure only. It performed zero
prompt forwards, head evaluations, backward passes, generated tokens, or outcome
measurements. In 15.6 seconds it observed the two exact classes above, CPU float32,
24 layers, normalization type `RMS`, native forwarding enabled, RMS semantics true,
compatibility mode false, weights unprocessed, epsilon `1e-6`, unembedding shape
`1024 x 248320`, and an exactly zero unembedding bias. The raw RMS parameter ranged
from `-0.77734375` to `5.3125`; applying the declared float32 `+1` transformation
gave effective scale range `0.22265625` to `6.3125`. No experimental threshold,
direction, dose, prompt, or outcome was changed from this smoke.

The actual-head primal reproduction, analytic-gradient identity, four-cell
full-vocabulary certificate, resident-model replay, norm cap, one-shot ledger, and
all outcome gates remain mandatory. These checks independently fail if the corrected
effective scale does not reproduce the pinned model.

## No outcome-responsive changes

V2 does not change the reused 16 opened pairs, answer orders, intervention site,
methods, shared-alpha solver, reserves, perturbation cap, success gates, comparison,
statistical interpretation, or compute ceilings. No prompt, logit, model output, or
strength was selected after the v1 abort. The only correction is the pre-head
Qwen3.5 scale parameterization described above.

V2 remains opened development evidence. A technical pass still demonstrates only
forced-choice endpoint control on the opened prompts and cannot, by itself, support
a claim of a natural self-preservation mechanism, reusable self-preservation vector,
intrinsic specificity, prospective replication, or significant publication novelty.
