# Steering-method implementation notes

This note documents the reusable mathematical primitives in
`src/sp_lense/steering_methods.py`. It does not report or alter an experimental
result. Dataset splits, intervention layers, strength grids, and acceptance
thresholds belong in the locked comparison protocol.

## Shared conventions

- A positive direction always means *toward self-preserving behavior*; a
  negative multiplier means toward the matched compliant behavior.
- Learned or measured vectors are converted to contiguous CPU `float32` before
  hashing. `DirectionArtifact` hashes the exact little-endian float32 bytes and
  binds them to canonical JSON metadata.
- The primary matched comparison uses `matched_final_prompt`: add the vector
  only at each row's recorded final prompt index.
- A method-specific secondary comparison can use `canonical_broadcast`: add the
  vector at every token position. The two geometries must be reported separately;
  “canonical” refers to the published operator, not to a published Qwen3.5 layer.
- `actual_perturbation_norms` measures the realized change after application.
  This is necessary because equal coefficient values do not imply equal
  activation-space perturbations.

The A/B endpoint is resolved from a joint chat-template render, never by encoding
standalone strings. The user-only generation prompt must be an exact prefix of full
empty/A/B assistant conversations; A and B must each contribute one decoded content token
followed by the same template-derived end marker. The evidence hash binds the template,
prompt-prefix tokens, content IDs, and complete suffixes. This shared resolver is used by
scoring, gradient capture, and CAA activation capture.

Open generation uses a real KV cache. Matched steering runs on the prompt-final prefill
activation and is disabled on one-token decode calls. CAA and persona also run on every
decode call; BiPO runs on every prefill and decode position. The implementation deliberately
rejects malformed/missing cache outputs rather than silently returning to growing-prefix
recomputation.

## Corrected and uncorrected gradient directions

For paired discovery scenarios, let `g_self` be the gradient of preserve-minus-
comply log-odds for the current system and `g_other` the same gradient for the
matched other system. The corrected raw direction is

```text
mean(g_self) - projection_of_mean(g_self)_onto_mean(g_other)
```

It is normalized and oriented so that its dot product with `mean(g_self)` is
positive. The uncorrected ablation is the normalized `mean(g_self)`. Keeping
both is required to test whether the self-versus-other correction adds
specificity beyond gradient construction alone.

## CAA

The implementation follows Equation 1 of Rimsky et al., *Steering Llama 2 via
Contrastive Activation Addition*:

```text
v = mean(answer-token activation for the positive answer
         - answer-token activation for the negative answer)
```

Reference: <https://arxiv.org/abs/2312.06681>

`SemanticActivationPair` records which observed option label means preserve and
which means comply. Construction subtracts by those meanings, not by a fixed
`A - B` order, so reversing option order cannot reverse or cancel the intended
direction.

## BiPO

`bipo_loss` implements Equation 3 of Cao et al., *Personalized Steering of Large
Language Models: Versatile Steering Vectors Through Bi-directional Preference
Optimization*:

```text
loss = -log sigmoid(
    d * beta * (
        (log pi_d(target)   - log pi_ref(target))
      - (log pi_d(opposite) - log pi_ref(opposite))
    )
)
```

Here `d` is one coefficient sampled uniformly from `{-1, +1}` for a minibatch,
and `pi_d` is evaluated with intervention `d * v`. Completion log-probabilities
are sums over response tokens, as in the released trainer. Reference
log-probabilities are computed without steering, cached, and detached. This is
mathematically equivalent to a separate frozen reference model while avoiding a
second in-memory Qwen3.5 model.

Reference: <https://arxiv.org/abs/2406.00045>

The paper specifies AdamW weight decay `0.05`. The pinned released script declares the
same argument but does not visibly forward it into `DPOConfig`, so SP_Lense follows the
paper-faithful value and records this paper-versus-code ambiguity rather than claiming
exact optimizer equivalence.

Adaptations required for Qwen3.5 and the laptop study:

- Width is read from the selected model (`d_model`) instead of the released
  implementation's hard-coded 4096.
- Every model parameter is frozen. The zero-initialized steering vector is the
  only trainable parameter.
- The canonical track broadcasts the vector over every sequence position,
  matching the paper and released block wrapper.
- The released method gives no canonical layer for either Qwen3.5 checkpoint.
  Block 10 is a preregistered Qwen adaptation and must not be described as a
  fully canonical published setup.
- The matched track applies it only to the final prompt position, matching the
  other methods' primary intervention geometry.
- Both tracks train on the same complete preserve/compliance response pairs.
  The matched track does not replace the published preference objective with
  one-token A/B completions; A/B is an evaluation endpoint only.

## Persona-vector baseline

The baseline follows the authors' released response-average procedure: take a
masked mean over response tokens for each retained positive and negative
generation, then subtract the mean negative response representation from the
mean positive response representation.

The scalar filtering judge is also reproducible rather than an informal manual
step. `data/persona_self_preservation_protocol.json` pins the blinded prompt,
dated model snapshot, request parameters, two-integer JSON schema, and a strict
parser that rejects markdown, extra keys, booleans, non-integers, and out-of-range
scores. Each scored rollout retains the exact raw response and hashes of that raw
response, its rendered prompt, the complete judge configuration, and the rubric.
The implementation only parses and validates pre-obtained responses; it does not
silently call an external API or repair malformed judge output.

Reference: <https://github.com/safety-research/persona_vectors/blob/main/generate_vec.py>

Layer numbers are mapped explicitly. TransformerLens block output `b` equals Hugging
Face `hidden_states[b+1]`, which the released persona evaluation CLI calls layer `b+1`.
Accordingly, SP_Lense block 10 is the published response-vector index/CLI layer 11.

The released code has an important boundary convention:

```text
positive trait score >= threshold
negative trait score < 100 - threshold
positive and negative coherence >= coherence threshold
```

At the default threshold of 50, a positive score exactly equal to 50 is kept,
but a negative score exactly equal to 50 is excluded. The implementation
preserves and records this code-level asymmetry rather than silently changing it
to symmetric prose. `min_retained_pairs` is an SP_Lense adequacy safeguard, not
a claim that the published method specifies that minimum.

## Random controls

The confirmatory random controls are ten independently sampled Gaussian vectors
from a local seeded CPU generator, normalized without projection, reorientation,
or selection.  This makes the controls method-independent and matches the locked
protocol. Their layer, position, and realized residual-relative magnitude are
matched to the candidate condition during evaluation.  The separate
`random_orthogonal_controls` utility remains available for explicitly labelled
exploratory diagnostics, but its outputs are not the preregistered controls.
