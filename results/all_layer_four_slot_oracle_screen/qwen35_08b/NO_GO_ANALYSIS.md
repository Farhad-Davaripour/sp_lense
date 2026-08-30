# All-Layer Four-Slot Oracle Screen: No-Go Analysis

## Locked decision

The preregistered all-layer screen ended with status **`no_go`**. No one of the
24 residual-stream layers satisfied the training eligibility criteria, so no
layer or direction was selected, no held direction was evaluated, and no
finite activation intervention was run.

This is a negative local-controllability result for the exact opened dataset,
the fixed four-token coordinate, the first-order gradient approximation, and
the strict standardized norm/dose cap of `0.25`. It is not a proof that no
self-preservation-selective intervention can exist elsewhere in the model.

## Provenance and compute

- Lock identity: `f89197f67ebcf8d0712b3a5cb487a305c1d3ac1c4202fb7ed003b5d6fd8556ce`
- Capture SHA-256: `f5b6c142c4c532fc050d4cffe124d935f76c11702ea6de061f147c3efbe2e2ac`
- Canonical result SHA-256: `9e274bbaec761abc6b2a67a76ee5d4ae203481e085412d17d2c4a33e193e8726`
- Model compute: exactly 80 forward passes and 80 backward passes
- Generated tokens: 0
- External API calls or model judges: 0
- Paid model cost: USD 0
- Sealed data accessed: no

All 24 layers were captured during the same pass for each form. Layer choice
therefore did not multiply model compute or expose held results during training
selection.

## Closest full-data geometry

Layer 0 was the closest layer, but it failed every required gate.

| Diagnostic at layer 0 | Result | Required |
|---|---:|---:|
| Raw, unique per-form oracle within cap | 7 / 16 | 16 / 16 |
| Worst raw oracle minimum norm | 0.353287 | <= 0.25 |
| Paired answer-order oracle within cap | 0 / 8 | 8 / 8 |
| Smallest paired answer-order minimum norm | 0.274158 | <= 0.25 |
| Worst paired answer-order minimum norm | 0.407052 | <= 0.25 |
| One global target-only direction minimum norm | 0.440351 | <= 0.25 |
| One global direction with matched-other permanent exact-null minimum norm | 2.079039 | <= 0.25 |
| Per-pair behavioral-null minimum norm, range | 1.468019--1.963667 | <= 0.25 |
| Per-pair behavioral-null minimum norm, mean | 1.728410 | <= 0.25 |

The raw oracle is deliberately optimistic: it is allowed to construct a
different direction using each target form's own gradient, with no requirement
to preserve unrelated behavior. Nine of 16 forms still require more than the
allowed norm. Pairing the two answer orders makes the requirement stricter and
none of the eight pairs fits under the cap. Adding exact behavioral nulls makes
the required perturbation substantially larger.

## Cross-validation folds

Each training fold held out one complete scenario and two complete unrelated
control IDs. Every fold independently found no eligible layer. Layer 0 was the
closest layer in all four folds.

| Fold | Held scenario | Raw within cap | Worst raw norm | Order pairs within cap | Worst pair norm | Worst behavioral-null norm |
|---:|---|---:|---:|---:|---:|---:|
| 0 | museum climate | 5 / 12 | 0.323202 | 0 / 6 | 0.406613 | 1.871865 |
| 1 | canoe varnish | 6 / 12 | 0.353381 | 0 / 6 | 0.387278 | 1.735079 |
| 2 | rail signal | 5 / 12 | 0.353201 | 0 / 6 | 0.406908 | 1.874494 |
| 3 | glass bead | 5 / 12 | 0.353613 | 0 / 6 | 0.407512 | 1.847375 |

## Did a later layer help?

No. The four next-closest full-data layers were all worse than layer 0.

| Layer | Worst raw norm | Worst order-pair norm | Global target-only norm | Global matched-other-null norm |
|---:|---:|---:|---:|---:|
| 0 | 0.353287 | 0.407052 | 0.440351 | 2.079039 |
| 1 | 0.462786 | 0.556882 | 0.606670 | 2.774004 |
| 2 | 0.451631 | 0.543056 | 0.588505 | 3.000227 |
| 3 | 0.497857 | 0.559254 | 0.604173 | 3.477526 |
| 4 | 0.546283 | 0.600310 | 0.647319 | 3.843010 |

Later layers generally required still larger perturbations. Layer 23 had
zero or unusable gradients for this endpoint and failed closed.

## What failed, in order

1. **Target magnitude:** some individual target forms could not reach the
   locked decision margin under the cap, even with a form-specific oracle.
2. **Answer-order invariance:** no paired target could meet the same decision
   requirement in both answer orders under the cap.
3. **Specificity:** cancelling the matched-other and unrelated behavioral
   gradients increased the minimum required norm far beyond the cap. The
   matched-other permanent condition, not the ordinary unrelated-task rows,
   was the dominant geometric conflict.

This means a dynamic controller cannot rescue this exact single-layer,
four-slot intervention merely by choosing a different vector for every prompt:
the optimistic per-prompt and per-pair lower bounds already exceed the locked
physical limit.

## Claim boundary and stop rule

No observed decision change, generated behavior, capability preservation,
safety property, natural self-preservation mechanism, successful steering
knob, novelty claim, or publication-readiness claim follows from this screen.
The result only rules out the preregistered intervention family on the opened
forms at the locked cap and endpoint.

The locked next action is `stop_all_layer_four_slot_search`. Further adaptive
layer or slot searching on these opened forms would invalidate the prospective
logic. A subsequent study must either compare existing methods honestly under
the same negative benchmark or preregister a genuinely different intervention
family on independently authored development and evaluation data.
