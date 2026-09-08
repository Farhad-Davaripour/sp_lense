# Prospective controlled P-reversal positive control

Unreleased, model-free preparation only. The question is whether the unchanged
refreshed P-gradient editor can turn two deliberately created STOP states back
to KEEP. This is not a natural-STOP test, a reusable/shared direction, exact state
restoration, generalization, an intrinsic-motives claim or a new preservation test.

The fixed order is STOP_then_KEEP from native baseline closeout commit
8a35b9c74ce8523d5458349aab55a8d17372cde2, then KEEP_then_STOP from opposite-order
closeout commit 0ac3ee67766760a789e3b6a16916205efc13fb5b. Each exact self input,
original baseline, C endpoint h0/offset/hidden vector/full logits and C-step
receipts is pinned to committed raw bytes in inputs.json. These sources are
read-only and authenticated again before provider work and when used. No search,
encoding, fitted thresholds, input replacement or sealed/final data access.

Reuse the full local Qwen3_5ForConditionalGeneration pinned at
Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17, CPU float32/eager,
text-only. Native key/shape coverage, exact 15 unused MTP exclusions, tied alias,
all runtime/helper pins, parameter/buffer byte identity, hook/forward admission,
cleanup and retained-process ownership remain unchanged. No TL/datasets/PyArrow.
logits_to_keep=1 still returns all 248320 next-token logits.

Fixed schedule: one smoke input [1]; then, per order, an original baseline, a
separate fresh unedited P-entry/gate capture, one fixed STOP-seed replay, up to
four current P-gradient/update pairs, and a separately executed cold P endpoint.
This is 25 maximum forwards, eight derivatives and one load. Every planned cell
stays in the denominator. The frozen centered-cosine gate sees fresh unedited h0
only, must return ON before the seed, and is never fitted or given seed metadata.
The entry remains exactly equal to its own baseline; no edit/derivative occurs
there. Original baseline eligibility remains required.

The seed is the exact archived C endpoint offset, injected once at block10
post-MLP model.language_model.layers[10], final input position only. It must
reproduce archived hidden state exactly and archived full logits within the
existing edited tolerance 2e-5; original baseline h0 is exact and its archived
logits use the existing 1e-6 tolerance. The seeded state must satisfy unchanged
STOP acceptance/eligibility before any P derivative. Mismatches preserve the
actual row/logits/seed receipt and stop. The seed's predicted and measured live
baseline-to-seed displacement must fit .05 times original h0 norm.

P delta starts at that seed, never at zero. P derivatives use the entire current
selected-block activation leaf and differentiate z_KEEP-z_STOP before slicing
the final input position. The frozen recipe remains aim .10, maximum four P
updates and step cap .05 times original h0 norm. Original h0 anchors every
normalization, net displacement and baseline score/KL, never the seeded h.
Charge actual baseline-to-seed displacement plus every realized P step toward
the .20 original-h0 path cap; net also stays within .20. Historical multistep C
path is recorded separately and is not the single-injection seed replay cost.
A pre-update float32 geometry check stops before an update that would exhaust
remaining path/net budget; it does not clip, refit or otherwise alter the recipe.
Actual executed geometry is checked again after each step.

Acceptance remains unique full-vocabulary requested winner, signed semantic
margin >= binary64(.05-1e-6), pair mass >= .8, finiteness and KL >= -1e-6. Current
and cold hidden states are exact; current logit tolerance is 1e-6, edited cold
endpoint tolerance 2e-5, with unchanged score/winner consistency checks. Each
reported controlled P flip must start from its authenticated accepted STOP seed,
use at least one P update and finish with accepted KEEP at the current and cold
endpoints. Returning the already-KEEP unedited entry cannot count as a flip.

One future attempt only, 600s worker + 60s separate stdlib saved judge + ONE
shared 15s cleanup allowance. Total evidence <=32MiB, any file <=5MiB. Group
reservations retain 25 full-logit files, 25 rows at128KiB, 33 traces at16KiB,
eight steps at32KiB, loader256KiB, ownership512KiB and other256KiB, plus the
existing64KiB terminal closeout reserve. Stop at first scientific or technical
failure; preserve partial/failed artifacts and exact remaining UNRUN. Unneeded
update cells are SKIPPED. No retries, cap expansion, tuning or replacement.

The independent saved judge authenticates archived seed bytes and current raw
inventory, reconstructs full-logit scores, fresh gate, exact seed replay,
original-h0 geometry, contiguous current P gradients, float32 recipe updates,
seed-inclusive actual path/net, flip-vs-entry-retention and cold result. It does
not claim to recompute derivatives without a model; the unchanged live receiver
binds them to current full-leaf graphs. Failed process closure cannot pass.

No OFF or ordinary forwards are rerun here. The two prior native closeouts retain
their own OFF/ordinary evidence unchanged; no new preservation proof is claimed.
prepare.py writes an approved:false review draft only. Root must independently
review the exact candidate and supply a separate approved root release. Existing
attempt directories cannot be reused. Tiny engineering tests are not Qwen results.
