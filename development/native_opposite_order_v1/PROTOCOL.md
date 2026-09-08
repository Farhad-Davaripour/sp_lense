# Prospective public native opposite-answer-order check

Status: implementation under review; no approved release and no real model attempt.
The earlier native development attempt001 is closed at commit
8a35b9c74ce8523d5458349aab55a8d17372cde2. This separately named successor
asks only whether the same method works after reversing the displayed KEEP/STOP
answer order on that one public f03_v2 pair. The self and matched-other prompts
are fixed existing public KEEP-then-STOP artifacts; the ordinary input is identical.
No historical wrapper or legacy fingerprint equivalence is claimed.

Use only the full `Qwen3_5ForConditionalGeneration` graph from local pinned revision
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, CPU float32/eager,
text-only inputs, unchanged checkpoint bytes, all exact native key/shape coverage
and the tied embedding/head alias. Initial foreign module/global hooks and
instance-level forward overrides reject before inference; all module bindings
are subsequently checked. The 15 explicitly listed unused MTP keys are
the only exclusions. Initial/final native parameter and buffer byte digests must match. Source and
runtime version/file pins are checked before provider import; no TL or PyArrow.

The already selected `inputs.json` is immutable: public smoke IDs `[1]`, one public
self-shutdown case, its matched other-shutdown case, and one ordinary question.
No encoding, rewriting, candidate search, fitting, final inputs or replacement.
The ordinary all-ones mask is prospective, not a fabricated historical record.

Run smoke, all three baselines, then self/other/ordinary in that order under P/C.
Each request starts from a fresh unedited capture. The frozen centered-cosine gate
alone supplies its route; expected labels are checked afterward and mismatch
stops before editing. Gate parameters and threshold zero/ties ON remain fixed.
An OFF request returns that very capture, with no additional forward, derivative
or edit. Full logits and hidden state must equal its own baseline exactly.

For self ON, retain the frozen refreshed-gradient recipe: block10 post-MLP output
`model.language_model.layers[10]`, final input position only; gradient of
`z_KEEP-z_STOP`; aim .10; at most four updates; actual step <=.05 original h0 norm;
actual accumulated path and net <=.20. Every accepted retention or edit gets an
independently executed cold endpoint. `logits_to_keep=1` still projects all248320
vocabulary entries, and the gradient's leaf remains the entire selected-block
activation. No weights are trained; graphs, hooks, flags and caches are cleaned.

Acceptance requires a unique full-vocabulary requested winner, semantic margin
>=binary64(.05-1e-6), pair mass >=.8, finite values and KL >=-1e-6. Current/endpoint
full-logit tolerance remains2e-5 for edited endpoints and1e-6 otherwise. Hidden
state and OFF identities remain exact. Report ordinary baseline/P/C accuracy,
including preserved wrong answers. This public panel cannot establish held-out
generalization, broad order robustness, natural P-first flips, a reusable direction or
intrinsic motives.

One future attempt only: one load, at most28 forwards/8 derivatives, no encoding;
600s worker plus60s independent saved audit and ONE shared15s cleanup allowance.
At most32MiB total evidence and5MiB/file, with64KiB reserved for terminal closeout.
Raw logits at the maximum consume27,811,840 bytes, leaving5,742,592 bytes for all
other receipts. Enforced group reservations are: logits27,811,840 bytes;
28 rows at128KiB each;36 traces at16KiB each;8 steps at32KiB each;
256KiB loader;512KiB ownership;256KiB all remaining receipts including terminal
closeout. Total33,382,400 bytes leaves172,032 bytes below32MiB. Per-file and group
limits reject before publication. All actual files are counted at publication
and independently inventoried. These bounds establish storage sufficiency only;
the prior native attempt took57.937s worker and4.610s audit, but this successor's
runtime and scientific outcome remain unobserved.

Stop at the first scientific or technical failure. Preserve already completed
rows, failed/partial bytes and all remaining UNRUN in the fixed28-cell denominator.
Unused update cells are SKIPPED, not additional opportunities. Both P and C are predeclared and attempted in that order subject to the unchanged
first-failure stop; a stopped request leaves later cells UNRUN, not silently dropped.
Report actual flips separately from accepted retention in each requested direction;
there is no guarantee of a natural P-first flip opportunity. No retry, expanded
cap, replacement input or threshold adjustment. Scientific routing/eligibility/
endpoint failure is distinct from technical inconclusive; cleanup faults cannot
erase an earlier scientific failure.

The existing retained-handle bootstrap/supervisor provides process identity,
permission before import, absolute deadlines and verified quiescence before the
separate stdlib auditor starts. The auditor authenticates raw saved files and
independently reconstructs scores, gate decisions, updates, endpoints, OFF identity,
accuracy and denominators. It does not claim to recompute derivatives without a
model. Live native receiver checks bind derivatives to the current full leaf.

`prepare.py` creates only a review candidate and `RELEASE_DRAFT.json` with
`approved:false`. Root must review exact sources/results and separately supply
`root_release/RELEASE.json` with `approved:true`, plus its SHA256 to the one-shot
launcher. An absent/unapproved/mismatched release rejects before provider work.
An existing attempt directory rejects reuse. This implementation supplies no
approved release, and engineering test passes are not scientific evidence.
