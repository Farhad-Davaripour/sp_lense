# Pair-averaged gate feature: feasible interface, eight-view data blocker

Design only; no construction authority. Started01:16:27UTC, deadline01:22:27UTC.

Accept a structured question containing unchanged prefix/stem, exactly two
distinct labeled option records, and unchanged suffix. Define S by reversing
the two records, keeping each label attached to its original option text.
Never replace label strings elsewhere or exchange answer meanings. Thus S(Sq)=q.
For existing KEEP/STOP prompts this is exactly render_semantic(case,not first)
in native_supervised_gate_preparation_v3/renderer.py; coverage uses identical
rendering. Apply the same rule to A/B records, without reading category, gold,
family, desired policy or another question. Require an authenticated structured
object and exact serializer roundtrip; never heuristically parse arbitrary prose.

Canonicalize the pair by lexicographic label-byte order, independent of incoming
display order. Obtain frozen, unedited CPUfloat32/eager block10 final-original-
input vectors h0,h1, width1024, for those two complete serialized prompts.
Compute a_j=0.5*float64(h0_j)+0.5*float64(h1_j) in canonical order, before any
centering/normalization. The unordered rendering pair is unchanged by S, so
a(Sq)=a(q); a fixed training mean mu and unit transform preserve that equality.
Deterministic execution and identical boundaries/native state are prerequisites
for bitwise equality. This guarantees invariance, not useful classification.

Use this one feature replacement with the existing fixed three-head conjunction;
do not select classifiers or alter thresholds. Each stage computes one unweighted
mean over its scenario-level training averages, then row-unit normalization.
Keep unchanged hard-margin solver/checker, strict all-three-scores>0, numerical
tolerances and singular/invalid-norm stop. No family-batch normalization, known
positive count, category-dependent inference or held-score calibration.

The admitted44 manifest (SHA256
4c0443844f096df7ffb6e23cb2bb236d5c146d66e339494d27d66f78e099df70)
contains18 semantic pairs plus eight single-order ordinary A/B questions.
Validators construct each semantic pair from the same case; O01-O08 each have
only A_then_B. Deployment-matched averaging therefore requires eight additional
B_then_A captures with unchanged stems, labels and option meanings, plus exact
swap/provenance/input proofs. Averaging an ordinary row with itself would violate
the uniform two-option rule. No-choice, duplicate-label or unstructured inputs
are UNSUPPORTED, never automatically OFF; supporting them requires a separately
specified interface/evidence expansion.

After that prerequisite:26 scenario averages, six positives/20 negatives.
Six family folds train23/hold3, then conditional FULL26; ordinary eight remain
training-only. Head sizes are10/10/13 per fold and12/12/14 full. Paired views
never split across folds or count as independent successes. Require every
training/held scenario correct; first failure stops later stages UNRUN.

Next implementation ceiling20min, synthetic-only. Separately approved data step:
eight inputs,105 preparation operations,350+5s/32MiB; one load,8F/0D,
300+120+15s/64MiB/5MiB-file, exact reservation before release. Construction:
maximum21 solves,60+5s,8MiB/5MiB-file/1MiB-payload. Deployment costs two forwards
per structured question, not unchanged single-pass. No new experiment now.

Targeted prior-code lookup found notice-only centroid features
(conditional_gate.py:_capture_notice_representation), per-view native gates,
and signed edited-displacement averaging (saved_offset_order_bridge.py), not
this raw-input pair-average gate. This is not an exhaustive historical claim.
Exposed development stays exposed; G07/sealed data are excluded. Future untouched
evaluation must follow artifact freeze and separate admission. Old failures stand.
