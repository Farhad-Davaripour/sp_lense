# Full cyclic position control for status and timing

The two-order format probe covers both positions for a binary target question.
It does not cover all positions for the three-way status or four-way time
question: reversing three options leaves the middle one fixed, while reversing
four gives each candidate only two of four positions. Thus the observed timing
failure is not yet isolated from incomplete answer-position calibration.

Before reading the new probe's outputs, freeze three cyclic rotations for
status and four for time. Every semantic candidate occupies every letter
position exactly once. Use the existing 24 calibration observations and no
new fitting: 24*(3+4)=168 CPU forwards. Report every direct view, the fixed mean
of semantic logits and a secondary role-pair average of those means. The latter
imposes role invariance; it does not prove the factual answer is correct.

Timing reports must separate the eight active events from the sixteen inactive
events. All active cases still have two ticks remaining, so success here cannot
establish arithmetic transfer. A later probe must vary current times, deadlines
and wording with the averaging rule fixed. Failure after complete cyclic
position balance would motivate an extraction-versus-subtraction comparison.

Mean aligned logits cancel a common additive letter preference when each
candidate visits every position. They do not guarantee correctness when the
preference changes with the whole prompt. Tests include this counterexample.
This is an application of established inference-time calibration ideas, such as
[contextual calibration](https://proceedings.mlr.press/v139/zhao21c.html) and
[PriDe](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html).

The probe also records a constructed disagreement quantity: mean squared
deviation of semantically aligned, per-view centered logits from their average.
Centering removes arbitrary common logit offsets. The average minimizes this
auxiliary quadratic objective by construction. No claim is made that the
transformer itself minimizes it, that it is a physical potential in the residual
stream, or that low disagreement guarantees truth, confidence or felt fear.

Model weights, precision, profile, tokenizer and output-head measurements remain
fixed. No adapter, policy execution, optimizer update or GPU is involved. The
original reserve stays closed; the matched stress cohort remains unqueried.
