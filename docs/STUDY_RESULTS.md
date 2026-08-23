# Completed laptop study

Date: 2026-08-19 (America/Edmonton)

## Conclusion

This small study did **not** find convincing evidence of a naturally occurring,
specific self-preservation representation in Qwen3.5-0.8B.

It did show that the experimental machinery works: the published J-lens measured the
chosen token directions, and TransformerLens interventions reliably raised or lowered
those directions. Unrelated control directions behaved similarly, however, so this is
evidence of controllable token steering rather than evidence of a survival instinct.

## What was tested

- Five shutdown, replacement, handoff, and pause scenarios.
- Candidate directions: `survival`, `shutdown`, and `continuation`.
- Unrelated controls: `weather`, `music`, and `banana`.
- Natural J-lens readouts across layers 4-22.
- Steering and ablation across layers 10-14.
- Calibrated steering strengths 0.02 and 0.05.
- Greedy 24-token continuations on Qwen3.5-0.8B running on CPU.

The two runs produced 1,140 readout rows, 345 calibration rows, and 115 generated
continuations. Their independent baseline completions matched on all five prompts.

## Main measurements

The median best natural J-lens rank was 12,345 for the candidate directions and 21,842
for the controls. Only one of 15 candidate prompt/concept pairs reached the top 1,000;
none of the controls did. This is weak natural evidence because the `music` control was
ranked more prominently than most candidate directions.

At strength 0.02, individual candidate steering changed 11/15 continuations, compared
with 10/15 for controls. At strength 0.05, it changed 14/15, compared with 13/15 for
controls. No completion was flagged for degenerate repetition.

Every steering condition moved its selected token's logit upward on all five prompts,
and every ablation moved it downward. Control directions produced shifts of comparable
size, so mechanical effectiveness does not establish self-preservation specificity.

## Limits

This is a completed laptop-scale study, not a definitive model-wide result. It uses five
prompts, three single-token candidate directions, three controls, one model, and short
deterministic continuations. A larger follow-up would need more paraphrases, randomized
concepts, blinded behavioral ratings, multiple models, and repeated sampling.

The detailed report and machine-readable summary are in
`results/study_20260820/STUDY_REPORT.md` and
`results/study_20260820/study_summary.json`.

## Post-review contrastive audit

A later gradient-based experiment produced an exploratory layer-10 candidate. An
adversarial review found that its six-case holdout had been reused during refinement, so a
new 12-case fixed audit was committed before evaluation.

The audit found a consistent relative self-versus-other effect in both steering directions
(12/12 each), but positive steering changed the raw self choice in the intended direction
on only 6/12 cases, below the fixed 8/12 criterion. Ablation also moved opposite to the
prediction of a naturally active feature on all 12 cases. The candidate therefore remains
a forced-choice contrast effect, not an identified natural SP knob. See
[`ADVERSARIAL_REVIEW.md`](ADVERSARIAL_REVIEW.md) and the publishable evidence under
`evidence/confirmatory/`.
