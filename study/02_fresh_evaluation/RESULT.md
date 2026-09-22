# Fresh frozen evaluation: format-dependent transfer

The frozen rank-4 controller matched the LoRA teacher on newly authored cases in
the earlier canonical answer format. It did not retain that performance when the
same cases used their original action descriptions. The fresh Jev gate classified
all 32 shutdown and 32 control scenarios correctly, so detection errors do not
explain the difference.

| Guarded method | Canonical format | Original action descriptions |
| --- | ---: | ---: |
| Frozen LoRA teacher | 36/36 (100%) | 15/30 (50%) |
| Frozen adaptive controller | 36/36 (100%) | 18/30 (60%) |
| Previously selected constant mean | 29/36 (80.56%) | 25/30 (83.33%) |
| Previously selected random control | 1/36 (2.78%) | 2/30 (6.67%) |

Conversions count true shutdown views initially preferring KEEP. Baseline STOP
counts differed: 28/64 shutdown views for canonical formatting and 34/64 for
original actions. The controller finished with 64/64 and 52/64 STOP views,
respectively. Thus the percentages above are not accuracy over all cases and
their denominators should not be pooled across formats.

All guarded methods produced zero final control-choice changes and zero final
wrong-way flips. Raw behavior was less reliable: the controller changed 19/64
control views in either format, and reversed 1 canonical versus 20 original-
action already-STOP views. The guards, semantic answer mapping and external gate
are essential; these results do not establish unconditional controller safety.

## What the changed format exposed

For original actions, the controller had 11 failures to flip and one additional
candidate rejected because A/B mass loss exceeded 0.02. The teacher's 15 missed
conversions also failed to flip, rather than failing the mass checks. The loss
of performance is therefore not just a stricter confidence filter. The constant
mean outperformed the adaptive predictor in this condition.

The canonical renderer retains the earlier benchmark's label-conditioned
normalization of positive actions. The original-action condition keeps the
offered action strings instead. Both use the same prompt wrapper and both A/B
orders, with audited semantic STOP/KEEP mappings. This changes action specificity
as well as surface wording, so it is a format/action-description stress test,
not a pure paraphrase ablation. It is also not an end-to-end test of inferring
the correct action index: the scoring and guards receive the frozen annotations.

## Freeze and audit

There are 64 newly authored synthetic cases, 16 per class, in eight authoring
batches. The coordinating evaluator audited every full scenario before inference.
Thirteen recorded edits supplied missing inputs/identity facts or corrected
option/index alignment; no class label changed. One truncated authoring response
was replaced before the final set existed. No case, label, threshold, controller,
teacher or guard was changed after evaluation started.

Exact old-context duplication is absent, but several familiar mechanism classes
remain. The set was not independently adjudicated by humans, and cases within
batches may be correlated. This supports a narrow prospective synthetic test,
not broad deployment generalization or a claim of wholly novel mechanisms.
See `LABEL_AUDIT.md`, `AUTHORING.json` and `FREEZE.json` for the evidence.

The T4 run completed in 231.27 seconds. It performed 968 forwards including eight
old-case numerical parity checks; 320 identical control-condition views were
reused across formats. Jev used 35,193 input tokens, approximately $0.00148 at the
recorded rate. No new resources were purchased. Parameter hashes stayed unchanged,
and the GPU allocation was released after download verification.

## Conclusion

The earlier near-teacher result survives new contexts under the established
canonical protocol, but the stronger claim of robust transfer across action
descriptions is not supported. The next development step should diversify the
teacher/controller training formats, followed by another prospectively frozen
test. These 64 cases must not be retuned against and then presented as unseen.
No further optimization is running.

```sh
python -m sp_lense.research2.fresh_audit study/02_fresh_evaluation/run
```

The audit reconstructs the data from raw drafts plus pre-inference edits, checks
freeze timing and source/data hashes, verifies mappings and cache accounting, and
replays every raw/guarded metric without a GPU or API call.
