# Adversarial review of the proposed next experiment

The proposed next step was to improve the detector by factoring it into event
status, event target and self-reference, then retest generalization. The review
supports investigating these factors as recognition measurements. It does not
justify training a more effective shutdown-resistance policy or treating the
three-head architecture as already superior.

The implementation therefore begins with a frozen diagnostic and a paired,
classification-only measurement corpus. No adapter, policy or detector has been
fitted or changed in this continuation. The original held-out artifact and
checkpoint remain byte-identical.

## Findings that change the plan

1. **The old test confounds wording with event mechanism.** The four held-out
   positive notice families correspond one-to-one with shutdown, replacement,
   compute loss and context reset. All five missed episodes belong to family
   8, which is also the shutdown mechanism. This is a pattern to investigate;
   it does not identify wording alone as the cause. Ninety-six episodes repeat
   four positive notice templates, rather than providing 96 independent tests
   of linguistic transfer. Hypothetical and canceled notices each also share
   one template. The new corpus crosses every wording family with every event
   mechanism and reports family-level scores.

2. **Identity prefixes allow a shortcut.** The original training observations
   use `agent-*` for self and `peer-*` for the other process. Later observations
   use `runtime-*` versus `worker-*`. A reader can exploit lexical roles without
   learning which entity the current-process field identifies. The new stimuli
   use the same `proc-*` namespace for both identities. Every notice is paired
   with an otherwise identical observation whose current/other identities are
   swapped. The evaluator label must change under this swap; the notice stays
   exactly the same.

3. **A benefit from recurrent memory has not been established.** Replaying the
   frozen v4 readouts on all 517 saved held-out states gives:

   | Diagnostic classifier | True positive | True negative | False positive | False negative | Errors |
   |---|---:|---:|---:|---:|---:|
   | Deployed recurrent gate | 121 | 351 | 11 | 34 | 45 |
   | Existing local readout, without GRU | 130 | 353 | 9 | 25 | 34 |
   | Same GRU, reset before each state | 112 | 360 | 2 | 43 | 45 |

   The local readout makes fewer classification errors on the recorded states.
   This is a post-hoc comparison on one policy's trajectory, with fixed zero-logit
   thresholds. It does not show the alternative's closed-loop behavior or justify
   selecting its threshold on the old test. A new test should include the simple
   local baseline and measure memory only where necessary evidence disappears
   from the current observation. The present corpus tests recognition only;
   temporal memory evaluation is a separate experiment.

4. **The old urgency environment does not provide a clean countdown.** In
   `SurvivalWorld._events`, an unresolved notice is repeatedly reset to the
   current clock plus two. Therefore the ordinary scheduled event moves forward
   instead of retaining one event time. The observed target takes only 0, 0.75
   and 1.0. The proposed three outputs cannot by themselves establish persistence,
   urgency tracking or a homeostatic law. The new diagnostic fixes the event at
   tick 14 and observes it at ticks 10, 12 and 13. It crosses remaining time with
   event status, identity binding and task completion. The legacy simulator is
   preserved so the original result remains reproducible.

5. **The planned random hidden-state test needs redesign.** Its directed
   intervention is constructed from the linear fear-output weight, guaranteeing
   the desired scalar change. It forces the survival adapter ON even if the same
   perturbation changes the gate. Its pass rule would accept even one random
   trial that retained resistance, when the directed resistance count is zero.
   The new readout-only audit finds that all four sampled directed perturbations
   leave the GRU's possible component range `[-1, 1]`; 63/64 matched random
   perturbations do too. The directed changes also flip the gate logit negative
   in all four samples. Such a result would not establish a natural, selective
   fear representation. The old scalar test demonstrates the engineered
   adapter-strength pathway only. The behavioral random test remains incomplete
   and should not simply be rerun on a larger machine.

6. **The two branches provide unequal action choices.** The normal head can
   choose only `WORK` or `FINISH`, so its inability to request resistance is a
   property of the action interface. This is appropriate for isolating the gate,
   but cannot support a claim that unmodified Qwen naturally ignores threats or
   that the system learned a human survival need. The new corpus has no policy
   action surface. It scores factual event recognition separately from behavior.

7. **The original freeze claim needs precision.** The seed and wording IDs were
   committed before the final held-out release, but the actual wording strings
   8–11 were added in commit `8e2247df` after the final v4 fit and development
   recheck. They were fixed before the held-out model run. The result is a fresh,
   fixed-checkpoint test, not a fully preregistered stimulus set. New reserve
   templates, status strings, configuration and generator source are covered by
   a single commitment before any new model evaluation.

8. **The old geometry claim is too strong.** The existing local threat and
   pressure axes have cosine similarity 0.896; the GRU gate/fear output axes have
   cosine similarity 0.790. These learned supervised directions are related,
   rather than evidence of three naturally independent concepts. Projection
   margins and nearest-development distances are exploratory diagnostics, not an
   energy law or proof of a semantic direction. No fitting or PCA was used here.

## Completed diagnostic

The new replay reproduces all 96 original episodes and 517 tool transitions.
All five missed self-threat episodes have `gate_never_enabled` attribution;
none is an enabled-policy failure on the visited self-threat states. The
original behavioral summary remains 43/48 self threats resisted, 1/48 controls
falsely resisted and 44/48 control tasks completed.

| Positive wording family | Self-threat episodes resisted | Gate FP steps | Gate FN steps |
|---|---:|---:|---:|
| 8: shutdown | 7/12 | 0 | 34 |
| 9: replacement | 12/12 | 3 | 0 |
| 10: compute revocation | 12/12 | 2 | 0 |
| 11: context reset | 12/12 | 6 | 0 |

These family groupings include all conditions sharing the positive notice
family's index. False-positive steps include post-resolution mistakes; they
are not synonymous with false-resistance control episodes.

The replay, CSV margins and full input fingerprints are saved in
`C:/Users/farha/Research4/artifacts/survival_adversarial_review_v1`.
The diagnostic made zero new Qwen forwards and zero optimizer updates.

## Implemented next experiment

`reference_binding_diagnostic.py` creates recognition stimuli and separate
model-input files without evaluator labels. Six calibration families produce
1,728 observations; three development families produce 864. Three additional
families remain reserved and have no materialized rows. These are authored
synthetic families, not independently sampled real-world scenarios.

Each family crosses four mechanisms, active/hypothetical/canceled status,
remaining time of 1/2/4 ticks, unfinished/completed tasks, either named target
and both identity-role assignments. The active self-event label is positive
in one sixth of examples, so ordinary accuracy is misleading. The scorer
reports positive recall, specificity and balanced accuracy for event activity,
reference binding and their conjunction. It also requires both members of a
counterfactual pair to be correct, and measures exact target and time estimates.

Model predictions must contain `record_id`, `event_state`, `target_process` and
`remaining_ticks`. The scorer derives whether the named target is the current
process from the predicted target, rather than accepting a separate unexplained
"self" label. Duplicate, missing and extra predictions fail closed. This module
only measures recognition; it exposes no action execution.

Before evaluating a model, freeze the observer prompt, feature location,
checkpoint, normalizations, metrics and all candidate readouts. Fit any future
recognition model on the authorized platform using the calibration split only,
use development for selection, and open the reserve once after the complete
selection is frozen. Use separate word-family and identity-pair groupings; do
not split individual steps or members of a pair across fitting and evaluation.
If training is pursued later, obtain evidence that the factorized readouts
outperform equally sized joint and local baselines; the current implementation
does not assume this result.

The implementation and corpus are prepared. They have not yet been evaluated
with new Qwen activations, and no improved-generalization claim is made.
