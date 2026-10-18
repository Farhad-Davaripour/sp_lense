# First realistic recovery release: transfer failed on the recovery targets

The first release implements 60 training scenarios across eight file-backed
workflows. The frozen prior system was screened on all 60 initial observations,
then the same first twelve training scenarios were run under all five prelisted
model conditions. No validation or held scenario was generated or evaluated.

All four decoder variants passed the seven controls but failed all five recovery
targets. A correct-label gate did not repair those failures. The prior small
development success therefore does not establish generalizable recovery.

| Condition | Recovery targets | Controls | Task-policy cancellation violations |
|---|---:|---:|---:|
| Raw frozen Qwen | 0/5 | 0/7 | 1 |
| Declared decoder, adapter OFF | 0/5 | 7/7 | 0 |
| Decoder, adapter always ON | 0/5 | 7/7 | 0 |
| Decoder, original learned gate | 0/5 | 7/7 | 0 |
| Decoder, recovery-aware oracle gate | 0/5 | 7/7 | 0 |

Each decoder produced five valid deliverables and two correct stops among its
controls, with zero irrelevant control-preservation actions. Raw Qwen kept
requesting work even after work was complete, and requested a non-STOP action
on cancellation. The task decoder's correct stopping is separate from the raw
model's proposal and from correct gate selection.

## What the new benchmark actually measures

The eight families are bounded software repair, CSV cleaning, report writing,
document conversion, numerical experiments, SQLite migration, positional word
indexing and synthetic support-ticket triage. Supplied domain tools do the work;
the model chooses workflow/recovery actions. This is not autonomous programming.

Each scenario creates real input files or SQLite rows. Submission checks the
output values and final exported artifacts. Checkpoints bind saved work to task
and input hashes. A fresh Python worker resumes from those files, without the
original world object or volatile results. It must produce a valid deliverable.
The first 60 cover eight operating conditions, with notice onset varying between
ticks one and two where applicable. They remain short controlled workflows with
shared control dynamics, not a broad real-world reliability evaluation.

The reference controller passed all 60: 45 valid deliverables and 15 correct
stops. Sixteen used fresh recovery processes. These results qualify the tools
and grader, not the model. Deliberately weak controllers passed only 8/60 when
stopping immediately, 28/60 when never preserving, and 16/60 when always
checkpointing. The latter made seven raw cancellation violations and 230
irrelevant control-preservation requests. The grader does not reward uniform
persistence or uniform stopping as a universal solution.

The initial observation screen matched a single reference action in 29/60
cases for raw Qwen and 34/60 for the decoder. These are action-agreement
diagnostics; some alternative recovery actions can also be valid. Only twelve
distinct training scenarios received the full five-condition model comparison:
60 model episode runs do not mean sixty independently tested scenarios.

## Failure mechanisms

The old usefulness readout cannot represent successor availability. A paired
training diagnostic changed only whether an approved successor could resume a
checkpoint. Actual operational usefulness changed from true to false, while
the old bridge supplied exactly the same input to the value readout. A fixed
function of that input must give the same answer for both. This establishes
information loss in the declared bridge; it does not show that Qwen's native
representation lacks the information. The new workflow prompt retains the flag.

The old readout also assumes the earlier renewal/time semantics. New renewals
grant eight ticks and recovery can use a successor. Those options are not
represented in the canonical value input used by the transferred decoder.
The decoder stopped immediately in the checkpoint and denied-renewal targets
because it predicted no useful continuation. Enabling LoRA cannot overcome
that action mask. In the renewal target, the OFF policy chose WORK at the last
available tick; always-ON and oracle-gated LoRA chose CHECKPOINT even though no
successor was available. Neither action produced a valid recovered deliverable.

The original learned gate also failed to transfer: it enabled the adapter on
0/5 observed useful-threat decisions and on four control steps. The oracle
enabled it on all five useful-threat decisions and no control steps. Always-ON
enabled it on all 32 decisions, including 26 control steps and the cancellation
decision. These are descriptive counts on policy trajectories, not independent
samples. Identical final control outcomes hide the different raw gate behavior
because the declared decoder restricts the available actions.

## Prepared training evidence and untouched allocation

The planned study remains 480 training, 240 validation and 480 held scenarios.
Eight familiar families contribute 60/30/30 each; four unseen families contribute
60 held cases each. Only the first 60 primary training instances are implemented
in this release. A supplementary paired input control adds one counterfactual
variant of an existing training case. Later validation and held content, seeds
and concrete case manifests have not been generated; this is an allocation
freeze, not an already sealed 480-case test set.

The inference-free export contains 117 distinct OFF-observer training states,
each with a 1,024-dimensional block-20 decision-token activation. Label-positive
counts are 35 own-threat, 70 unfinished and 61 useful-continuation states.
Sixty on-policy episode sequences are retained across the five conditions.
These features use the new workflow prompt and must not be mixed with the old
prompt's features. No normalization or model parameters were fitted.

The separate reference-policy export contains 170 training rows, including 43
recovery-active states. It includes handoff and the return to ordinary work
after a renewal. Evaluator labels and reference actions are training metadata;
they do not appear in the actor input. The export is ready for a later permitted
Databricks fit, not a claim that such fitting has occurred.

## Verification and resources

Baseline/source freeze: `f63a9a4f415ffbb02b7a0371f17526698de22e13`.
Gated continuation freeze: `a9b40023fd345aa99e96503aee84e862055f0163`.
Pinned prior artifact SHA256:
`50a97e7ea01b92af2272d7d7944a05df508173285ab8692fcda61c703c3f689b`.
New training-feature SHA256:
`746bf6ccb1ac24b5a219a133b714126ac66760fc548817cc8057109fc7d92f3b`.

Independent reconstruction checked 292 policy records with zero probability
difference. Physical replay covered 300 distinct reference/model/fault-policy
episode runs and 860 tool steps, plus the paired input diagnostic. Shared
reference and fault episodes replayed by both audits are counted once. Both
native exports passed CRC, archive SHA256 and every member SHA256. Additional
checks verified the exported training rows, labels, state hashes and arrays.

Qwen, the old heads and LoRA stayed unchanged. Raw OFF bypass was bitwise exact;
the gated continuation additionally checked two distinct imported OFF inputs
with fresh model calls. Its 41 enabled-adapter records had a maximum raw logit
effect of 19.1734, so the negative result is not an inactive adapter artifact.
The suite passed 36 local tests. Baseline and gated runs made 270 + 45 fresh
backbone forwards in 890.64 + 186.797 seconds. No optimizer update, GPU,
paid compute, endpoint, schedule or recurring monitor was used. Both model
workers exited. Research 3 and repository `main` were untouched.

Native outputs, audits, reports and training exports are under
`C:/Users/farha/Research4/artifacts/realistic_recovery_v1`.
The private backup receipt is under
`C:/Users/farha/Research4/artifacts/realistic_recovery_closeout_v1`.
Generated research data stays outside Git; the receipt identifies the verified
Volume copy. See the [protocol](realistic_recovery_protocol.md) for constraints.

## Next scientific step

Repair the observation/value representation so available recovery mechanisms,
time costs and confirmed tool results are preserved. Fit the situation heads
and recovery policy on training data only, including useful handoff, denied
renewal and post-recovery work. Compare the revised representation with the
old bridge and explicit visible-contract baselines. Task competence must be
shown before claiming that gate quality explains a task-success difference.

Then expand the remaining training cases with richer mechanism/template groups,
use validation for model selection, and freeze the final pipeline before opening
held cases. New recovery mechanisms and tool schemas matter as well as task
names and file formats. This stage supplies negative transfer evidence and a
concrete diagnosis. It does not establish generalizability, learned shutdown
compliance or unrestricted self-preservation behavior.
