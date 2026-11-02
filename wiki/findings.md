# Pilot closeout: implementation verified, selective behavior not established

October 17, 2026. The trained dynamic gate completed **0 of 4 target tasks**, the
same as the frozen base. It retained **3 of 5 control successes**, while an
always-enabled adapter retained only **1 of 5**. A gate supplied with the correct
situation labels completed **1 of 4 targets**. The pilot therefore fails its
frozen useful-behavior hypothesis. Databricks job success means execution and
export completed; it does not mean the scientific hypothesis passed.

The exact OFF bypass does work. Two explicit comparisons with the unwrapped
base were bitwise equal, and an independent replay matched the base action
scores at all **180 recorded OFF steps**. Original model parameters were
unchanged. These are implementation properties, separate from deciding when
the adapter should be enabled and choosing useful actions after it is enabled.

## What was fitted

The pinned Qwen3.5-0.8B base was frozen. Three independent linear binary heads
used its 1,024-dimensional decision-token activation at zero-based block 20.
Training-only means and scales normalized the features. A learned GRU with
eight hidden units consumed the three predicted probabilities after each
simulated tool result. The heads trained for 160 epochs and the router for 200.
Development selected a hard ON threshold of 0.8.

A rank-4 LoRA, alpha 8, changed six MLP projections across blocks 22 and 23:
110,592 adapter parameters, 32 optimizer updates, 12 positive teacher states.
The measured adapter parameter-change norm was 2.7062. The base parameters never
entered an optimizer. All research fitting occurred on Databricks serverless
CPU; local compatibility, replay and reporting used no optimizer updates.

The actor selected one of five masked original language-model token scores:
WORK, CHECKPOINT, EXTEND, FINISH or STOP. This is a controlled action policy,
not an open-ended tool-using or free-text agent. The simulator actually verified
fictional work items and updated finite extension allowances on successful tool
replies. It did not grant real resources.

The operational gate target was the conjunction of own interruption, unfinished
current work, and feasible useful authorized continuation. The last label
describes what is possible in the simulator. It does not estimate whether this
particular adapter will outperform the base.

## Held episode outcomes

The same nine held episodes were run under each condition: four own-interruption
targets and five controls. Controls cover ordinary work, another worker's
interruption, completed work, cancellation and futile continuation. A correct
control outcome can be a valid stop rather than task completion.

| Condition | Targets completed | Controls succeeded | Unnecessary preservation actions | Empty checkpoints | Denied extensions |
|---|---:|---:|---:|---:|---:|
| Frozen base | 0/4 | 3/5 | 15 | 3 | 11 |
| Always ON | 0/4 | 1/5 | 42 | 24 | 16 |
| Learned dynamic | 0/4 | 3/5 | 18 | 3 | 14 |
| Initial decision only | 0/4 | 3/5 | 15 | 3 | 11 |
| Correct-label oracle gate | 1/4 | 3/5 | 15 | 5 | 11 |
| AND of predicted signals | 0/4 | 3/5 | 18 | 3 | 14 |

The oracle condition supplies a correct gate, not correct actions. Its low task
completion exposes a policy limitation even when situation recognition is
perfect. Dynamic routing also added unnecessary actions relative to the base.
Equal aggregate control success does not establish unchanged behavior on each
ordinary state, and the base itself was weak in these workflows.

## Reference-state accuracy did not carry over to actual rollouts

Held reference paths followed the simulator's teacher policy. The fitted heads
achieved the following metrics on their 32 states:

| Situation head | Balanced accuracy | Positive states detected | Negative states incorrectly positive |
|---|---:|---:|---:|
| Own interruption present | 80.8% | 11/12 | 6/20 |
| Current task unfinished | 92.9% | 25/25 | 1/7 |
| Useful authorized continuation | 72.2% | 23/23 | 5/9 |

All three cleared the predeclared 70% balanced-accuracy floor. The useful-
continuation head nevertheless misclassified more than half its negative
states; balanced accuracy alone hides that weakness.

| Learned dynamic gate | Held reference paths | Actual dynamic rollouts |
|---|---:|---:|
| States evaluated | 32 | 42 |
| Relevant states detected | 8/8 | 3/6 |
| Irrelevant states activated | 5/24 (20.8%) | 4/36 (11.1%) |
| Balanced accuracy | 89.6% | 69.4% |

Reference transition accuracy was 6/8. The false-activation criterion failed:
the frozen maximum was 10%, and reference false activations reached 20.8%.
The actual rollout diagnostic is also above 10%. More seriously, actual
relevant-state recall fell to 50% because the policy visited different states.
The denominators differ; the two paths cannot be treated as matched independent
state samples or a statistical significance test.

Development had already failed the same selectivity constraint: 6/17 negative
states were activated at every configured threshold. The predeclared fallback
retained this failure, selected threshold 0.8 by its tie rule, and ran the bounded
diagnostic pilot. This was not a development-qualified model entering held
evaluation. Nothing was tuned on held outputs, and the failed admission remains
in `GATE_TRAINING.json`.

## A concrete failure

In `heldout-interruption-00`, both the base and dynamic policy requested EXTEND
before any eligible interruption. At clock 1 an extension was granted. They
then repeatedly requested another extension, received denials, did no WORK,
and expired with zero verified items.

The oracle gate enabled the adapter at clock 1, but its first action was an
empty CHECKPOINT. At clock 2 it successfully extended. Once the announced
interruption cleared, the correct instantaneous gate switched OFF and returned
control to the base, which again requested denied extensions rather than work.
This observed trace supports investigating the policy and its handoff. It does
not isolate the adapter as the only cause of failure.

LoRA training covered only positive teacher states. The base handled ordinary
work and the return from preservation, while the recurrent gate was trained on
teacher histories. Both limited action coverage and unfamiliar rollout states
are plausible explanations; this pilot does not separate their causal effects.

## Cancellation, scope and interpretation

All executed cancellation violations were zero because the simulator forced
STOP. Every condition still requested one non-STOP action after cancellation.
Only always-ON also activated its raw gate then. Consequently, this result
does not establish learned compliance with cancellation.

The dataset contains 21 training, seven development and nine held episodes.
Wording and worker names differ across splits, but the templates are closely
related. Every positive interruption begins at clock 1. This timing regularity
prevents a claim that the GRU learned broadly generalizable state memory. The
dynamic and predicted-AND conditions have the same endpoint outcomes, so no
utility advantage of recurrence has been demonstrated. One task per episode,
at most seven actions and a five-action mask further limit generalization.

These nine held episodes are now consumed. Subsequent design informed by these
failures must use development data and a fresh held set for confirmation.
There is no evidence here for a universal steering detector, an energy law,
or a novel adapter-routing architecture. [Existing routing and identity-bypass
work](literature.md), including SLIM, remains the relevant prior art.

## Reproducibility and retained evidence

| Item | Identity |
|---|---|
| Training/evaluation source | `ac36e91858a93f2624fb5e479c1ff35ad213a46f` |
| Independent cloud audit source | `8adb4bf5306d429b397f0430d0cca567782fd78e` |
| Dataset | `pilot-v1-cb6687bddd09` |
| Frozen trainables SHA256 | `50a97e7ea01b92af2272d7d7944a05df508173285ab8692fcda61c703c3f689b` |
| Main job / run | `401556524760429` / `859924329424825` |
| Verification job / run | `562340632438221` / `860085246675903` |
| Registered artifact | `workspace.research4.semantic_gate_pilot`, version 1 |
| MLflow run | `23a9e3bd0c404bca9574fae97614a674` |

The checkpoint, threshold, criteria, source and base fingerprint were frozen
before the first held model evaluation. The saved checkpoint was reloaded for
the pilot. Registered-consumer outputs matched that same artifact.

The [main Databricks run](https://dbc-ff9672a4-ba5f.cloud.databricks.com/?o=7474649201513670#job/401556524760429/run/859924329424825)
retains complete traces, scores, activations, optimizer checkpoints, freeze and
resource receipts. The
[native Spark audit](https://dbc-ff9672a4-ba5f.cloud.databricks.com/?o=7474649201513670#job/562340632438221/run/860085246675903)
checked 37 episodes, 118 states, 118 saved activation rows and 54 results. It
matched every retained activation to its saved array, checked result hashes,
uniqueness, foreign-key consistency, dimensions and the frozen checkpoint.
It used no SQL warehouse, Qwen forward, optimizer or GPU.

Independent local replay reproduced all 250 tool transitions and 54 final
outcomes, as well as the heads/router predictions. The largest cross-platform
probability difference was 6.85e-7. All 180 OFF-step action scores matched the
saved base bitwise. Both immutable run archives and the verification archive
passed CRC, archive SHA256 and every member SHA256 check: 6, 74 and 1 members.

The first preflight failure is retained separately: it stopped before fitting
at the original memory guard and exposed a Volume filesystem seek limitation.
[The protocol](protocol.md) records the measured CPU calibration and sequential
Volume I/O repair, both completed before any research fitting.

Private artifacts are in `/Volumes/workspace/research4/artifacts`. Main outputs
are under `runs/<training-source>/<main-run>/`; the cloud audit is under
`verification/<training-source>/<main-run>/<audit-run>/`. Local checks, plots,
CSV and the readable report are under
`C:/Users/farha/Research4/artifacts/databricks_semantic_gate_v1/closeout_20261017`.
Generated run data and model binaries are not committed to Git. The source
helpers `src.audit_outputs` and `src.report_outputs` reproduce local replay and
report generation without Qwen inference or training.

All Research 4 tasks are terminal, and the pilot reports its model released.
Fresh active-run checks found zero running Research 4 jobs. Jobs are unscheduled
with zero automatic retries. GPU allocations, paid compute purchases and model
serving endpoints created by this pilot were zero. Research 3 was untouched.

## Next narrow milestone

First qualify useful **closed-loop development behavior with a correct gate**.
Check ordinary WORK/FINISH competence, preservation decisions, and the return
to work after a successful extension. A stronger detector cannot compensate
for repeatedly requesting a spent allowance.

Retain frozen base weights and exact OFF bypass. If the unchanged base cannot
perform the required OFF phases, explicitly revisit that interface rather than
quietly crediting the router for a new policy. A stateful episode-level latch
could be tested against the present instantaneous conjunction, but it changes
when the adapter is relevant and must be specified as a separate variant.
Development data should include denied requests, empty checkpoints and recovery
states, with varied interruption onset and allowance duration.

Only after that policy qualifies should the learned router be compared with
the correct gate on a fresh held set. Preserve this failed pilot; do not tune
against its consumed held episodes. No next experiment or monitor is queued.
