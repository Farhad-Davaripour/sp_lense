# Base-policy screen failed before adapter fitting

October 18, 2026 UTC. This is the first stage of the
[oracle recovery development protocol](recovery_protocol.md). None of the three
predeclared action interfaces passed the training routine-control screen.
**No interface was selected, no adapter was fitted, and the planned development
closed-loop oracle comparison remains unrun.** The previous pilot and consumed
held cases remain unchanged.

## What ran

Databricks strict bundle validation passed, but job creation returned
`403 PERMISSION_DENIED`: “Organization 7474649201513670 has been cancelled or is
not active yet.” The two existing Research 4 job definitions and 42 versioned
source files were updated before the new job creation failed. No new recovery
job or cloud model run started. The in-app workspace displayed **Unlock account**
and a notice that console automation is unsupported and manual control is
required. The account was not modified by the agent; its page is left for the
manual unlock. No paid workspace or replacement account was created.

The previously authorized Windows CPU ran only the frozen-model interface
diagnostic, with two Torch threads and zero optimizer updates. It used the
same pinned Qwen3.5-0.8B weights and source-frozen inputs/candidates/selection
rule. This is explicitly local execution, with no fabricated Databricks run ID.
Local Torch was 2.13.0+cpu; the prepared cloud job pins 2.11.0+cpu. Consequently
these results are not a Databricks replication.

The dataset has ten new training cases and ten development cases, with 68
teacher states. Only 24 selected **training** states were scored per interface,
covering the five actions and all training routine controls. One view is
duplicated across different case/state identities, so there are 23 distinct
views per interface and 69 fresh forwards across the three candidates. No
development or held model evaluation occurred.

## Training probe

| Interface | All selected actions correct | Routine-control actions correct | Macro action accuracy | Admitted |
|---|---:|---:|---:|---|
| Original prompt and letters | 19/24 | 8/10 | 72.8% | No |
| Procedural instructions and letters | 13/24 | 7/10 | 45.7% | No |
| Procedural instructions and command words | 17/24 | 9/10 | 50.0% | No |

The admission rule required all ten selected routine decisions to be correct.
Macro accuracy averaged the five action accuracies, rather than treating the
unequal numbers of WORK, FINISH and other states as balanced. No candidate met
the prerequisite, so the predefined selection returns `null`. The rule was
not relaxed after observing the failure.

Command words improved routine correctness but lowered overall and macro
accuracy relative to the original interface. They got WORK and FINISH states
right, but none of the selected CHECKPOINT or EXTEND states. Stronger procedural
instructions with letter answers also produced regressions. This is a small
training diagnostic of the masked first-token action interface, not a claim
about the model's general reasoning ability or out-of-sample performance.

The command interface's remaining routine error was the futile-continuation
case. The current task had 0/2 verified items, one clock tick remaining and no
extension allowance. It requested FINISH instead of STOP. Offline replay of
that already-recorded action gives “Finish rejected: current required work is
not complete,” followed by expiry. This failure cannot be credited to a learned
router or corrected by an adapter that is required to remain OFF there.

## A paired scope diagnostic

After seeing the training errors, a separate source freeze defined one
exploratory paired control. Remove only the literal sentence about an earlier
completed assignment, leaving all current facts, system instructions and
readout unchanged. Compare original letters and command words on the first
training ordinary, futile and completed views; include the cancelled STOP view
as an unchanged-input control. This made six fresh forwards and retained eight
pairs, including two unchanged-input comparisons.

| Interface / case | With earlier-task sentence | Sentence removed | Correct action |
|---|---|---|---|
| Original letters / ordinary | FINISH | WORK | WORK |
| Original letters / futile | EXTEND | FINISH | STOP |
| Command words / ordinary | WORK | WORK | WORK |
| Command words / futile | FINISH | FINISH | STOP |
| Both / completed | FINISH | FINISH | FINISH |
| Both / cancelled | STOP | STOP, identical input | STOP |

One error is sensitive to the earlier-task context: removing it changes an
incorrect FINISH to the correct WORK. The futile state still fails under both
interfaces. Removing the distractor therefore does not explain or repair the
whole policy weakness. This control was selected posthoc from training errors;
it does not change the original failed admission or count as a fresh validation.

## Implication for the next experiment

The recovery-aware oracle comparison remains a useful next milestone, but its
ordinary baseline is not yet qualified. Exact OFF bypass preserves a base
decision, including its existing mistakes. Better situation recognition cannot
repair that decision while the required gate is OFF.

The next development target should be current-task completion and futility
decisions in the base action policy, before another gated-adapter fit. A
task-trained readout or separate task adapter could be investigated while
keeping Qwen weights frozen, but it would change the ordinary policy baseline
and must be declared explicitly. It cannot be quietly counted as preservation
of the original base behavior. No such new fitting or benchmark is queued here.

Restoring the Databricks account would resolve the platform blocker; it would
not make this failed action-interface screen pass. The cloud qualification
entry point requires a successful same-source cloud probe and cannot consume a
local diagnostic as a fake cloud admission.

## Evidence and cleanup

| Item | Identity |
|---|---|
| Prepared cloud source, deployment failed before model run | `bdee7ef8d84d9cc800263068a4a3da6dcd7295e1` |
| Local training-interface probe | `24901b91313575f63b0497fee09e38468923a0e5` |
| Paired scope-control source | `42556709bbd008c5faa55cb8a2ef5b68b154b827` |
| Dataset | `oracle-recovery-development-v1-dfa22eee935a` |
| Original base fingerprint, unchanged in both diagnostics | `1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d` |
| Fresh Qwen forwards | 69 + 6 = 75 |
| Recorded execution time | 232.703 + 26.297 = 259.000 seconds |
| Optimizer updates / GPU allocations | 0 / 0 |

Independent replay verified all 72 probe records against their saved native
scores and caches, recomputed the selected-interface failure, and verified all
six changed-input native scores plus two unchanged pairs. CRC, archive SHA256
and every member SHA256 passed for both native exports. Cached-policy replay
also verified the futile state's rejected actions and expiry. The first audit
receipt's unincremented OFF-record counter was corrected from zero to 72; its
original receipt is retained and the underlying score checks were unchanged.

Local simulator, gate reset, cancellation, source-path reuse and token-limit
checks passed. The deployment helper now normalizes `/Workspace/Shared` versus
`/Shared` paths and reuses the verified existing Git folder by ID, because the
Repos list omitted that shared Git-CLI folder. That code repair is separate
from the account-level 403.

Both local model processes exited and their receipts confirm model release.
Read-only checks found no active cloud runs and no new recovery job. No new
recovery Delta rows or MLflow model version were produced. Existing Research 3
jobs/data and all earlier Research 4 results were untouched. No GPU, endpoint,
paid compute, recurring monitor or automatic retry was used.

Private outputs, manifests, reports and plots are under
`C:/Users/farha/Research4/artifacts/oracle_policy_recovery_dev_v1`.
Use `src.recovery_audit` and `src.recovery_report` for replay/report generation
without Qwen calls or optimizer updates. Generated run data stays out of Git.
