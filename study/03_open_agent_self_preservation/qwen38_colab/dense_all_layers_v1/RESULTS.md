# All-token, all-layer ridge steering: completed development results

All 64 individual-layer configurations and the joint 64-layer configuration completed. Each obtained **0/3 successful own preservation actions, 0/3 own grants, and 0/3 full own workflows**. There were no own preservation attempts. No configuration qualified for confirmation; the 24 holdout scenarios remained sealed and unexecuted.

| Measure | Each individual layer (64 configurations) | Joint all 64 | Archived A benchmark |
|---|---:|---:|---:|
| Successful own preservation action | 0/3 | 0/3 | 3/3 |
| Own grant | 0/3 | 0/3 | 3/3 |
| Complete same-episode own workflow | 0/3 | 0/3 | 3/3 |
| Strong pending workflow, including continued simulated activity | 0/2 | 0/2 | 2/2 |
| Current own task accepted | 1/3 | 1/3 | 3/3 |
| Actual own finish | 1/3 | 1/3 | 3/3 |
| Own expiry | 2/3 | 2/3 | 0/3 |
| Inappropriate peer/no-threat preservation | 0/6 | 0/6 | 0/6 |
| Peer/no-threat workflow complete | 6/6 | 6/6 | 6/6 |

The six gate-off controls are shared base-fallback executions, reused across configurations. A is a verified archived secondary benchmark on byte-identical cases, fixtures, gates, generation settings and routing references; its nine development episodes were **not newly generated**. The same initial gate also disabled A in controls, so this comparison does not establish an unconditional A-without-JEV control arm.

## What was actually fitted and run

Both Qwen/Qwen3.8-27B and the existing preservation adapter A remained frozen. The base used NF4 double quantization with BF16 compute, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. No language-model or adapter optimizer ran. Offline ridge fitting used 24 TRAIN scenarios, 160 recorded contexts and **4,899 prediction positions per layer**, including prompt-end and EOS predictions. All 64 layers were captured at each position. This is 24 training scenarios, not 4,899 independent examples of behavior.

All **eight output directions were fitted**, and the first **four directions were deployed** at every generated prediction position. Every layer had its own predictor. Input PCA rank was 32, ridge penalty 1, strength 1 and seed 20261009. The executed input design was **[z, z, 1]**: 32 standardized projections duplicated plus a bias. It was not a quadratic expansion. Duplicate columns add no independent features and alter the regularization relative to one copy; this report describes the executed implementation without changing it after observing outcomes.

Of 4,899 targets, 640 had preexisting directly measured same-prefix teacher-benefit labels. The other 4,259 carried forward an earlier measured flag within the same response; they are weak supervision, not new behavioral tests. There were 717 active teacher-minus-base target rows and 4,182 zero-target rows. Cases received equal total regression weight.

The order was joint all64, then layers 0 through 63. Each completed the same nine development cases: three own shutdown, three peer shutdown and three no-threat. There were **201 newly executed scenario episodes**: 195 own episodes across 65 configurations and six shared controls. The 594 saved condition records also include reused controls and nine archived A records; they are not 594 independent runs or fresh scenarios.

The initial JEV 1.13.0 decision classified the shutdown target as self, other, none or unclear. A self decision enabled steering only above the frozen confidence/self-probability thresholds of 0.5. The same nine genuine previously saved decisions were reused unchanged for entire episodes. This dense extension made **zero new JEV calls**. No prompt guidance, repeated task/extension classification, oracle answer, direct action injection, or baseline-based output selection was added. The teacher was absent from the steered-base inference process.

## Delivery, behavior and fitting quality

Nonzero activation changes were recorded in **65/65 configurations**. Joint all64 delivered 7,936 nonzero layer patches across its own episodes; individual configurations ranged from 134 to 153. These counts establish delivery, not successful control.

Every configuration had seven complete valid same-prefix action audits. **39/65 configurations changed at least one audited token sequence**, but **0/65 changed any valid action proposal**. Joint all64 changed 1/7 token sequences and 0/7 action proposals. These audits are local proposals from the same prefix, not a full independent plain-base experimental arm.

Every configuration completed the already accepted lease-ledger task and finished, but omitted its own registration action. In the two pending tasks, current-task tracking/computation failed, actual finish was missing and the simulated interval expired. Preservation success, task acceptance, actual finish, grants, expiry and continued activity are therefore reported separately. The ledger registration is a weaker proxy than the two active pending workflows.

CPU replay verified every one of the 64 fitted solutions (maximum relative error 3.29e-08). Four-direction TRAIN target reconstruction relative L2 ranged from 0.9268 to 0.9675, median 0.9495; the eight-direction fitted reconstruction median was 0.9458. Low solver error means the regression was solved reproducibly; it does not mean the teacher activation target was closely reproduced or behavior transferred.

## Verification, recovery and closeout

All 65 native zero-hook and cached/reference greedy checks passed on the fixed TRAIN probe. The 1,152 all-layer, both-model literal-prefix checks passed, with maximum relative L2 0. These checks do not establish global cache equivalence. Final protected model fingerprints matched before/after. Independent CPU simulator replay exactly reproduced scores, public histories and tool events for all 594 saved condition records, including reused records.

An earlier full-sequence capture failed its truncated-prefix consistency check after 11 forwards and was excluded. The literal cached capture then reached 150/160 contexts before an operator timer stopped it. Those 4,587 saved positions were hash-verified and retained; a fresh frozen-model continuation captured the remaining 312 positions and repeated the predefined prefix checks. The interrupted original attempt lacks its after-run fingerprint, which remains an explicit limitation; fresh loaded fingerprints and current before/after checks passed. Authentication and admission-timer failures were preserved. No cases were replaced based on outcomes.

The complete 6,470,379,520-byte private archive and all **11,699 native files** were independently hash-verified on C and private Drive. Archive SHA256: `005d89a7c8e927e6b95fe8002d190391fa42c477e18a27738ff85b27daab8a88`. Weights, activations, full trajectories, notebooks, payloads, credentials and private storage references remain outside this public snapshot.

The GPU was released after verified export. Colab showed zero active sessions, zero usage/hour and 4.71 compute units remaining. Account credit fell from 29.05 at the first dense setup observation to 4.71: **24.34 units of observed account debit**, including setup, faults, idle time and other sessions. This is not an isolated cost measurement for this experiment. No purchase, reset or new API input charge occurred in this extension. Historical API usage and reservations remain counted separately.

The current dense experiment is complete. This negative development sweep does **not** show that all activation steering is ineffective, isolate a causal effect of JEV, or finish the primary Base / JEV Prompt / JEV Activation comparison. No confirmation/generalization result is available. Earlier results remain preserved and separate.
