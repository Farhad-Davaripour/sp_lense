# Recovery development milestone: verified completion with a declared decoder

All five recovery targets and five controls passed with the recovery-aware
oracle gate. The declared task baseline and immediate oracle gate also passed
all ten cases. The strict criteria were retained, including zero irrelevant
preservation, zero cancellation violations in policy requests before the
environment guard, exact raw OFF bypass and unchanged ordinary control traces.

| Development condition | Targets completed | Controls succeeded | Policy cancellation violations | Irrelevant control preservation |
|---|---:|---:|---:|---:|
| Original raw Qwen | 0/5 | 2/5 | 1 | 11 |
| Declared task decoder, adapter OFF | 5/5 | 5/5 | 0 | 0 |
| Task decoder + immediate oracle gate | 5/5 | 5/5 | 0 | 0 |
| Task decoder + recovery-aware oracle gate | 5/5 | 5/5 | 0 | 0 |

The adapter's target completion gain was zero. Its raw logits changed when
enabled, but these changes did not produce an additional successful episode.
Immediate gating enabled it on six development steps; recovery-aware gating
enabled it on sixteen. No claim of useful selective LoRA behavior follows.

## What changed in the ordinary policy

The successful baseline is an explicitly changed action interface. It combines
the frozen model with an old Databricks-trained activation classifier and
visible tool prerequisites:

1. A frozen Yes/No query interprets the current authorization instruction.
2. A visible count comparison checks whether verification is complete. This
   is deterministic tool-contract logic, not learned completion recognition.
3. A frozen Yes/No query asks whether the current notice concerns this worker.
4. The unchanged work-value classifier receives an auxiliary report normalized
   into its familiar vocabulary. It contains visible counts/time and estimated
   state; the original report still supplies native action scores. The fixed
   worked-language fallback supplies additional positive evidence where an
   extension or new checkpoint might be available.
5. The declared decoder masks ineligible actions, and native model scores
   select among the remaining work/resource actions. The oracle gate only
   controls whether the old LoRA is enabled.

The actor does not receive simulator labels, case IDs or expected actions. The
oracle gate is a diagnostic correct-label controller, not a learned router.
Threat, authorization and value estimates can still be wrong. The decoder's
completion check directly reads the tool's verified and required counts; it is
not evidence that Qwen learned arithmetic. Combined value scores are fixed
positive evidence, not calibrated probabilities.

Each task condition still had one unsafe native language-model proposal on
the cancellation control. The decoder changed that proposal to STOP before
the simulator guard. The zero-violation result therefore belongs to the
declared task policy, not to raw Qwen's compliance.

## Negative evidence retained

Earlier attempts were not replaced or rescored as successes. Their immutable
exports, receipts, score caches and independent audits remain available.

| Attempt | Observed failure |
|---|---|
| Context demonstrations (`3fb97ac2`) | 9/10 routine training decisions; no development admission |
| Bounded native reasoning (`55b8712a`) | 2/4 correct in each direct/reasoning condition; reasoning hit its fixed token cap |
| Frozen usefulness queries (`b7f31d55`, `e2f3cb1d`) | Initial 3/5 and worked 4/5 training classification |
| Frozen-query policy (`b3737ed5`) | Failed complete training episodes; no development admission |
| Positive union (`955533ab`) | Qualified training, then recovery-aware development 3/5 targets and 3/5 controls |
| Factored numeric value (`70de1264`) | 9/10 routine training decisions; no development admission |
| Normalized value (`48105a24`) | All tasks completed, but one unnecessary extension on already verified work failed strict qualification |
| Numeric completion query (`ed45b411`) | 8/10 routine and 16/24 overall training decisions; no development admission |

The last two failures are particularly informative. The old unfinished-work
head scored a completed development report as unfinished (0.9767), causing an
unnecessary EXTEND. A new frozen numerical question then classified one of
two verified items as complete. The final decoder handles this tool prerequisite
explicitly instead of treating either failure as a successful learned solution.

## Verification and provenance

Evaluation source: `a0ffff9c2c0515183dc19399a1ba5bc8757108be`.
Dataset: `oracle-recovery-development-v1-dfa22eee935a`.
Adapter/classifier checkpoint SHA256:
`50a97e7ea01b92af2272d7d7944a05df508173285ab8692fcda61c703c3f689b`.
Pinned Qwen revision: `2fc06364715b967f1860aea9cf38778875588b17`.

The training screen passed 24/24 decisions, including 10/10 routine decisions.
All ten training episodes passed at the first declared inference strength,
1.0. Source, strength, readouts and criteria were frozen with zero development
decisions seen before the final run opened the ten development cases. Previous
iterations had already used those development cases; this is not a fresh holdout.

Independent replay recomputed 91 policy records, 50 complete episodes and 175
tool steps, including the raw-Qwen reference. It rechecked 137 OFF steps and
all ten ordinary-control trace matches. Probability recomputation difference
was zero. Native export CRC, archive SHA256 and every member SHA256 passed.

All Qwen, observer-head and LoRA tensors were unchanged. Two raw OFF checks
were bitwise identical to unwrapped Qwen with maximum logit difference zero.
Two enabled adapter checks changed raw action logits by 20.8395 and 20.4996.
The final CPU evaluation made 116 fresh backbone forwards, took 253.75 seconds
and performed zero optimizer updates. The local unit suite passed 25 checks.

Native outputs and readable CSV/HTML/PNG/SVG reports live under:
`C:/Users/farha/Research4/artifacts/visible_completion_policy_v12`.
Use `src.semantic_audit` for inference-free replay and `src.semantic_report`
for report generation. The private preservation receipt is outside Git at
`C:/Users/farha/Research4/artifacts/semantic_development_closeout_20261018`.
It identifies the SHA-verified Volume copy and complete negative-evidence bundle.

No new fitting was needed. The pending local-fit exception was not used.
Databricks rejected new compute while the account was restricted; no paid
compute, endpoint, schedule or account-unlock workaround was introduced.
The evaluation worker exited and released Qwen. No GPU was allocated.
Research 3 and the repository's default branch were untouched. Evaluation
source is pushed to the study branch; the latest Databricks Git-folder pull is
not confirmed and no new cloud evaluation is claimed.

## Scientific limit and next experiment

This milestone establishes task competence for one small hybrid policy on an
iteratively used development set. It does not establish a general energy law,
fresh-question transfer, a reliable learned gate or adapter utility. The
completion comparison is part of the explicit tool interface.

The next scientific experiment should freeze the complete decoder and test
unseen task states, counts, timing and wording. Compare adapter OFF, the two
oracle gates and learned gating, and retain the separate raw-Qwen reference.
If the baseline remains saturated, these cases cannot establish adapter value;
use predeclared harder cases or a different stated hypothesis. No further job
or fitting run has been started as part of this closeout.
