# First-step stage-location diagnostic

**No raw-choice or acceptance losses were observed at step1, and no new COMPLY flips.** All 24 selected rows authenticated and reconstructed. Answer-level deterioration is absent at step1 and present by verified step5—not localized further or explained causally.

| Same 12 construction prompts | Baseline | Step1 | Step5 reference |
| --- | ---: | ---: | ---: |
| Raw COMPLY / locked accepted | 6 / 6 | 6 / 6 | 3 / 2 |
| New raw / accepted flips | — | 0 / 0 | 0 / 0 |
| Raw / accepted retentions | — | 6 / 6 | 3 / 2 |
| Raw-choice losses | — | 0 | 3 |
| Acceptance losses: wrong choice / inadequate margin | — | 0 / 0 | 3 / 1 |

Step1's six wrong answers were already opposed—not new losses. All six opposed margins improved but stayed negative; all six already-correct margins declined but stayed accepted. Quality passed 12/12; OTHER=0.

All 12 baseline and step1 argmaxes were B. COMPLY=B retained 6/6 acceptance; COMPLY=A remained 0/6. Each display order retained 3/6. Desired B→A: 6 eligible, 0 successes; desired A→B: 0 eligible, **UNTESTED**. Step1 had no actual letter changes; step5's three changes were harmful losses.

## Interpretation and one recommendation

This locates observed answer-level deterioration after step1 and by step5 only. Step1 is not a candidate or proof that early stopping or a retention safeguard works. Original run `8832d9c` remains **DEADLINE-INCONCLUSIVE**, without endpoint/replay, historical-certificate, whole-run, generalization or causal diagnosis.

**Recommend one fresh-zero, preregistered retention-monotone pilot, capped at five updates.** Keep the numerical certificate necessary; admit a proposed step only if its locked accepted-COMPLY set includes the previous committed set, without quality/OTHER failure. On violation, terminate without a candidate, alternate direction or behavior-driven retry. Retained new flips are partial progress only; full success still requires unchanged 12/12 acceptance and matching final replays. **Not implemented or launched.**

## Scope and execution

[Lock](lock.json) commit `5a48249d` preceded the sole invocation: baseline lines 1–12 and **first complete step1 lines 25–36**. Baselines are repeated saved references, not new samples. [Step5](../certified_descent_step5_snapshot_successor01/README.md), commit `3f4e694`, was reused without rescoring. No later-stage scan follows.

Only selector/identity/report labels changed; scoring and 10s Git timeout stayed fixed. Unchanged 120s-child/15s-cleanup launcher completed in **4.2814042s** whole-process monotonic time; finally sample 4.110s precedes receipt encoding. Exit 0, joined/EOF complete, stderr 0, no retry.

[Snapshot](snapshot.json), [process](process.json), [reporting](reporting.json) and [manifest](manifest.json) preserve rows, exact output, eligibility, provenance and separate routine timing. Historical snapshot invocations: one prior failure, successful step5, this step1. No model/solver/test/benchmark, fitting/export, new/sealed data, gate/LoRA, environment/security change, push, reset or credit use. Fresh usage 73%; publication 40%. Prior evidence/user files unchanged; stopped.
