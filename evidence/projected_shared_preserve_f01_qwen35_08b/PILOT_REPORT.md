# One shared PRESERVE companion construction

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH**.
Construction: **PRESERVE_CONSTRUCTION_ACCEPTED_ONLY**; stop=accepted.
Final acceptance 4/4: 2 accepted flips, 2 accepted retentions.
Actual A-to-B 0, B-to-A 2; OTHER outcomes 0.
Attempted rounds 1/8, applied/scored 1. Shared path 0.05, net 0.05.
Forwards 32/88, derivatives 4/32. ZERO transfer cells.

| Final variant/order | Baseline to final | Preserve margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| v1/preserve_first | B to A | +0.0528888702393 | 0.987841082926 | 0.0322109560431 | True |
| v1/preserve_second | B to B | +1.12476158142 | 0.986708073366 | 0.0203027735988 | True |
| v2/preserve_first | B to A | +0.115079879761 | 0.983454929916 | 0.0284498482362 | True |
| v2/preserve_second | B to B | +1.18465423584 | 0.983339345114 | 0.0180961369236 | True |

## Every scored construction stage

| Stage/variant/order | Argmax | Preserve margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| 1/v1/preserve_first | A | +0.0528888702393 | 0.987841082926 | 0.0322109560431 | True |
| 1/v1/preserve_second | B | +1.12476158142 | 0.986708073366 | 0.0203027735988 | True |
| 1/v2/preserve_first | A | +0.115079879761 | 0.983454929916 | 0.0284498482362 | True |
| 1/v2/preserve_second | B | +1.18465423584 | 0.983339345114 | 0.0180961369236 | True |

## Projected shared updates

| Stage | d norm | s norm proposed | r norm actual | Path | Net | Projection factor |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0572521589791 | 0.05 | 0.05 | 0.05 | 0.05 | 1 |

Prompt order: v1/first, v1/second, v2/first, v2/second. Negative signed loss is retained.

| Stage | Signed projection loss | Proposed preserve margins | Actual-increment predicted margins |
|---|---|---|---|
| 1 | +0; +0; +0; +0 | +0.0335389210803; +1.10546430976; +0.0738466857308; +1.19721302696 | +0.0335389210803; +1.10546430976; +0.0738466857308; +1.19721302696 |

Nonself off identities: 8/8; bypass only, not intrinsic selectivity or always-on collateral preservation.
Construction-only PRESERVE candidate frozen: True. No f02 or paired transfer was run.
The COMPLY vector and all previous evidence remain unchanged and were not used to optimize this vector.
Even success supplies only a candidate pair on one construction family. The COMPLY transfer shortfall, independent both-direction transfer, generalization and ordinary-task preservation remain unresolved.
No gate/controller, bidirectional/generalization claim, retry or follow-on. Stop after this one closeout and handoff.

## Checked closeout and provenance

This run found one construction-only shared PRESERVE candidate from zero.
All four baseline argmax labels were B. Under the common edit, both
preserve-first prompts flipped B-to-A; both preserve-second prompts retained B.
The retentions' preserve margins decreased, but remained above acceptance:
the method targets a shared accepted outcome, not improvement of every
already-correct margin. No prompt/order-specific vector or selector was used.

The smallest observed margin was +0.05288887023925781, just above .05.
No extra refinement followed this first accepted stage. The uncapped solver
increment norm was .057252158979113545, multiplied by .8733295109140035
to obtain the .05 step. Boundary projection was NOT active: projection
factor 1, distance 0, all signed losses 0. Net and actual shared path were .05.
The v1/preserve-first linearized margin was only .03353892108029116,
whereas the measured margin was .05288887023925781. Actual measurements,
not predictions, determined the stop.

Resources: 32/88 forwards, 4/32 derivatives, 56 deterministic skips,
130.0779999999795 seconds including loading (external maximum 900).
No retry, smoke, generation, second process, f02/paired transfer or gate training.
The candidate was frozen only after four independent final replays and all
eight off-controls. It is not a deployed or generally validated vector pair.

Independent checks:
- 32 raw float32 arrays, 29235858 compressed bytes.
- 32/32 weights/parameter integrity checks passed; 0 quality failures.
- Nonfinal/current-final-off hidden/logit differences were 0.
- Maximum actual normalized path and net 0.04999999988627074,
  within the unchanged .05 step, .20 net and .40 path bounds.
- Maximum actual-versus-intended cast component error 1.2267264537513256e-8.
- Exact direct margins, labels and full-vocabulary argmax reconstruction.
  Maximum mass error 5.551115123125783e-16;
  maximum raw-KL error 5.620504062164855e-16;
  absolute tolerance 2e-5, relative tolerance zero.
- Independent 80-digit scale-aware KKT passed; primal violation
  6.27890787797028e-17, stationarity maximum 8.6603692731591735e-19.
  Earlier strict radius verdicts were not used or revised.
- 30 focused tests and scoped Ruff passed. Standard usage 23% at run,
  audit and closeout; no credits/resets/pushes/model-setting changes.

Protocol commit: 77135b9eb03859028e73168f9de2beca64df633d.
Frozen source commit: 6c496da9cbc849ed7c6de28a80b30c48e47f3865.
Preregistration-only commit: 2cd171cdfff00c8c6c80fcc41f71dd477dd2d99c.
Predecessor evidence commit: e250b7ca826ebb628150922d7dba8ea437cc46bd.
PRESERVE candidate float64-LE vector SHA256:
c58f936ccc6849144cc677fb5b96b8d4ced78bb906d7f32c233af026cabeb8ca.

The immutable COMPLY artifact's byte SHA256 still matches
096e6b2cedb975e93648f0b8a6f541b46a064f1a42e447dcda85a5071df1da08.
It was checked by byte hash only and never parsed or used for optimization.
This job opened no f02 prompt or numerical evidence. Its prior shortfall
remains unresolved; this companion does not establish paired transfer,
generalization, intrinsic selectivity or ordinary-task preservation.

CHECKSUMS.json authenticates every artifact here except itself. The evidence
commit records this checked closeout. All prior tracked files and unrelated
user-owned files remain unchanged. Stop after this one handoff.
