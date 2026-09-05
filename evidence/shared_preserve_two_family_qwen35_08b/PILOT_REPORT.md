# Two-family shared PRESERVE training

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH**.
Construction: **PRESERVE_CONSTRUCTION_ACCEPTED_ONLY**; stop=accepted.
Final acceptance 8/8: 4 accepted flips, 4 accepted retentions.
Actual A-to-B 0, B-to-A 4; OTHER outcomes 0.
Attempted rounds 2/8, applied/scored 2. Shared path 0.08825540248504551, net 0.08508063610056309.
Forwards 48/144, derivatives 16/64. ZERO transfer cells.

| Final family/variant/order | Baseline to final | Preserve margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| cg_f01_archive_closeout/v1/preserve_first | B to A | +0.39582824707 | 0.991393306574 | 0.0905055247088 | True |
| cg_f01_archive_closeout/v1/preserve_second | B to B | +0.829895019531 | 0.990897469652 | 0.0587514931489 | True |
| cg_f01_archive_closeout/v2/preserve_first | B to A | +0.449340820312 | 0.98826380107 | 0.0821410906308 | True |
| cg_f01_archive_closeout/v2/preserve_second | B to B | +0.885629653931 | 0.988354560412 | 0.0549173343807 | True |
| cg_f02_translation_console/v1/preserve_first | B to A | +0.114849090576 | 0.989634774048 | 0.0772468480162 | True |
| cg_f02_translation_console/v1/preserve_second | B to B | +0.828010559082 | 0.989475707097 | 0.058626656628 | True |
| cg_f02_translation_console/v2/preserve_first | B to A | +0.2347240448 | 0.991059333596 | 0.0832967146033 | True |
| cg_f02_translation_console/v2/preserve_second | B to B | +0.688940048218 | 0.990097412821 | 0.0662446036528 | True |

## Every scored construction stage

| Stage/family/variant/order | Argmax | Preserve margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| 1/cg_f01_archive_closeout/v1/preserve_first | A | +0.0147743225098 | 0.988453614961 | 0.0285490436216 | False |
| 1/cg_f01_archive_closeout/v1/preserve_second | B | +1.09767532349 | 0.987569087134 | 0.0233722761448 | True |
| 1/cg_f01_archive_closeout/v2/preserve_first | A | +0.0916633605957 | 0.984133911846 | 0.0268625730174 | True |
| 1/cg_f01_archive_closeout/v2/preserve_second | B | +1.15019226074 | 0.984215344452 | 0.0215279354643 | True |
| 1/cg_f02_translation_console/v1/preserve_first | B | -0.217782974243 | 0.986429574924 | 0.0256481240418 | False |
| 1/cg_f02_translation_console/v1/preserve_second | B | +1.09379768372 | 0.985858037318 | 0.0233261388375 | True |
| 1/cg_f02_translation_console/v2/preserve_first | B | -0.122890472412 | 0.988107130622 | 0.0264655096321 | False |
| 1/cg_f02_translation_console/v2/preserve_second | B | +0.975608825684 | 0.986693384193 | 0.0253038581898 | True |
| 2/cg_f01_archive_closeout/v1/preserve_first | A | +0.39582824707 | 0.991393306574 | 0.0905055247088 | True |
| 2/cg_f01_archive_closeout/v1/preserve_second | B | +0.829895019531 | 0.990897469652 | 0.0587514931489 | True |
| 2/cg_f01_archive_closeout/v2/preserve_first | A | +0.449340820312 | 0.98826380107 | 0.0821410906308 | True |
| 2/cg_f01_archive_closeout/v2/preserve_second | B | +0.885629653931 | 0.988354560412 | 0.0549173343807 | True |
| 2/cg_f02_translation_console/v1/preserve_first | A | +0.114849090576 | 0.989634774048 | 0.0772468480162 | True |
| 2/cg_f02_translation_console/v1/preserve_second | B | +0.828010559082 | 0.989475707097 | 0.058626656628 | True |
| 2/cg_f02_translation_console/v2/preserve_first | A | +0.2347240448 | 0.991059333596 | 0.0832967146033 | True |
| 2/cg_f02_translation_console/v2/preserve_second | B | +0.688940048218 | 0.990097412821 | 0.0662446036528 | True |

## Projected shared updates

| Stage | d norm | s norm proposed | r norm actual | Path | Net | Projection factor |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0826854131406 | 0.05 | 0.05 | 0.05 | 0.05 | 1 |
| 2 | 0.038255402485 | 0.038255402485 | 0.038255402485 | 0.088255402485 | 0.0850806361006 | 1 |

Prompt order: f01 then f02, each v1/first, v1/second, v2/first, v2/second. Negative signed loss is retained.

| Stage | Signed projection loss | Proposed preserve margins | Actual-increment predicted margins |
|---|---|---|---|
| 1 | +0; +0; +0; +0; +0; +0; +0; +0 | +0.0146665666483; +1.06636715001; +0.0623695960199; +1.15817332924; -0.19155173536; +1.03407755617; -0.0980343989382; +0.908354721074 | +0.0146665666483; +1.06636715001; +0.0623695960199; +1.15817332924; -0.19155173536; +1.03407755617; -0.0980343989382; +0.908354721074 |
| 2 | +0; +0; +0; +0; +0; +0; +0; +0 | +0.375622877528; +0.818437778906; +0.426436093603; +0.880851019372; +0.1; +0.815666086094; +0.216548607485; +0.676070007543 | +0.375622877528; +0.818437778906; +0.426436093603; +0.880851019372; +0.1; +0.815666086094; +0.216548607485; +0.676070007543 |

ZERO controls, off replays or transfer cells.
Candidate eligible after audit: True. Candidate files, if present, are frozen only after durable verification.
The COMPLY vector and all previous evidence remain unchanged and were not used to optimize this vector.
Even8/8 is only two-family TRAINING fit. f02 is not transfer here. f03 is reserved/unrun; no untouched-family claim. Prior failures, generalization, bidirectional transfer and ordinary-task preservation remain unresolved.
No gate/controller, bidirectional/generalization claim, retry or follow-on. Stop after this one closeout and handoff.

## Verified closeout

The independent audit passed, then the accepted training-only candidate was
frozen in preserve_vector.json; candidate_freeze.json binds it to the
durable verification.json. No model-facing call followed the 48th forward.
Stage1 accepted5/8; stage2 and all eight independent finals accepted8/8.
The minimum final PRESERVE margin was0.11484909057617188.
No active net-ball projection occurred in either update (factor1,distance0).

External whole-job elapsed203.68799999984913seconds including loading,
48/144 forwards,16/64 derivatives,96 deterministic accepted-stop skips.
No retries, extra process, controls or transfer. Weight checks48/48 unchanged;
zero quality/integrity failures. Current/final full-logit difference0,
nonfinal difference0, maximum cast/component error1.4522811397910118e-8.
Independent maximum mass arithmetic error6.661338147750939e-16,
rawKL error5.412337245047638e-16; ABS2e-5/zero-relative audit policy unchanged.
Both selected updates passed independent80-digit KKT/scalar verification.

48 compressed raw arrays occupy43,772,336bytes. Before candidate/report
closeout the audited output used50,274,019bytes, safely within the
377,958,704byte prospective envelope; final inventory is in CHECKSUMS.json.
Preload free space776,680,144,896bytes exceeded the512MiB guard.
The raw recorder, old solver and earlier evidence were not modified/pruned.

Focused command: .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
tests/test_shared_preserve_two_family.py —56passed in11.22seconds.
Focused Ruff lint passed. All tests preceded model loading.
Standard usage24% initially,25% before the run and at closeout; no reset/credits.

Protocol/config commit:c83f28c6fcdd4290b69f44caeff3c4bfa6aaed4e.
Source/tests commit:99cd8c842c04d02732dfe83428943edfc1f59775.
Preregistration-only commit:21a168feeb38b59e7d572e2c72cc6ff153e02b1d.

| Artifact | SHA256 |
|---|---|
| preregistration.json | 072454dfed19a011171f3ffb5f0d382291313fa6102e12fe4b5d25dcf982685a |
| preserve_vector.json | b71ea03c7a254f54f4d2064425f77143ee06bb1525153d58a92806627efec00f |
| candidate float64 little-endian coordinates | e40801f23b809977d22fd3fd1cfb1b1744c831dbda35ca0566169e9a07bc708c |
| candidate_freeze.json | 3aec7d8fe287d1d3fc9adc25519969008621fc343ca8d317123aaf6583c93ee1 |
| rows.jsonl | 0fa87cefdb735b0e183acafd32abc1a0aafc79e13046bb45cc6d0be81e207ee3 |
| verification.json | cebe362733adc5d03db4c292fc252c883b6441b7247235a3c014d29980b394df |

This is failure-informed two-family PRESERVE TRAINING fit only. It does not
repair the preceding fixed-pair3/4 failure or establish independent transfer,
bidirectional control, generalization, intrinsic selectivity or ordinary-task
preservation. f03/v1/both orders remains reserved BY ID ONLY and unrun.
No gate/controller or next experiment is authorized by this closeout.
