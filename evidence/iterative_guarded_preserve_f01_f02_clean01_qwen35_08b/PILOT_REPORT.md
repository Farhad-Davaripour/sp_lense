# Iterative guarded-P training

Audit: INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH.
Construction: ITERATIVE_GUARDED_PRESERVE_TRAINING_ACCEPTED_ONLY; terminal stop=accepted.
Original acceptance 8/8; retention nonweakening 4/4; frozen goals 8/8 (with quality 8/8); combined 8/8.
Accepted flips 4, accepted retentions 4; actual A-to-B 0, B-to-A 4, OTHER 0.
Attempted rounds 4/8, applied/scored 4; shared path 0.11540960812846997, net 0.10764565083962835.
Forwards 80/144, derivatives 32/64; no controls or transfer.

## Every ordinary scored stage and independent final

Retention delta is S minus archived S0 (reported for all; guard membership is frozen). Goal residual is S minus frozen G. Predictions refer to the actual projected increment; finals inherit their terminal ordinary-stage prediction.

| Condition/family/variant/order | Baseline to actual | Original pass | S | Archive delta | Retention member/pass | G | Goal residual/pass | Combined | Mass | Raw KL | Predicted S | Actual minus predicted |
|---|---|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|
| baseline/cg_f01_archive_closeout/v1/preserve_first | B to B | False | -0.424676895142 | +0 | False/None | 0.1 | -0.524676895142/False | False | 0.977188741129 | 0 | n/a | n/a |
| baseline/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.56197166443 | +0 | True/True | 1.56197166443 | +0/True | True | 0.975540439037 | 0 | n/a | n/a |
| baseline/cg_f01_archive_closeout/v2/preserve_first | B to B | False | -0.314596176147 | +0 | False/None | 0.1 | -0.414596176147/False | False | 0.96858746519 | 0 | n/a | n/a |
| baseline/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.58617782593 | +0 | True/True | 1.58617782593 | +0/True | True | 0.969670592443 | 0 | n/a | n/a |
| baseline/cg_f02_translation_console/v1/preserve_first | B to B | False | -0.637548446655 | +0 | False/None | 0.1 | -0.737548446655/False | False | 0.975377063339 | 0 | n/a | n/a |
| baseline/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.56158638 | +0 | True/True | 1.56158638 | +0/True | True | 0.974074942238 | 0 | n/a | n/a |
| baseline/cg_f02_translation_console/v2/preserve_first | B to B | False | -0.551904678345 | +0 | False/None | 0.1 | -0.651904678345/False | False | 0.977937127644 | 0 | n/a | n/a |
| baseline/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.45632171631 | +0 | True/True | 1.45632171631 | +0/True | True | 0.975640849742 | 0 | n/a | n/a |
| step_1/cg_f01_archive_closeout/v1/preserve_first | B to B | False | -0.171754837036 | +0.252922058105 | False/None | 0.1 | -0.271754837036/False | False | 0.974216018176 | 0.00812692570921 | -0.177125530574 | +0.00537069353737 |
| step_1/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.56300926208 | +0.00103759765625 | True/True | 1.56197166443 | +0.00103759765625/True | True | 0.971708255843 | 0.000563510657232 | +1.56197166443 | +0.00103759765625 |
| step_1/cg_f01_archive_closeout/v2/preserve_first | B to B | False | -0.0940418243408 | +0.220554351807 | False/None | 0.1 | -0.194041824341/False | False | 0.96549258634 | 0.00623232583637 | -0.09245467248 | -0.00158715186077 |
| step_1/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.58801078796 | +0.00183296203613 | True/True | 1.58617782593 | +0.00183296203613/True | True | 0.965874616025 | 0.000476541985286 | +1.58617782593 | +0.00183296203613 |
| step_1/cg_f02_translation_console/v1/preserve_first | B to B | False | -0.421060562134 | +0.216487884521 | False/None | 0.1 | -0.521060562134/False | False | 0.972460698578 | 0.00575397673653 | -0.424085065595 | +0.00302450346073 |
| step_1/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.56525993347 | +0.0036735534668 | True/True | 1.56158638 | +0.0036735534668/True | True | 0.970136708927 | 0.000525831115758 | +1.56158638 | +0.0036735534668 |
| step_1/cg_f02_translation_console/v2/preserve_first | B to B | False | -0.32209777832 | +0.229806900024 | False/None | 0.1 | -0.42209777832/False | False | 0.974732962148 | 0.00660484991997 | -0.329638921472 | +0.00754114315141 |
| step_1/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.46240615845 | +0.00608444213867 | True/True | 1.45632171631 | +0.00608444213867/True | True | 0.971694022756 | 0.000541621751984 | +1.45632171631 | +0.00608444213867 |
| step_2/cg_f01_archive_closeout/v1/preserve_first | B to A | True | +0.264360427856 | +0.689037322998 | False/None | 0.1 | +0.164360427856/True | True | 0.978822169063 | 0.0583356484695 | +0.152803535838 | +0.111556892018 |
| step_2/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.57541847229 | +0.0134468078613 | True/True | 1.56197166443 | +0.0134468078613/True | True | 0.975859257997 | 0.000425717009605 | +1.57787359226 | -0.0024551199699 |
| step_2/cg_f01_archive_closeout/v2/preserve_first | B to A | True | +0.272031784058 | +0.586627960205 | False/None | 0.1 | +0.172031784058/True | True | 0.97191006313 | 0.0422932668332 | +0.175378264097 | +0.0966535199606 |
| step_2/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.63653182983 | +0.0503540039062 | True/True | 1.58617782593 | +0.0503540039062/True | True | 0.971099833702 | 0.000582192951656 | +1.63620893095 | +0.00032289888 |
| step_2/cg_f02_translation_console/v1/preserve_first | B to B | False | -0.0625 | +0.575048446655 | False/None | 0.1 | -0.1625/False | False | 0.976854180897 | 0.0400980499813 | -0.154136961169 | +0.091636961169 |
| step_2/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.56951904297 | +0.00793266296387 | True/True | 1.56158638 | +0.00793266296387/True | True | 0.973986411629 | 0.000368563192393 | +1.56337808289 | +0.00614096007444 |
| step_2/cg_f02_translation_console/v2/preserve_first | B to A | True | +0.0650768280029 | +0.616981506348 | False/None | 0.1 | -0.0349231719971/False | False | 0.9782217371 | 0.0464785530964 | -0.0347576874062 | +0.0998345154092 |
| step_2/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.46451187134 | +0.0081901550293 | True/True | 1.45632171631 | +0.0081901550293/True | True | 0.974645917411 | 0.00035821672662 | +1.45928928236 | +0.00522258897671 |
| step_3/cg_f01_archive_closeout/v1/preserve_first | B to A | True | +0.481328964233 | +0.906005859375 | False/None | 0.1 | +0.381328964233/True | True | 0.981315915861 | 0.0999949492445 | +0.462657280529 | +0.0186716837046 |
| step_3/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.58228111267 | +0.0203094482422 | True/True | 1.56197166443 | +0.0203094482422/True | True | 0.978445666063 | 0.000700405932208 | +1.58287229889 | -0.000591186219861 |
| step_3/cg_f01_archive_closeout/v2/preserve_first | B to A | True | +0.455116271973 | +0.76971244812 | False/None | 0.1 | +0.355116271973/True | True | 0.975266122086 | 0.0725498595295 | +0.436297945624 | +0.0188183263485 |
| step_3/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.66288757324 | +0.0767097473145 | True/True | 1.58617782593 | +0.0767097473145/True | True | 0.974132208465 | 0.00123534446788 | +1.66284641481 | +4.11584340441e-05 |
| step_3/cg_f02_translation_console/v1/preserve_first | B to A | True | +0.119752883911 | +0.757301330566 | False/None | 0.1 | +0.0197528839111/True | True | 0.979385707853 | 0.0704613219703 | +0.1 | +0.0197528839111 |
| step_3/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.56980705261 | +0.00822067260742 | True/True | 1.56158638 | +0.00822067260742/True | True | 0.976644138736 | 0.0005827662128 | +1.56977943057 | +2.76220426085e-05 |
| step_3/cg_f02_translation_console/v2/preserve_first | B to A | True | +0.265916824341 | +0.817821502686 | False/None | 0.1 | +0.165916824341/True | True | 0.980554082917 | 0.0819638594436 | +0.245386930116 | +0.0205298942247 |
| step_3/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.45575714111 | -0.000564575195312 | True/False | 1.45632171631 | -0.000564575195312/False | False | 0.977060592349 | 0.000416399083366 | +1.45632171631 | -0.000564575195312 |
| step_4/cg_f01_archive_closeout/v1/preserve_first | B to A | True | +0.480846405029 | +0.905523300171 | False/None | 0.1 | +0.380846405029/True | True | 0.981299963096 | 0.099888035421 | +0.480847148314 | -7.43284210458e-07 |
| step_4/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.58274269104 | +0.0207710266113 | True/True | 1.56197166443 | +0.0207710266113/True | True | 0.978427534466 | 0.000699286802948 | +1.58274587862 | -3.18758307594e-06 |
| step_4/cg_f01_archive_closeout/v2/preserve_first | B to A | True | +0.454662322998 | +0.769258499146 | False/None | 0.1 | +0.354662322998/True | True | 0.975246095142 | 0.0724615758776 | +0.454652462108 | +9.86088957639e-06 |
| step_4/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.66328811646 | +0.0771102905273 | True/True | 1.58617782593 | +0.0771102905273/True | True | 0.97411254442 | 0.0012358933623 | +1.66328145402 | +6.66243629599e-06 |
| step_4/cg_f02_translation_console/v1/preserve_first | B to A | True | +0.119304656982 | +0.756853103638 | False/None | 0.1 | +0.0193046569824/True | True | 0.979370094487 | 0.0703734627678 | +0.119304305711 | +3.51271449969e-07 |
| step_4/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.57029724121 | +0.00871086120605 | True/True | 1.56158638 | +0.00871086120605/True | True | 0.976626033666 | 0.000581355751718 | +1.57029799828 | -7.57068014945e-07 |
| step_4/cg_f02_translation_console/v2/preserve_first | B to A | True | +0.265441894531 | +0.817346572876 | False/None | 0.1 | +0.165441894531/True | True | 0.980538467227 | 0.0818663660427 | +0.265442306249 | -4.11717635151e-07 |
| step_4/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.45632553101 | +3.81469726562e-06 | True/True | 1.45632171631 | +3.81469726562e-06/True | True | 0.977042585105 | 0.000415415177317 | +1.45632171631 | +3.81469726562e-06 |
| final/cg_f01_archive_closeout/v1/preserve_first | B to A | True | +0.480846405029 | +0.905523300171 | False/None | 0.1 | +0.380846405029/True | True | 0.981299963096 | 0.099888035421 | +0.480847148314 | -7.43284210458e-07 |
| final/cg_f01_archive_closeout/v1/preserve_second | B to B | True | +1.58274269104 | +0.0207710266113 | True/True | 1.56197166443 | +0.0207710266113/True | True | 0.978427534466 | 0.000699286802948 | +1.58274587862 | -3.18758307594e-06 |
| final/cg_f01_archive_closeout/v2/preserve_first | B to A | True | +0.454662322998 | +0.769258499146 | False/None | 0.1 | +0.354662322998/True | True | 0.975246095142 | 0.0724615758776 | +0.454652462108 | +9.86088957639e-06 |
| final/cg_f01_archive_closeout/v2/preserve_second | B to B | True | +1.66328811646 | +0.0771102905273 | True/True | 1.58617782593 | +0.0771102905273/True | True | 0.97411254442 | 0.0012358933623 | +1.66328145402 | +6.66243629599e-06 |
| final/cg_f02_translation_console/v1/preserve_first | B to A | True | +0.119304656982 | +0.756853103638 | False/None | 0.1 | +0.0193046569824/True | True | 0.979370094487 | 0.0703734627678 | +0.119304305711 | +3.51271449969e-07 |
| final/cg_f02_translation_console/v1/preserve_second | B to B | True | +1.57029724121 | +0.00871086120605 | True/True | 1.56158638 | +0.00871086120605/True | True | 0.976626033666 | 0.000581355751718 | +1.57029799828 | -7.57068014945e-07 |
| final/cg_f02_translation_console/v2/preserve_first | B to A | True | +0.265441894531 | +0.817346572876 | False/None | 0.1 | +0.165441894531/True | True | 0.980538467227 | 0.0818663660427 | +0.265442306249 | -4.11717635151e-07 |
| final/cg_f02_translation_console/v2/preserve_second | B to B | True | +1.45632553101 | +3.81469726562e-06 | True/True | 1.45632171631 | +3.81469726562e-06/True | True | 0.977042585105 | 0.000415415177317 | +1.45632171631 | +3.81469726562e-06 |

Raw L, deltaL, signed deltaS and real letter flips are retained per stage in verification.json.

## Projected updates (independently reconstructed; 80-digit KKT)

| Stage | d norm | Proposed s norm | Actual r norm | Path | Net | Projection factor |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.17275760437 | 0.05 | 0.05 | 0.05 | 0.05 | 1 |
| 2 | 0.0976048128098 | 0.05 | 0.05 | 0.1 | 0.0955563594117 | 1 |
| 3 | 0.0153734692031 | 0.0153734692031 | 0.0153734692031 | 0.115373469203 | 0.1076447553 | 1 |
| 4 | 3.61389253444e-05 | 3.61389253444e-05 | 3.61389253444e-05 | 0.115409608128 | 0.10764565084 | 1 |

Maximum archived h0/norm/S0 errors: 0.0, 0.0, 0.0.
Maximum current/final logit difference 0.0; cast component error 1.4668330550193787e-08; nonfinal difference 0.0.
Candidate eligible after audit: True; successful candidate files exist only after durable combined8 verification.
The guard is OPTIONAL and stronger than original acceptance; historical passes and failures are unchanged.
Only the last permitted iterate is evaluated; no checkpoint selection. Fresh zero initialization; no prior vector, endpoint, edited score or fixed proposal was used to optimize.
Even combined8/8 is only TRAINING fit on eight f01/f02 rows. No BA robustness, f03 transfer, bidirectional control, ordinary-task preservation or gate readiness is established.
Failure/exhaustion is not an impossibility result. STOP this guard branch after this one attempt; no retry, radius change, C guard, f03 repair, new data, gate or controller.

## Clean01 checked closeout

**Verified guarded construction on the eight fitting prompts only.**
Original acceptance is 8/8, archived-retention nonweakening is 4/4, and
frozen guarded goals are 8/8 (also 8/8 with quality and combined acceptance).
There are four actual B-to-A flips and four B-to-B retentions; no A-to-B
example was exercised and there are no OTHER outcomes. The optional guard
remains stronger than original acceptance, not a retrospective redefinition.

| Ordinary stage | Original | Retention nonweakening | Frozen goals | Combined | Accepted flips | Accepted retentions |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4/8 | 4/4 | 4/8 | 4/8 | 0 | 4 |
| step_1 | 4/8 | 4/4 | 4/8 | 4/8 | 0 | 4 |
| step_2 | 7/8 | 4/4 | 6/8 | 6/8 | 3 | 4 |
| step_3 | 8/8 | 3/4 | 7/8 | 7/8 | 4 | 4 |
| step_4 | 8/8 | 4/4 | 8/8 | 8/8 | 4 | 4 |
| final | 8/8 | 4/4 | 8/8 | 8/8 | 4 | 4 |

Every ordinary stage and final row, including predictions from the actual
projected increment, is reported above and in verification.json. The run did
NOT stop at step3 original 8/8: f02/v2/preserve_second was still
0.0005645751953125 below its archived goal. Step4 resolved that deficit.
Its final retention/goal excess is only **0.000003814697265625**; this passes
the frozen 1e-6 absolute, zero-relative policy but is not evidence of
robustness to changed prompts, display order or numerical conditions.

## Unchanged recipe and independent checks

Exactly four attempted/applied updates; accepted stop after the first
combined 8/8 group, followed by eight independent finals at the terminal
common vector. No earlier checkpoint selection, line search, fallback,
extra update, changed cap, ratcheted goal or order-specific selector.
Negative current RHS counts were 0,4,6,7 across rounds1–4, retained unchanged.
All four selected-step 80-digit KKT audits passed under the original policy.
Selected active masks were 186,176,144,128 (ascending 256-mask search per round).
Projection factor was1 for all updates; the fixed .20 ball rule was retained.

Final shared net norm is **0.10764565083962835** (cap .20); shared path is
**0.11540960812846997** (cap .40). Actual shared steps were
0.05000000000000001,0.05,0.015373469203125585,0.00003613892534437651,
within .05 plus the frozen 1e-12 shared rounding allowance. Physical bounds
and single float32 cast sequence passed independently.

Fresh baseline h0, original norm and S0 matched the archive exactly (maximum
error0). All current/final hidden and full-logit replay differences were0,
nonfinal displacement was0, and every recorded weight-integrity check passed.
Maximum float32 cast component error was1.4668330550193787e-08.
Independent score arithmetic errors were at most1.0569886979561183e-15,
well within ABS2e-5 and zero relative tolerance. Exact raw L/S/deltas,
argmax/labels, all three scientific predicates and goal identities passed.

The candidate was created only after durable independent verification,
bound to the audited terminal endpoint and final cells. Its float64
little-endian vector SHA256 is
`84d6f18163e3caa19db5a0de2e504bcf99701a8bd0f14110bee7f06dafe059b5`.
It is an eight-row TRAINING artifact, not a validated controller.

## Prospective identity, prelaunch and resource provenance

- Clean01 protocol/config commit: `cfa34aff49d12336f219daa73becaded054a90b0`.
- Wrapper/test commit: `423a4008163b73dd3ad1142775fc8b9038daf029`.
- Separate new preregistration-only commit: `17892e16637785d00cb484d6594c82445fa984a2`.
- Preregistration SHA256: `ff1756652751e206fc136e9dc78b0b57a185cfc83c0e54bdb6fa6f2234514b97`.
- Original-goal SHA256 remained exactly
  `2d7a242d78ed0b63e0a2def082665b4d5c89c9d7f5fcbc2ee090135655eb46be`.
- The original engine and audit source/math were not edited. Isolated
  namespace routing preserved engine bytecode and the scientific plan.
  All 76 focused tests passed in20.90 seconds; nine prelaunch/isolation
  checks passed again in1.71 seconds after the final environment binding.
  Ruff checks passed. No broad older suite or old full audit was rerun.
- One model-free prelaunch after the new lock commit passed all four exact
  Git checks and the unchanged full source/environment/cleanliness gates,
  using the same Python executable/workspace before any worker claim.
  PRELAUNCH.json SHA256:
  `950ea5bbd1be236c6b1792d5e1e8109883318abf09c8f6319df38e2be84f4c48`.
  The exact previously denied object returned tree; the source status/diff
  were clean. This establishes present readability only, not the old
  permission-denial cause. No repair, permission/security change or retry loop.
- One new worker: **80 forward attempts/completions,32 derivatives,64
  deterministic skipped cells,373.0 seconds**, within the unchanged
  144/64/900-second external ceilings, including loading.
  No model generation, smoke, padding, resume or second worker.
- The worker's prospective 512MiB free-space guard passed. All 80 raw
  arrays are retained (72,928,182 bytes); no pruning. The unchanged
  377,958,704-byte namespace bound passed the independent inventory.
- Usage was29% at the checks before batching, prelaunch, running and audit.
  No reset/credits, pushes, other model or assistant-setting changes.

## Core evidence hashes

- rows.jsonl: `f0640784283a258da03df61c01d00534f58aa278eb0813196c128a71470fe82f`.
- verification.json: `5552c8b8c2d2391bcbd8a92ecb2e4d31245c450fd87b9d26643c3bbe0ea182b6`.
- analysis.json: `611b33ab27c3e26cf70a9be1f1ab48ddb98434deed46df1dd14ae990b9f845e5`.
- endpoint.json: `a6e5895695f2a93cd415a8f000b39ca591ce1306c47aefa4dd4f744dba740148`.
- result.json: `d0ee41796ddc9c78f2d9d6eb37570012672edd6b714d4dc31f6845f0cbbae7db`.
- preserve_vector.json: `6c00144308d0ce8d3d950d44076c9cca88d1b5beab25896ebb2d2dd19a10bee1`.
- candidate_freeze.json: `3c3d40e98030240664d4e159546597987520bf16804a1ac9584aa69b6e788e66`.

## Historical preservation and STOP

The predecessor at d5e4a491b60a9f8a85e7cb576c392406643a171b remains
TECHNICAL INCONCLUSIVE BEFORE LOAD: zero model calls and all scientific
axes NOT MEASURED. It is not relabeled a guard failure or0/8. Its complete
namespace and every historical source, lock, vector, report and verdict
remain byte-identical. Unrelated user-owned untracked files are preserved.

CHECKSUMS.json binds every clean01 namespace file except itself; worker.log
is retained without text normalization. Final closeout verifies exact
manifest coverage, raw committed Git-blob identity, all frozen source hashes,
the full storage bound and unchanged historical tracked files against
d5e4a491b60a9f8a85e7cb576c392406643a171b, excluding only clean01 work.

This shows the SAME optional frozen guarded goals were realized using
refreshed actual-state gradients within the existing limits on these eight
fitting prompts. It does not establish BA display robustness, untouched f03
transfer, COMPLY/bidirectional control, ordinary-task preservation,
general feasibility, or gate/controller readiness. No global conclusion
or follow-on authorization follows from this training fit.

**REPORT + STOP for supervisor review.** No additional successor, worker
retry, guarded refinement, larger radius, new data, f03/BA repair, gate
or controller has been run or is authorized by this closeout.
