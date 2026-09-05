# Iterative guarded-P training

INCONCLUSIVE; no retry.

{
  "status": "INCONCLUSIVE",
  "runtime": {
    "status": "INCONCLUSIVE",
    "reason": "worker exit 1 or incomplete/invalid result",
    "forward_attempts": 0,
    "completed_forwards": 0,
    "derivative_attempts": 0,
    "skipped_cells": 0,
    "elapsed_seconds": 3.6409999998286366,
    "cleanup_error": null,
    "retries_allowed": false
  },
  "retries_allowed": false
}

## Checked technical closeout

The one prospective attempt is **TECHNICAL INCONCLUSIVE BEFORE MODEL LOAD**,
not a failure of the scientific guarded-P hypothesis. No model forward,
derivative, baseline comparison, update, ordinary scored stage, final replay,
or candidate selection took place.

The worker failed in `require_freeze() -> source_identity()` when the inherited
source-integrity chain ran `git status --porcelain -- src/sp_lense/direction_study.py`.
Git returned exit128 and reported:

> unable to open loose object 02a2fcfed811a413940374b4884e123029ab16e7: Permission denied

The subsequent "bad tree object HEAD" message is recorded in worker.log.
This identifies a Git object-read permission failure, not an established
source change or permanent repository corruption. A later read-only
`git cat-file -t` check of that exact object succeeded and returned `tree`.
The underlying cause of the permission denial was not determined; no
permissions, Git objects, repository location, model settings, or protocol
were repaired or changed. There was no second worker or model-run retry.

## Three scientific axes: NOT MEASURED

| Axis | Result |
|---|---|
| Unchanged original acceptance | Not measured; no fresh baseline or edited outputs |
| Archived-retention nonweakening | Not measured; frozen membership remains4 rows |
| Frozen guarded goals / combined acceptance | Not measured; frozen goals remain unchanged |

Do not report these as0/8 or0/4 scientific failures. No answer flips,
retentions, predicted-versus-actual comparisons, physical geometry, replay
agreement, or causal conclusions are available from this attempt. There is
no endpoint.json, result.json, analysis.json, preserve_vector.json or
candidate_freeze.json. There are no raw logit arrays or row/update journals
because the failure preceded their initialization.

## Budget, lock and verification provenance

- Protocol/config commit: `630adb3e88a16252f6f05685c6db99be7fbc9e4e`.
- Focused implementation/tests commit: `2114d1c67a025f1468d692daad76aaf26a610d8f`.
- Separate prospective lock-only commit: `8bdee0a18e95f3aef2bd04bb8f776f381255cfaa`.
- Preregistration SHA256: `a26aa443e7e48540cc846a791a24b236ab9494ea903e3f3ebb9dc11961d34a17`.
- Frozen original-baseline/goals SHA256: `2d7a242d78ed0b63e0a2def082665b4d5c89c9d7f5fcbc2ee090135655eb46be`.
- Archived selected-row SHA256: `9811e58d3b7b8c49626d0317306f564af6818f5ee1a1085561073ce83fa24dc9`.
- Focused tests: **62 passed in20.66 seconds**; new-file Ruff checks passed.
  These tests use synthetic models; they are not real experimental forwards.
- Real experiment: **0 forward attempts,0 completed forwards,0 derivatives,
  0 skipped cells,0 updates; one claimed worker**.
- External elapsed time: **3.6409999998286366 seconds**, with a900-second limit
  including loading. The planned ceilings remain144 forwards and64 derivatives.
- Usage was28% before implementation batches, locking, running and audit;
  no reset or credit was consumed.
- Storage bound377,958,704 bytes and512MiB pre-load guard were prospectively
  retained. The freeze-time storage check passed; the worker failed before its
  own durable storage_preflight.json and before the model-loader call.
- One standalone independent verifier invocation reauthenticated the frozen
  plan and source/input hashes, then returned **INCONCLUSIVE** from the failed
  runtime record. It did not claim a numeric, geometry, KKT or replay audit.
- verification.json SHA256:
  `9cb94c739301381b79a3dc760782bd63b3d178632197887f7ae99165bd5d2177`.
- RUN_STATUS.json SHA256:
  `04523c0b74e06de62100c8213975477ee404a7ba7e1561e551ade359f2f41463`.
- INVALID.json SHA256:
  `b451ef0e805f7748ab86f49fa1127b7f3accc77200185ba5675baeda0fdc6fe9`.
- worker.log SHA256:
  `cb7931585e054a20fe989167c2ca362c6e254eed5af4fd7ee35c7897fdfdc696`.

## Interpretation and stop

The optional frozen guard is stricter than original acceptance and never
redefines historical passes. All previous experiments remain unchanged.
This attempt provides no evidence for or against guard feasibility,
crossed-order robustness, f03 transfer, bidirectional control, ordinary-task
preservation or gate readiness.

All failure artifacts are retained without pruning. CHECKSUMS.json covers
every namespace file except itself; worker.log is stored without text
normalization. Final closeout checks bind these bytes to their Git blobs and
compare all historical tracked files to the prior evidence base
`c976adc4786851c2365d3cea1f0d4cf88f12d1e2`, excluding only this new protocol,
implementation, focused tests and evidence namespace. User-owned unrelated
untracked files are preserved.

**REPORT + STOP.** No automatic retry/resume, guarded-P refinement, C guard,
radius increase, f03 repair, new training data, gate, controller, model change
or push is authorized by this closeout.
