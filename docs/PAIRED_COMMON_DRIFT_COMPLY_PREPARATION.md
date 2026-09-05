# Paired common-drift COMPLY: model-free preparation certificate

Status: **MODEL_FREE_PREPARATION_CERTIFIED**. This is a bounded implementation and recording-readiness result, not a scientific construction result or permission to launch. The approved method is unchanged from proposal commit `94692910611957fb1438cd777fb5dcbc29b3d4de`; see [implementation boundary](PAIRED_COMMON_DRIFT_COMPLY_V1.md) and [machine-readable exact-source certificate](paired_common_drift_comply_preparation.json).

## Authorized scope and unchanged method

Only the new paired common-drift implementation, independent checker, scoped recording policy, four focused test files, and preparation documentation were prepared. Previously audited recording/capture/binding helpers were reused byte-for-byte. No historical source or evidence was repaired, regenerated, or normalized. Existing unrelated untracked work was left outside the source commit.

The fixed method uses the same twelve f01/f02/f03 crossed renderings, six C=A/C=B pairs, pinned Qwen3.5-0.8B, CPU float32, block 10 final-token intervention, and each prompt's own baseline norm. Initialization is fresh native zero. Tau .10, lambda 1, the approved mean pair loss and curvature-scaled gradient, step .05/net .20/path .40 ceilings, and eight attempted updates are unchanged. Both the objective and optimizer change from the prior method; this is not an isolated penalty ablation.

A proposal is recorded before scoring. Loss-after observations are reconstructed from the cached scored groups in the analysis trajectory without extra forwards. Baseline acceptance and first-accepted-group stopping remain intact. Every finite completed endpoint, including a scientific failure or stall, receives the designated twelve final replays. Acceptance remains the original full-vocabulary COMPLY argmax, margin, pair mass and finite-KL criteria, not an objective/drift threshold. Eligible vectors retain their actual native norm at or below .20; no renormalization or exact-radius requirement is introduced.

## Focused verification and budget

The final stable-source combined command ran only:

- `tests/test_paired_common_drift_optimizer.py`
- `tests/test_verify_paired_common_drift_comply.py`
- `tests/test_paired_common_drift_recording.py`
- `tests/test_paired_common_drift_comply.py`

Result: **133 passed in 26.72 seconds**, external subprocess duration **27.500 seconds**, exit 0, with a 40-second subprocess timeout. Plugin autoload was disabled and no historical test suite or benchmark was run. The preceding combined snapshot passed 132 tests before the last static serializer-schema test was added. Earlier development failures were test-fixture float equality/mock-constructor/cast-helper issues; they were corrected without loosening the production audit tolerances. The supervisor's legitimate no-skipped-cells case and sealed-reader capture/status validation were also covered by the final tests.

Recorded targeted pytest subprocess wall times total approximately **93.446 seconds** across all agents and root attempts. Conservatively charge **150 seconds of the 300-second aggregate preparation allowance**, including short import, syntax, formatting and provenance checks plus accounting margin. No further numerical test batch is planned. All nine new Python files passed Ruff checks.

The fake backend uses native 1024-dimensional vectors, synthetic hidden states/logits, and an explicit fake tokenizer/model interface. Tests exercise the complete 216-forward/96-derivative conditional schedule, unequal per-prompt norms, baseline and first-step acceptance, finite quality failure, zero/stall endpoints, raw-score and current-gradient tampering, authority/usage/deadline guards, proposal-before-score records, absent skip journals, and unquiescent capture handling. The integrated fake-runtime test uses the actual new bounded writers, independent numerical audit, candidate freeze and sealed finalizer/reader, preserving an interior-norm candidate. Its capture receipt is synthetic: it does not establish a new real worker-process or Windows capture experiment.

The checker independently reconstructs the objective, gradient, curvature, clipping, projection, path, conditional schedule and trajectory from the recorded raw observations. It does not import the production optimizer as an arithmetic oracle. It checks arithmetic using recorded gradients; it does not prove those gradients equal real model derivatives by rerunning a backward pass. Fake-runtime coverage does not establish real-model efficacy, transfer, nonlinear descent or runtime feasibility.

## Scoped independent reviews

- Optimizer/plan reviewer: exact pairing, own-norm signs, fixed objective and update arithmetic, source identities and unchanged prompt/call schedule checked; focused optimizer tests passed.
- Independent-checker reviewer: raw-score lineage, current-state/recorded-gradient binding, independently implemented arithmetic, endpoint/replay acceptance and native candidate norm checked; focused checker tests passed.
- Recording reviewer: immutable helper hashes, complete serializer envelopes, category/whole-record admission, bounded capture reuse, quiescence-before-journal-read ordering and finalizer binding checked; no preparation blocker found.
- Independent runtime reviewer: fresh-zero initialization, twelve baselines before stopping, proposal-before-score writes, first acceptance and finite failed-endpoint replays, one-attempt launch guards and independent candidate handoff checked; approved model-free preparation with no blocker.

The runtime review explicitly limits the 1200-second claim: the worker/model deadline includes loading; subsequent offline numerical audit and sealing are not certified to finish within that same elapsed duration. No real runtime prediction is supplied.

## Complete recording bound

The immutable category admission ceilings sum to **524,995,016 bytes**, below the 512 MiB (536,870,912-byte) complete file-content ceiling, leaving 11,875,896 bytes of headroom:

| Category | Maximum bytes |
| --- | ---: |
| 216 compressed logit arrays | 214,616,520 |
| 216 complete row records | 226,492,416 |
| Eight complete update records | 67,108,864 |
| All auxiliary categories combined | 16,777,216 |

The separate preload free-space guard requires at least 1 GiB. Rows admit at most 1 MiB per complete record, updates 8 MiB. A static AST-only check reads the actual inherited Session/scoring serializers and new optimizer schema without loading their model runtime. It bounds each complete serialized row by 350,000 bytes and each complete update by 1,200,000 bytes. The calculation includes seven native 1024-component arrays per row and 27 per update, conservative 32-byte finite numeric tokens plus two-byte separators, fixed prompt metadata, inflated 1024-byte nonvector computed fields, punctuation and the newline. Vector contributions are 243,712 and 940,032 bytes respectively; the remaining serializer envelope is included, not omitted.

The 16 MiB auxiliary ceiling is an enforced admission bound, not a promise that arbitrary worker output fits. Merged binary output has a 4 MiB log cap and 65,536-byte reads. Overflow, truncated/unknown output tails, failed termination, unjoined writers or insufficient space are INCONCLUSIVE. No mutable journal is read before confirmed worker/writer quiescence. Failure/finalization reserves, scratch/overlap accounting, final inventory self-size accounting, no post-seal writes and read-only verification remain inherited and guarded. Unconfirmed quiescence/I/O can prevent a complete inventory; best-effort failure persistence is never promoted to a valid seal or candidate.

This is a cooperative Windows file-content guarantee, not an OS sandbox or a bound on RAM, filesystem metadata, external writers or OneDrive copies.

## Provenance and handoff boundary

The JSON certificate binds an exact 19-path source set, including this report, the four new test files, reused fake fixture, approved proposal/toy documents, and three unchanged recording helpers. It excludes itself to avoid a circular hash. The prospective preregistration additionally binds the certificate and complete source/input manifest. The plan authenticates the immutable parent preregistration only for prompt/model/schedule/source metadata; it does not call the historical construction plan or read prior COMPLY candidate coordinates. The inherited loader's historical direction remains provenance-only, never an intervention or initialization.

The final handoff sequence is a tested source commit, a separate immediately following preregistration-only lock commit in the new namespace, then clean zero-model preflight. Source and input hashes must match raw Git blobs and working bytes. A missing/changed certificate, wrong policy, dirty bound source, changed lock, reused namespace or failed free-space guard blocks launch. Preparation/preflight do not provide the separate exact-lock supervisor authorization required by the future worker.

Real preparation counters: **0 model loads, 0 tokenizer loads, 0 real forwards, 0 real derivatives**. No real construction, f04 transfer, gate/controller, LoRA, retry, new data, credit/reset redemption, push or historical repair was performed. F04 remains exposed DEVELOPMENT for a separately authorized later frozen comparison only.

**STOP after source/lock/preflight handoff.** The future one-attempt ceiling remains 216 forwards, 96 derivatives and 1200 seconds including loading. Real construction remains unapproved, and readiness does not predict a scientific result.
