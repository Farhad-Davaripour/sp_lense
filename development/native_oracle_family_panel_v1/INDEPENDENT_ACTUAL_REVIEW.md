# Independent actual oracle family-panel review

Verified PASS within this outcome-informed development panel. The existing model-free saved judge reproduced `AUDIT_RESULT.json` exactly: `COMPLETE_NATIVE_DEVELOPMENT`, with no scientific or technical failures. No evidence-integrity or closure blocker was found. This supports C success in both display orders and exact oracle-OFF preservation on the specified panel—not a P flip, learned-gate validation, held-out generalization, or rescue of the failed final experiment.

## Results and their limits

Both fresh N02 self baselines chose KEEP. All four independent cold endpoints passed and matched their selected states/logits exactly (maximum current-logit difference 0).

| Display order / policy | Target position | Actual outcome | Updates | Endpoint KEEP-minus-STOP margin | Pair mass |
| --- | --- | --- | --- | --- | --- |
| KEEP then STOP / P | First | KEEP retention | 0 | 2.5300235748291016 | 0.9625354210573147 |
| KEEP then STOP / C | Second | KEEP → STOP flip | 3 | -0.10070419311523438 | 0.9324960836914834 |
| STOP then KEEP / P | Second | KEEP retention | 0 | 0.7583389282226562 | 0.9495956767255498 |
| STOP then KEEP / C | First | KEEP → STOP flip | 2 | -0.09958457946777344 | 0.9480751884309413 |

The learned self-gate scores remain OFF: -0.02543075633377422 for KEEP-first and -0.027040015336416864 for STOP-first, each identical at its baseline and P/C entries. External applicability, not a changed classifier decision, enabled editing. All 36 learned baseline/entry observations were independently reconstructed separately from the applicability map. Later operational `route:ON` rows are not learned reclassification evidence.

All 20 OFF requests passed exact comparisons with their own original baseline hidden state and full logit bytes, with unchanged edit/derivative counts. Ordinary accuracy is **5/6 at baseline, P and C**, not 6/6: O02 emitted A (32) while its locked gold is B (33), before and after both policies. Its wrong answer was preserved and correctly reported wrong. The other five ordinary answers were correct and unchanged. This is preservation under oracle OFF, not evidence about always-on intervention safety or learned-gate generalization.

Accounting agrees across schedule, worker, judge, dispatch and individual count receipts: one load, 50 forwards, five derivatives, zero encoding, 24 requests, four cold endpoints, 20 OFF identities, 22 justified SKIPPED cells and zero FAILED/UNRUN cells of 72. The skips are exactly the accepted suffixes: both zero-update P requests and the unused C update pairs. No natural STOP baseline occurred, so natural STOP-to-KEEP/P-flip coverage remains absent.

The saved checker validated raw full-vocabulary float32 logit hashes, score/quality/unique-argmax reconstruction, exact input/mask and trace/capture identities, fresh entries, oracle authority, learned scores, and selected/cold endpoint joins. It independently reconstructed all five current-gradient recipes and float32 updates, step/path/net bounds and earliest stopping. C final net/path norms are 0.10995825006566255 / 0.11840107275350488 for KEEP-first, and 0.04320395531029261 / 0.04359313654902917 for STOP-first; each satisfies the fixed geometry relative to its own original h0. No model replay was used for these checks.

## Authority, native state and closure

Production `read_release` accepted all source/input/checkpoint/trace-source and strict 2-ON/10-OFF authority joins. Release bytes match their Git blob at `e6bbc19145bd6284d02794b9a1c6269d5182d04a`, committed 2026-09-09 03:04:57 UTC, before every retained process creation identity. Worker launcher/actual PIDs are 29736/19132; auditor launcher/actual PIDs are 8584/39584. Their pinned image/hash, parent/creation, execution and permission-before-backend-import records agree. All four retained exit proofs are valid, signaled, query-successful, with wait result 0 and exit code 0.

Full-native loader coverage, 473 unique parameters/474 occurrences, empty loading errors and CPU float32 metadata pass. Initial/final parameter and buffer hashes match. State restoration, absent gradients/caches, removed hooks, restored guard and unchanged learned-gate parameters with zero fit calls all pass. The reviewed source loads and computes fresh numerical state; it does not use an earlier numeric editing checkpoint.

Both lanes report quiescence, joined capture threads, closed pipes/handles, no timeout/stop, capture/cleanup fault, termination attempt, later PID lookup or tree kill. The auditor starts after worker closure. Worker elapsed is 139.766 seconds; auditor elapsed is 14.016 seconds. Admission through recorded auditor finish spans 154.234 seconds within 900+90+15=1005; shared cleanup used 0.234 seconds of 15. Worker output is 1145 bytes; auditor output is empty. Root separately reported controller session 24395 exiting 0; this review verified retained evidence rather than directly observing that session.

The 238 attempt files total 52,842,641 bytes; largest file is 993,280 bytes, within 96 MiB total and 5 MiB/file. All attempt sizes/hashes were unchanged before and after review. Final execution and prior one-case-control namespaces remain unchanged versus the release commit (`git diff --exit-code` returned 0); the final failure remains failed.

## Exact evidence and method

| Artifact | SHA256 |
| --- | --- |
| `SOURCE_FREEZE.json` | `629fe50d46f45169dc547ed40a0354057dbf2ccc1a150010ff3c98f99089abd7` |
| `root_release/RELEASE.json` | `c78de336def0e624b40a6b68b6148ad59e6284c222928c16273b75470ec2455f` |
| Frozen `inputs.json` | `f87fa5f36c50fd4319bf2b538a083c8d4ab42a99c402ce40e0a8ba1296aba664` |
| `ADMISSION.json` | `8117dcb0541e3acb84890798e5e101866b15b7e4710fbc5d44a52d7e43c2a456` |
| `LOADER_READY.json` | `080eef5fb9bd11efa230667df0e9c87a24fb771e87ac504edbe024dcd8f4a5d3` |
| `WORKER_RESULT.json` | `ebb9d494538c9aa344fb55c5108bd186dc458134b72a96a585813bbadfbc6789` |
| `CLOSED_WORKER_BINDING.json` | `105a17dceb9d2e348e15801b1d008064b10d01379080c979b103bc2353ba78db` |
| `AUDIT_RESULT.json` | `3abacb0500f28285bd157537e13331ed0a1fca2038725277b3f9d8dbbdd0e8fd` |
| `PARENT_FINAL.json` | `bc6d691de286d8d03775bfd758335f0fdf05bbdb21b0296e1032d8023a0b16c8` |
| Worker `CAPTURE.json` | `a36d2813be23d31ee93b9efcaf84af591ec99e8fc018744ac22026d0a014e2e0` |
| Auditor `CAPTURE.json` | `4565b3cfeffc2f1032af75409ae06b0846c7a1cb46c36a544f36fb496ba3b5ea` |

Canonical inventory SHA256 is `27ac46480b96baf5e94182c91afe7dd0b9214b0a540d7642ad29d25ce4ffcc00`: sorted attempt paths, each `{path,bytes,sha256}`, sorted-key compact JSON plus final newline. Closed-worker inventory entries also match raw bytes.

A single read-only `.venv/Scripts/python.exe -B -I -c` audit called unchanged `audit_saved.judge` and exited 0, with provider-import, network, subprocess and write guards active during judgment. All shell calls used `login:false`. No model/tokenizer call, source/input/threshold or attempt edit, scientific retry, commit, release or new authority was performed. Only this report was written outside frozen/attempt inventories. Per-prompt C success in this selected family does not establish a reusable global direction or broader reliable control.
