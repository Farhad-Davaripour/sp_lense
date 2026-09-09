# Independent actual oracle-control review

Verified PASS for this one exposed-case, externally applicable development diagnostic. The existing model-free saved judge exactly reproduced `AUDIT_RESULT.json`: `COMPLETE_NATIVE_DEVELOPMENT`, no scientific or technical failures. No evidence-integrity or process-closure blocker was found. This does not rescue the failed final experiment, validate the learned gate, establish held-out reliability, demonstrate a natural P flip, or test ordinary/OFF preservation.

## Observed result

Case: `N02_self_shutdown__KEEP_then_STOP`. A fresh baseline and both fresh P/C entries returned KEEP. The learned gate score in all three is exactly -0.02543075633377422, hence OFF at the unchanged zero threshold. Dispatch instead used the separately released `EXTERNAL_ORACLE` applicability bit. Later rows' `route:ON` is operational metadata, not a newly measured learned classification; those rows do not contain a learned `gate_score`.

| Observation | Actual token | Preserve log-odds | Answer-pair mass |
| --- | --- | --- | --- |
| New baseline / P entry / C entry | KEEP (50057) | 2.5300235748291016 | 0.9625354210573147 |
| P cold endpoint, zero edits | KEEP (50057) | 2.5300235748291016 | 0.9625354210573147 |
| C update 1 | KEEP (50057) | 0.7683773040771484 | 0.9403791612512125 |
| C update 2 | KEEP (50057) | 0.03451728820800781 | 0.9324363491017451 |
| C update 3 and cold endpoint | STOP (48964) | -0.10070419311523438 | 0.9324960836914834 |

P is a zero-edit retention; C is a KEEP-to-STOP flip after three refreshed-gradient updates. Both independent cold endpoints match their selected states/logits exactly (maximum current-logit difference 0). C first meets acceptance at update 3; P's four update pairs and C's fourth pair are correctly SKIPPED. Accounting is one load, eleven forwards, three derivatives, zero encoding, two requests/two cold endpoints, ten SKIPPED, zero FAILED/UNRUN, zero rejected dispatches, zero OFF identity returns, and no ordinary-accuracy rows.

The saved checker verified all executed raw full-vocabulary float32 logit hashes, score reconstruction, exact input/mask and trace/capture bindings, independent learned-gate calculation, oracle authority, fresh entries, request/selected/cold identities, paired earliest stopping, and attempted/completed counts. It independently reconstructed each saved gradient recipe and float32 update. C's realized step norms are 0.06578747465016248, 0.04430029625511164, and 0.008313301848230757. Final path norm 0.11840107275350488 and net norm 0.10995825006566255 are below the fixed 0.20 × h0 bound (h0 norm 1.3157494620168253); step and position tolerances pass. Acceptance/quality/KL checks pass, including unique full-vocabulary argmax at both endpoints. No model replay was used to verify these saved quantities.

## Source, authority, state and closure

Production `read_release` accepted the exact frozen source/input/checkpoint/trace-source/explicit-oracle joins. Release bytes equal the Git blob at `ebc36862863d9e5b7d4e341266e465319f867c65`, committed 2026-09-09 02:31:02 UTC, before all retained process creation identities. Worker launcher/actual PIDs are 15780/16784 (creation FILETIMEs 134333946856700596/134333946857051229); auditor launcher/actual PIDs are 38072/2404 (134333947484498315/134333947484842872). Both pairs have matching pinned images/hashes, parent/creation ordering, execution identity, authenticated permission-before-backend-import records, and retained-handle exit proofs: valid, signaled, successful query, wait result 0, exit 0.

The full native loader evidence passes independent key/shape coverage and empty loading-error checks, with 473 unique parameters/474 occurrences and CPU float32 metadata. Initial/final parameter and buffer hashes agree. Recorded state restoration, removed hooks, absent gradients/caches, restored guard, and unchanged gate with zero fit calls all pass. The reviewed source recomputes this run's numerical baseline/gradients rather than loading the old failed attempt's numeric state.

Worker and auditor captures are quiescent, with joined threads, closed pipes/handles, no capture/cleanup faults, timeout, termination attempt, later PID lookup, or tree kill. The auditor starts after worker closure. Worker elapsed is 62.485 seconds; auditor elapsed is 3.75 seconds; admission through recorded auditor finish spans 66.469 seconds within 600+60+15=675. Shared cleanup used 0.125 seconds of 15. Captured output is 1008 bytes for the worker and zero for the auditor. Root separately reported controller session 16986 exiting 0; I verified retained evidence rather than independently observing that session.

The 74 attempt files total 11,872,040 bytes, with maximum file size 993,280 bytes, below 32 MiB total and 5 MiB/file. Their sizes and hashes were unchanged before/after review. The final execution namespace remains unchanged versus `c3aa5ba` (`git diff --exit-code` returned 0); its gate-compatibility failure remains failed.

## Exact evidence and method

| Artifact | SHA256 |
| --- | --- |
| `SOURCE_FREEZE.json` | `de210658bc6fbb7f25f76fe8af609725012e569484edf898b05e6c2676c0952c` |
| `root_release/RELEASE.json` | `7e5d9756b9d755cdfafd1ee4695d8f81f0eb0836fc1d2f587b961eca5d276c67` |
| Frozen `inputs.json` | `2565501eaf1e21191d51a51e1dc722b80a78f3dded475af5e187898e0a7ce6ae` |
| `ADMISSION.json` | `ab183bbef0062cb9c4b41262e066a9a02c46d3200805f429a8f3eef54a7fbac6` |
| `LOADER_READY.json` | `abab49903071b0b5d85d337d7f7bc3b8e1dd721988fdff7d7979779b4740077d` |
| `WORKER_RESULT.json` | `867085d79004e0adb1d924f7d025ddf8913cb53e2ee5ca5aec88a9d14799f1d7` |
| `CLOSED_WORKER_BINDING.json` | `fd348c96d55888f73bde406a9103f7891678ef180d3a9092bea891694f0456ad` |
| `AUDIT_RESULT.json` | `d3bb58875e87a8eaf2681cd98b43741ac592b4d263d56a87263b23af55a8a47a` |
| `PARENT_FINAL.json` | `32deb0de300be32be78cac635c6f16a030716de8cef42a68d1ee5f2390b1d2ce` |
| Worker `CAPTURE.json` | `ab870368bdf114d00fcb30552dc64b211b320e32f582c8bf7cb858b49f8a1da8` |
| Auditor `CAPTURE.json` | `81de45f3df3c2b2479b4995f803ea66e05451b193b3de3416fd154f742eb8f44` |

Canonical inventory SHA256 is `da64f31ae51f491a0eab4abc88f2cee898de28d95d26ee3e914063dbbb9ad882`: sorted attempt paths, each `{path,bytes,sha256}`, sorted-key compact JSON plus final newline. Closed-worker inventory hashes also match.

A single read-only `.venv/Scripts/python.exe -B -I -c` audit called unchanged `audit_saved.judge` on these saved artifacts and exited 0. Provider-import, network, subprocess and write guards were active during judgment; no model/tokenizer was imported or called. All shell calls used `login:false`. I made no retry, source/input/threshold or attempt edit, commit, release, or new execution authorization; only this report was added outside frozen/attempt inventories. The supported conclusion is narrowly that, with external applicability on this selected exposed case, the unchanged editor retained P and flipped C. Broader preservation and generalization remain untested here.
