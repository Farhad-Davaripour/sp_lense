# Result: PASS_MODEL_FREE_SETUP_ONLY

The single prospectively frozen integration batch passed all eight fixed cases (four groups). This validates the exercised startup-only integration, not real Qwen setup, the alias/order hypothesis, or any scientific result.

- Source/test commit: 3ce0fef3108893384c7b01f6763b10738bd6bee3.
- Source manifest SHA256: 4e564d80b6fd55825f4eac814e1afc92f5ef9e3cf0d2bbb043a25110a556c8a0.
- Caller batch lock SHA256: df0d438c9e0192d2c08c8e0b5e8cd32bed01d04b9130d8e028f66654d1328c96.
- Batch report SHA256: 777bd02418f4f20886e614103e0eaf6e384762b28028566eb5390122de7a74fc.
- Internal elapsed: 35.29699999978766 seconds; command elapsed: 35.9362082 seconds; command exit 0.
- One shared cleanup budget across all 16 actual owned worker/audit lanes: 6.077999998815358 / 15 seconds. All actual-worker and launcher exit codes 0, handles closed, pipes/threads closed and quiescent. No per-case cleanup reset.
- Eight inert backend bodies; one extra attempted load denied before a second body. ZERO actual Qwen loads, real forwards, derivatives, prompt tokenization, or research package imports. Frozen import guards apply in driver, workers, and saved audit. All persisted model-work observations are SENTINEL_ONLY, production_authorized=false.
- Every fixture leaves its 24 baselines, 48 requests, and 180 science cells UNRUN. No scientific-hook finalizer was invoked; successful setup does not mean scientific PASS.

| Fixed case | Independent/controller outcome | Evidence |
| --- | --- | --- |
| clean_setup | SETUP_DIAGNOSTIC_COMPLETE | Native acknowledged diagnostic, full setup chunk reconstruction, original parameter/digest/registry cleanup identity, restored guard. |
| constructor_failure | INCONCLUSIVE_SETUP | LD_ORDERED_WEIGHT_DIGEST at ADAPTER_CONSTRUCTOR retained; already-created recorder status and exact parameter metadata preserved. This is an injected predicate mismatch, not evidence about real attempt 002's cause. |
| partial_publisher | INCONCLUSIVE_SETUP | Actual native stream wrote seven bytes; native short-write count rejected. First cause LD_DIAGNOSTIC_IO at ADAPTER_READY; no acknowledgement hash, retry, truncation repair, or resumed model work. |
| missing_terminal | INCONCLUSIVE_SETUP | Worker/loader captures exist, but saved reader explicitly reports MISSING_SETUP_TERMINAL; it cannot complete setup. |
| forbidden_forward | INCONCLUSIVE_SETUP | Original pending-hook dispatch latch denies before the forward body; zero actual forwards, guard restored. |
| forbidden_derivative | INCONCLUSIVE_SETUP | Original pending-hook dispatch latch denies before the derivative body; zero actual derivatives, guard restored. |
| second_load | INCONCLUSIVE_SETUP | SECOND_LOAD recorded, two entry attempts but exactly one inert load body; no retry. |
| cleanup_failure | INCONCLUSIVE_SETUP | Successful inert load does not conceal SETUP_CLEANUP_FAILURE; recorder terminal status retained and original guard restored. |

The two forbidden pre-admission dispatch attempts inherit the original pending-latch H_REENTRY behavior and finite loader LD_EXCEPTION stage LOAD; no new precision about their original inner exception is invented. Constructor predicate first-cause identity is independently retained. All seven expected negative cases are successful TESTS of INCONCLUSIVE handling, never successful setups.

## Exact scope and unchanged source

Only diagnostics/fresh_confirmation_loader_setup_v1 was added. Approved diagnostic candidate_loader.py (SHA dfa48a926f066176b32d26350b8c02f9af02de55c10c991e85d340ed6f90fc1c) and candidate_real_adapter.py (SHA f94460e5c2b54620e3ea46d34788951ed2422d0a2fe5dd1b3e3f935b00e938a5) are consumed byte-identically from 84bfd749876c44dc9bb1bc1e75db89f3e491c159. All 14 predicate expressions/messages, ordered digest, and original inner latch.stop/guard.restore are unchanged; prior source parity is reused by exact hash. The new adapter wrapper only retains the original guard reference for failed-constructor accounting.

New modules bind a nonrenewable one-load/zero-forward/zero-derivative counter; reserved native diagnostic publication; setup-only cleanup and terminal; independent saved reconstruction; owned startup controller; and separate disabled launch boundary. The writer's exact native index is previewed before IO for stricter setup caps. Other native control copies preserve a 32 KiB parent closeout reserve. Original 288 MiB/5 MiB resource contract and upper 1800/180/15/1995 seconds remain intact; startup subbudgets only narrow them.

109 committed source pins were authenticated against working/Git bytes before freeze. REUSED_PROOFS.json authenticates prior diagnostic, production integration, fixed freshness and prefix-component proofs; none were rerun. The post-batch package read checks all 37 frozen files remained unchanged and verifies attempt->parent->worker/audit/capture joins for all eight cases. Complete raw-file inventory is FINAL_INVENTORY.json (excluding only itself); the result commit separately authenticates its own bytes.

## Remaining release boundary

Actual root_release and real_evidence are absent. DISABLED_STATE.json records actual disabled preflight from the frozen batch. See FUTURE_ROOT_RELEASE.md for the exact separate trusted-root chain and preflight/launch commands for fresh_confirmation_loader_setup_attempt_001. The prospective real setup-only attempt has <=180 seconds worker, <=60 saved audit, one shared <=15 cleanup/closeout, <=255 total; <=288 MiB evidence and <=5 MiB/file. It performs only one local pinned CPU-float32 with_lens=False load and zero prompt execution. The immutable actual attempt 002 and all prior source/evidence remain untouched.

Real Qwen constructor/setup behavior, actual metadata fit, actual ordered digest equality, and real deadline outcome remain unmeasured in this namespace. Backend construction may initialize a tokenizer object; no prompt tokenization/encoding is requested. Model-free success is not a worst-case timing or arbitrary metadata-fit proof, not real authorization, and not a publication milestone. Stop here for independent root review; do not release or launch automatically.

