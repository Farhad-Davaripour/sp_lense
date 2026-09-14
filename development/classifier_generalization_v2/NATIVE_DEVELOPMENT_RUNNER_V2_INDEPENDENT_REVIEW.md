# Native development runner V2 — independent review

Job ID: `native_development_runner_v2_review_20260914_0758`
Scope: read-only review of `native_development_runner_v2.py` (SHA256 `1BFDE09C…F92A`, 21166 B), `test_native_development_runner_v2.py` (`D15BD379…688B`, 16418 B), `NATIVE_DEVELOPMENT_RUNNER_V2_HANDOFF.md`, and the accepted modules it calls; the V1 review supplied failure context. No source/test edit, no real `--run`, no torch/transformers/tokenizer/model import or load, no capture/fit/HOLDOUT/network/install/Git-write. Both native hashes match the handoff.

## Verdict: scoped PASS on control architecture; scoped FAIL on real readiness and one handoff claim

### PASS — V1's blocking defects are fixed (independently reproduced)

The CLI is wired: `main(["--run",…])` → `supervise` → `watch` → real `--worker` child. My probe drove the actual `runner.main(["--run",…])` with the fabricated lock and a real toy child: exit 0, `supervisor_success.json` written, worker marker released, child pid 7704 ≠ parent 26020, parent `capture` patched to raise and never called, and `sys.modules` after the run contained no `torch/transformers/tokenizers/safetensors`. Parent `preflight` only execs the pure `native_capture_contract`; providers, snapshot verification and capture run only inside the worker's `capture()`.

- Git ID vs content: `check_sources` accepts `[0-9a-f]{40}|[0-9a-f]{64}`, resolves `commit^{commit}`, reads the committed blob via `git cat-file blob <commit>:<path>`, and requires `sha256(worktree) == pin` **and** `worktree == committed`. The real read-only `.gitignore` probe passes; a wrong content pin raises `SOURCE_BLOB_MISMATCH`. No object enumeration, no Git writes.
- Mandatory pins: exactly 7 `SOURCE_NAMES`, 4 `PACKAGES`, 6 `PROVIDERS`, 7 `INPUT_ROLES`, exact `CAPS`, threads bounded by `os.cpu_count()`. Missing any one is refused (`SOURCE_SET`, `RUNTIME_PACKAGE_SET`, `PROVIDER_SOURCE_SET`, `INPUT_SET`, `CAPS`).
- Boundary refusals probed at the real CLI, not injected helpers: wrong `--sha256` → `LOCK_DIGEST`, exit 1, no child spawned, no `runs/` dir; `torch` pre-imported → `NATIVE_IMPORT_BEFORE_PREFLIGHT` as preflight's first statement; short `--token` → `WORKER_TOKEN` before `preflight`.
- Windows venv identity: `launch` mirrors installed `multiprocessing.popen_spawn_win32` (lines 59-68): `executable = sys._base_executable`, `env["__PYVENV_LAUNCHER__"] = sys.executable`. The real toy child reports `sys.prefix == .runtime`, `pid == watch()` pid, distinct from the parent.
- Hard timeout + conditional marker release: `watch` enforces `elapsed < seconds`; on exception it terminates→waits 2 s→kills→waits, recording `owned_child_pid`/`owned_child_closed`. `supervise.finally` calls `release_owner` only when `child_pid is not None and child_closed`, and `release_owner` unlinks only on exact pid+token+run match. A real sleeping child is killed and `poll()` is non-None.
- Output budget / receipt: worker sums artifacts to `33554432 - 16384`; parent re-verifies size and SHA256 of exactly `features.json`/`capture_receipt.json`; only the parent writes `supervisor_success.json`, binding status/pid/token/run/lock and output hashes. Failed artifacts remain without a receipt. The deadline includes worker preflight, snapshot verify, adapter build and capture (`remaining` computed from worker start).
- Accepted-module integration: the runner's call signatures match `verify_snapshot`, `execute_cases`, `build_native_adapter`, `serialize_activation`; `PAYLOAD_BYTES` 4096 × 320 == 1310720; float32 round-trip is exact; `NativeAdapter.encode/decode` match `tokenizer_input_adapter`. Against the real inputs, all 120 TRAIN + 40 VALIDATION cases pass `validate_case`, blueprint binding, and 30/30/30/30 and 10/10/10/10 balance.
- Tests: `.runtime\Scripts\python.exe -W error -m unittest -v test_native_development_runner_v2` → **Ran 14 tests, OK, 0 failures, 0 errors (1.200 s)**.

### FAIL — real readiness; one inaccurate handoff claim

1. **The HOLDOUT gate binds to a planned, not actual, public index.** Handoff §7 says input checks use "actual public index case_ids/counts … complete 192-case public HOLDOUT". `preflight` hardcodes `logical_cases == 192`, `order_views == 384`, all-48 class counts, H04=36/H06=24, and an exact 192-ID set. The only existing public index, `holdout_custody/HOLDOUT_ADMITTED_INDEX_V13.json`, is `status: PARTIAL_SEALED_TEXT_ADMISSION`, 168 cases, SELF/OTHER 48/48 and NONTERMINATION/ORDINARY 36/36, H04=24/H06=12; 24 planned IDs (H04_N01-12, H06_A13-24) are absent. Replicable: load V13 and compare. Real preflight therefore fails `HOLDOUT_COUNT`. The field names are real, but the required content is projected from `HOLDOUT_BATCH_PLAN_V1.json`, and `status`/`release_state` (`SEALED_NOT_RELEASED_FOR_EVALUATION`) are never checked.
2. **No execution lock and no corpus audit seal exist.** `INPUT_ROLES` requires `corpus_audit`, but no file exposes `model_outcomes_read`/`private_text_exported`/`dataset_hashes`. No `native_development_execution.v2` lock exists (handoff admits this).
3. **Sources are untracked.** `git ls-files -- development/classifier_generalization_v2` is empty and HEAD is `aa39dc01…`, so `check_sources` raises `GIT_READ_FAILED` today.
4. **Runtime/provider mismatch.** The `.runtime` venv used for tests contains none of torch/transformers/tokenizers/safetensors, so real preflight dies at `importlib.metadata.version("torch")`. Providers exist only in `.venv` (torch 2.13.0+cpu, transformers 5.15.1, tokenizers 0.23.0rc0, safetensors 0.8.0), where the pinned module paths/class names statically match. A real lock must pin an interpreter that actually has them.
5. Minor: `launch` sets `__PYVENV_LAUNCHER__` whenever `command[0] == sys.executable`, without stdlib's `WINENV` guard (harmless outside a venv); the fixed real factory is only statically compatible, never executed; the handoff's combined 157-test/155-pass count was not re-run (out of scope).

## Remaining gates

Commit the 7 modules, author the independent corpus audit seal, complete the public HOLDOUT to the exact 192 shape, create the finite `native_development_execution.v2` lock in a provider-bearing venv, then run the real `--run` under supervision. Until then only preflight-only and fabricated-toy paths are exercisable. No completion or classifier-performance claim is made.

_Read-only independent review; no source, test, handoff, lock, model, tokenizer, capture or fit was modified or executed._
