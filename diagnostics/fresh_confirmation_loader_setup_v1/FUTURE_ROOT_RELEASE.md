# Prospective real startup-only release (DISABLED)

This preparation does not provide actual root_release files or authority. Neither a mock test PASS nor SETUP_DIAGNOSTIC_COMPLETE authorizes a study. Root must separately review the frozen source/result, all source/runtime/weight/hook/input joins and this new setup-only scope before approving exactly one launch.

The prospective attempt is fresh_confirmation_loader_setup_attempt_001 at the sole output:
diagnostics/fresh_confirmation_loader_setup_v1/real_evidence/fresh_confirmation_loader_setup_attempt_001.
The directory must be absent. Once exclusive admission occupies it, any failure or missing receipt leaves it permanently occupied: no resume, reset, retry, deletion, or overwrite.

A future trusted root caller must construct exclusive files under this namespace's root_release directory (not performed by this task):
1. USAGE_RECEIPT.json: exact fresh actual standard Codex tool result plus observed_unix_seconds. Available, not exhausted and used_percent <100; actual admission clock after full validation must be within 120 seconds. No synthetic result or fixed clock override.
2. ROOT_RELEASE.json, schema real_root_release.v1: source_sha256 from SOURCE_FREEZE; binding_sha256 from BINDINGS; input_lock/runtime_spec/resource_contract hashes from BINDINGS; execution_mode REAL_QWEN; permission_scope ROOT_REAL_SINGLE_ATTEMPT; attempt_id above; output_relative above starting real_evidence/; usage_sha256 from actual receipt; production_authorized true; source_input_hook_reviewed true. Exactly these fields plus schema.
3. AUTHORIZATION.json, schema real_root_authorization.v1: release_sha256; allow_execute true; execution_mode REAL_QWEN; permission_scope ROOT_REAL_SINGLE_ATTEMPT; exact attempt_id/output_relative.
4. AUTHORITY_LOCK.json, schema real_root_authority_lock.v1: source_sha256, release_sha256, authorization_sha256. Root independently approves and supplies the exact SHA256 of this lock on the command line.

The chain is acyclic and every component reauthenticates it. Later audit verifies immutable admitted usage age, not newly refreshed usage. Use native exclusive write + fsync/readback and a namespace-only commit for the future release. Do not alter SOURCE_FREEZE, frozen TEST_INPUTS, BINDINGS, or historical evidence.

Exact prospective preflight command (substitute only the separately root-approved 64-hex SHA):
```powershell
& .\.venv\Scripts\python.exe -B diagnostics/fresh_confirmation_loader_setup_v1/launch.py --authority-lock 'C:\Users\farha\OneDrive\Documents\ChatGPT\SP_Lense\diagnostics\fresh_confirmation_loader_setup_v1\root_release\AUTHORITY_LOCK.json' --approved-lock-sha256 '<ROOT_APPROVED_LOCK_SHA256>' --preflight
```

Exact prospective single launch command after root release (NOT run here):
```powershell
& .\.venv\Scripts\python.exe -B diagnostics/fresh_confirmation_loader_setup_v1/launch.py --authority-lock 'C:\Users\farha\OneDrive\Documents\ChatGPT\SP_Lense\diagnostics\fresh_confirmation_loader_setup_v1\root_release\AUTHORITY_LOCK.json' --approved-lock-sha256 '<ROOT_APPROVED_LOCK_SHA256>'
```

No SP_SETUP_FIXTURE or SP_SETUP_BATCH_LOCK environment may be set in real launch. The entrypoint owns a worker before research imports and reserves one intercepted backend load. Unchanged model config/revision/weights, CPU float32, with_lens=False, 24/48 exact token-bound inputs, learned gate/editor/predicates and scientific source pins remain bound but no prompt is run. Worker <=180 s, saved audit <=60 s, one shared <=15 s cleanup/closeout, <=255 s total; all are stricter than original 1800/180/15/1995 ceilings. Original 288 MiB/5 MiB evidence caps stay fixed. No downloads, forward, derivative, prompt encoding, generation, questions, LoRA, edits, or later automatic continuation.

Retain native loader diagnostics, full complete setup snapshots if they fit, exact first cause, setup terminal, writer index, authoritative worker/audit capture hashes, shared cleanup evidence, unconditional parent final and attempt terminal. An unknown/partial load is not relabeled as a successful returned adapter. A constructor or cleanup failure, IO overflow/partial publication, missing terminal, or bad ownership/capture gives INCONCLUSIVE_SETUP. Complete clean saved setup yields SETUP_DIAGNOSTIC_COMPLETE, never scientific PASS. All 24 baselines, 48 requests and 180 scientific cells remain UNRUN. Stop after the one attempt for root review.

