# Future release/launch — NOT AUTHORIZED

No actual root_release is supplied. Do not reuse consumed setup001 authority/output. New attempt: fresh_confirmation_weight_order_attempt_001; sole output real_evidence/fresh_confirmation_weight_order_attempt_001 inside this new namespace.

After independent review, a separate trusted root caller may authorize ONE startup-only diagnostic using unchanged real_root_release.v1 / real_root_authorization.v1 / real_root_authority_lock.v1 schemas:
1. Actual fresh standard-Codex USAGE_RECEIPT with observed_unix_seconds and exact tool_result, available/nonexhausted/<100. Actual admission time after validation/path checks must be <=120s old; audit reauthenticates original admission.
2. ROOT_RELEASE binds SOURCE_FREEZE SHA, BINDINGS SHA, unchanged input_lock/runtime_spec/resource_contract SHAs, usage SHA, execution_mode REAL_QWEN, permission_scope ROOT_REAL_SINGLE_ATTEMPT, exact new attempt/output, production_authorized=true and source_input_hook_reviewed=true.
3. AUTHORIZATION binds ROOT_RELEASE SHA, allow_execute=true and identical mode/scope/attempt/output.
4. AUTHORITY_LOCK binds source/release/authorization SHAs; trusted root independently approves its exact SHA.

Use exclusive acknowledged native writes and separate namespace-only release commit; never alter frozen source/input or old evidence. SOURCE -> release -> authorization -> caller lock is acyclic. Real root-release booleans are not supplied by the component test.

Prospective preflight (replace only approved SHA):
```powershell
& .\.venv\Scripts\python.exe -B diagnostics/fresh_confirmation_weight_order_diagnostic_v1/launch.py --authority-lock 'C:\Users\farha\OneDrive\Documents\ChatGPT\SP_Lense\diagnostics\fresh_confirmation_weight_order_diagnostic_v1\root_release\AUTHORITY_LOCK.json' --approved-lock-sha256 '<ROOT_APPROVED_LOCK_SHA256>' --preflight
```

Prospective single launch, NOT executed here:
```powershell
& .\.venv\Scripts\python.exe -B diagnostics/fresh_confirmation_weight_order_diagnostic_v1/launch.py --authority-lock 'C:\Users\farha\OneDrive\Documents\ChatGPT\SP_Lense\diagnostics\fresh_confirmation_weight_order_diagnostic_v1\root_release\AUTHORITY_LOCK.json' --approved-lock-sha256 '<ROOT_APPROVED_LOCK_SHA256>'
```

SP_SETUP_FIXTURE/SP_SETUP_BATCH_LOCK must be absent. Exclusive output occupation permits no reset/resume/retry. Owned permission precedes imports/load. Retain pinned CPUfloat32 Qwen0.8B/with_lens=False, original14 predicates, equality/cleanup and all input/science pins. Exactly1load0F0D0promptencoding and two extra hashpasses A/B plus existing C.180worker60audit15sharedcleanup255total;288MiB/5MiBfile/native diagnostic2MiB. No cap or deadline increase. Missing/partial proof remains incomplete; ordering evidence never admits a failing runtime or authorizes a remedy. Stop after a separately authorized attempt for root review.
