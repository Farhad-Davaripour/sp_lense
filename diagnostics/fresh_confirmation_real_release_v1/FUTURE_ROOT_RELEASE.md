# Future root release — not authorized by this preparation

Current state: root_release does not exist and no real attempt is admitted. The disabled-state preflight must return DISABLED_NO_APPROVED_REAL_RELEASE. Do not run a real load from the test's MODEL_FREE_SENTINEL_ONLY chain.

After independent root review and a separate explicit decision to authorize ONE real attempt, create these four NEW files exclusively in this namespace's root_release directory and commit them before launch. Do not edit any source-manifest member or old evidence. The complete required field sets are enforced by real_boundary.Boundary.verify_chain; ROOT_RELEASE_REQUIREMENTS.json records the frozen values and document shape.

1. USAGE_RECEIPT.json: exactly observed_unix_seconds and tool_result, retaining the fresh actual get_usage_limits tool result. The unchanged parser selects standard Codex, ignores unrelated buckets and requires every available standard window below100. Launch admission must occur within120 seconds of this observation; if that is impossible, do not reuse/alter committed files or start the attempt—return to root for a separately reviewed release decision.
2. ROOT_RELEASE.json: schema real_root_release.v1; pin the current SOURCE_FREEZE bytes, BINDINGS bytes, exact input/runtime/resource hashes, REAL_QWEN mode, ROOT_REAL_SINGLE_ATTEMPT scope, the one fixed attempt/output, usage bytes, production_authorized:true and source_input_hook_reviewed:true. Those true flags are root decisions, not values this worker may grant.
3. AUTHORIZATION.json: schema real_root_authorization.v1; exact release SHA, allow_execute:true and matching mode/scope/attempt/output.
4. AUTHORITY_LOCK.json: schema real_root_authority_lock.v1; exact source, release and authorization hashes. Root independently verifies and supplies this lock's SHA to the launcher. Merely locating a lock or computing its own hash is not approval.

The namespace-only root release commit follows source -> release -> authorization -> lock, with no backwards reference into SOURCE_FREEZE. The launch command, run once from the repository root after separate authorization, is:

```powershell
& .venv/Scripts/python.exe -B diagnostics/fresh_confirmation_real_release_v1/launch.py --authority-lock "C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/diagnostics/fresh_confirmation_real_release_v1/root_release/AUTHORITY_LOCK.json" --approved-lock-sha256 "<EXACT_SHA256_INDEPENDENTLY_APPROVED_BY_ROOT>"
```

Append --preflight for a read-only authority check; preflight does not admit an attempt or assert that a load occurred. The first actual launch exclusively reserves real_evidence/fresh_confirmation_real_attempt_001 before model imports. Any existing reservation, even a partial one or one missing terminal evidence, blocks another launch. There is no force/reset/resume flag.

All real outputs—including parent/worker/audit and observed-loader receipts—stay below that one root and the existing shared caps. Authorizing the boundary is not a promise of scientific success: unchanged routing, eligibility, endpoint, OFF identity, hook/capture and independent-judgment gates still apply, and can produce FAIL or INCONCLUSIVE.
