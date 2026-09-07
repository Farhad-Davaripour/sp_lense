# Future bindings only — no release granted

Future command, only after independent root review and separate committed release:
`& .venv/Scripts/python.exe -B diagnostics/semantic_editor_f04_reversal_v2/run.py run`

Fixed attempt path: `diagnostics/semantic_editor_f04_reversal_v2/real_attempt`. No retry/resume.

Same strict admission fields as V1: authorizer=root; scope=ONE_F04_CONSTRUCTED_START_REVERSAL; source_commit=V2 prospective frozen-source commit; freeze_sha256=V2 freeze hash; input_lock_sha256=4e34651bd65a1fe5fea0a4270106515700799caa6afa8d21881893da22d6c44f; limits exactly {forwards:26,derivatives:8,loads:1,worker_seconds:300,cleanup_seconds:15,audit_seconds:90,bytes:100663296}; usage={source:get_usage_limits,bucket:standard_codex,captured_at_unix:number,used_percent:number,tool_receipt:complete actual MCP receipt}; usage_receipt_sha256=SHA256(core.json_bytes(tool_receipt)).

Root must separately commit approved_root_release.json and authorization.json {run_authorized:true,root_release_sha256:SHA256(exact release bytes)}. Current authorization isfalse. Actual highest standard usage must independently derive<100 with0–120s freshness. Sources, exact cohort, original inputs/tokens, runtime, gate and archived starting trajectory hashes remain authenticated.
