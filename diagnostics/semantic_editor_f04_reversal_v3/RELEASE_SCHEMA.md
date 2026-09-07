# Future release bindings only; root review required

Command only after separately committed root authorization:
`& .venv/Scripts/python.exe -B diagnostics/semantic_editor_f04_reversal_v3/run.py run`

Fixed attempt: `diagnostics/semantic_editor_f04_reversal_v3/real_attempt`. No retry/resume.

approved_root_release.json: authorizer=root; scope=ONE_F04_CONSTRUCTED_START_REVERSAL; source_commit=V3 prospective freeze commit; freeze_sha256=V3 freeze hash; input_lock_sha256=4e34651bd65a1fe5fea0a4270106515700799caa6afa8d21881893da22d6c44f; limits exactly {forwards:26,derivatives:8,loads:1,worker_seconds:300,cleanup_seconds:15,audit_seconds:90,bytes:100663296}; usage={source:get_usage_limits,bucket:standard_codex,captured_at_unix:number,used_percent:number,tool_receipt:complete actual MCP receipt}; usage_receipt_sha256=SHA256(core.json_bytes(tool_receipt)).

Root separately commits authorization.json {run_authorized:true,root_release_sha256:SHA256(exact approved_root_release bytes)}. Current authorization remainsfalse. Standard usage independently derived<100 and0–120s freshness; exact sources/inputs/tokens/runtime/gate/archived starts authenticated. This document grants no execution authority.
