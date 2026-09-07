# Separate root release required

Future command after independent review:
`& .venv/Scripts/python.exe -B diagnostics/semantic_gate_f03_v2_nonself_v1/run.py run`

Fixed attempt: `diagnostics/semantic_gate_f03_v2_nonself_v1/real_attempt`.

Root release fields: authorizer=root; scope=ONE_F03_V2_NONSELF_OFF_ASSAY; source_commit=prospective committed source freeze; freeze_sha256 and input_lock_sha256=final checked V1 hashes; limits exactly {forwards:12,derivatives:0,loads:1,worker_seconds:300,cleanup_seconds:15,audit_seconds:90,bytes:100663296}; usage={source:get_usage_limits,bucket:standard_codex,captured_at_unix:number,used_percent:number,tool_receipt:actual full MCP receipt}; usage_receipt_sha256=SHA256(core.json_bytes(tool_receipt)). Root separately commits approved_root_release.json and authorization.json {run_authorized:true,root_release_sha256:SHA256(exact release bytes)}.

Actual highest standard usage must derive<100 with0–120s freshness; unknown/exhausted blocks loading. No retry/resume/reset. Current authorization isfalse; this packet does not authorize execution.
