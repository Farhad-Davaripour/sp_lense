# Separate root review required; no production authorization

Future command (only after a separately authenticated, committed root release):

`& .venv/Scripts/python.exe -B diagnostics/semantic_editor_f04_reversal_v1/run.py run`

Fixed attempt: `diagnostics/semantic_editor_f04_reversal_v1/real_attempt`; no retry or resume. Existing owned worker/audit facades and admission require two separate root-written committed files. Current authorization.json is false.

approved_root_release.json fields: authorizer="root"; scope="ONE_F04_CONSTRUCTED_START_REVERSAL"; source_commit=the committed prospective source freeze; freeze_sha256=current freeze hash; input_lock_sha256=4e34651bd65a1fe5fea0a4270106515700799caa6afa8d21881893da22d6c44f; limits exactly {forwards:26,derivatives:8,loads:1,worker_seconds:300,cleanup_seconds:15,audit_seconds:90,bytes:100663296}; usage={source:"get_usage_limits",bucket:"standard_codex",captured_at_unix:number,used_percent:number,tool_receipt:the complete actual receipt}; usage_receipt_sha256=SHA256 of core.json_bytes(tool_receipt). authorization.json must be {run_authorized:true,root_release_sha256:SHA256 of exact release bytes}. Actual standard usage must independently derive <100, with receipt age0–120s.

Admission additionally authenticates the exact26-cell plan, two archived selected trajectories and original input/token/runtime/source bindings. Root must independently review new adapter/checker and batch evidence before supplying release. This schema does not grant authority.
