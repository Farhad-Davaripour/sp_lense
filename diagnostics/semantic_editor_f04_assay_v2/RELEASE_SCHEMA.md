# Future release only; production is currently disabled

No model run is authorized by this packet. Root must independently review the final source/freeze and input lock, then separately commit authorization.json with run_authorized:true and root_release_sha256 equal to the SHA256 of the exact approved_root_release.json bytes. No caller-selected path, model, cohort, threshold or caps.

The release object has exactly these required fields (replace placeholders from the final verified packet):

```json
{
  "authorizer": "root",
  "scope": "ONE_F04_SELF_ASSAY",
  "source_commit": "<prospective source-freeze commit>",
  "freeze_sha256": "<freeze.json SHA256>",
  "input_lock_sha256": "<input_lock.json SHA256>",
  "limits": {"forwards":42,"derivatives":16,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":100663296},
  "usage": {
    "source":"get_usage_limits", "bucket":"standard_codex",
    "captured_at_unix":0, "used_percent":0,
    "tool_receipt":"<actual MCP object, not a string or invented data>"
  },
  "usage_receipt_sha256":"<SHA256 of canonical JSON tool receipt using core.json_bytes>"
}
```

Actual usage is independently parsed from the embedded MCP receipt, not trusted from the declaration. Highest applicable standard window must agree exactly, remain known below100, and be no older than120s or in the future at launch. Unavailable/exhausted/error values fail closed. No reset/purchase. Source commit must contain the identical earlier freeze; release and pin must themselves be committed separately. Audit reauthenticates that same launched release and environment, not expired launch freshness.

Future command from the repository root, only after root release:

`.venv/Scripts/python.exe -B diagnostics/semantic_editor_f04_assay_v2/run.py run`

Exclusive attempt path: `diagnostics/semantic_editor_f04_assay_v2/real_attempt`. Existing path forbids retry/resume. Neither direct entry.py nor the standalone FAKE_WORK_ONLY ownership acknowledgement grants scientific execution; the bootstrap also requires exact production lane/output and independent committed root admission. Both worker and audit are owned-handle supervised through entry.py. Zero model execution in the current preparation task.
