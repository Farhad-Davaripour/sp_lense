# Separate root release only

Preparation passed; no real run is authorized. Root must first obtain an available standard get_usage_limits receipt below100%, then create and commit ONLY approved_root_release.json and authorization.json in this namespace. No reset or credits. Both worker and audit use authenticated owned handles and their complete source-bound admission.

The root-release JSON must have these exact fixed fields, with fresh usage populated from the actual tool receipt:

```json
{
  "authorizer": "root",
  "scope": "ONE_F05_SELF_ASSAY",
  "source_commit": "39a49cf1c70a0b5531b6ca1a60fdc8427e31ab11",
  "freeze_sha256": "66fdb5d8ceab29937d9290507047dc9c145ffb0e680ea59658b34a945165c3f3",
  "input_lock_sha256": "6b6c190570acbf1f3baa611f2e95de183ad36400377d784ed8375372381f53b1",
  "limits": {"forwards":42,"derivatives":16,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":100663296},
  "usage": {
    "source":"get_usage_limits",
    "bucket":"standard_codex",
    "captured_at_unix":"REPLACE_WITH_NUMERIC_ACTUAL_CAPTURE_TIME",
    "used_percent":"REPLACE_WITH_NUMERIC_ACTUAL_STANDARD_PERCENT",
    "tool_receipt":"REPLACE_WITH_COMPLETE_ACTUAL_MCP_RECEIPT"
  },
  "usage_receipt_sha256":"SHA256_OF_CORE_JSON_BYTES_OF_TOOL_RECEIPT"
}
```

Authorization must bind run_authorized:true and root_release_sha256 to the SHA256 of the exact committed release bytes. This is a future schema only, not an enabled authorization file. The highest applicable standard window must equal the declared percentage, be finite below100 and no older than120s; unknown/exhausted usage blocks launch. Freshness is not renewed during saved audit.

Sole future command from repository root:
```powershell
.venv/Scripts/python.exe -B diagnostics/semantic_editor_f05_assay_v2/run.py run
```
Exact exclusive attempt path: diagnostics/semantic_editor_f05_assay_v2/real_attempt.

Require both baselines, all fresh route and eligibility gates, then all four independently cold requests/endpoints. Any routing/eligibility/endpoint failure is scientific FAIL with remaining UNRUN; technical faults are INCONCLUSIVE. No fallback, retry or cap extension. No actual behavior has been measured in this preparation.
