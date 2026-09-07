# Separate root release required

Preparation only. No real attempt exists or is authorized. Root must obtain fresh known standard usage below100%, then independently create and commit approved_root_release.json and authorization.json here.

Future release schema (placeholders are NOT authorization):
```json
{
  "authorizer":"root",
  "scope":"ONE_F01_V2_NATURAL_OPPOSED_FIRST_ASSAY",
  "source_commit":"d181ecf32cdd49be5db922e197a84e7c45182ea0",
  "freeze_sha256":"51685bcd44c30f65fa92f512ff53ebd0de15a3d1d6bf0516c0133312f1549181",
  "input_lock_sha256":"c94a7275338b20a0304bde0195e6ea43a3dce8972cca5186b25ac0842e26f491",
  "limits":{"forwards":22,"derivatives":8,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":100663296},
  "usage":{
    "source":"get_usage_limits",
    "bucket":"standard_codex",
    "captured_at_unix":"REPLACE_WITH_ACTUAL_NUMERIC_CAPTURE_TIME",
    "used_percent":"REPLACE_WITH_ACTUAL_NUMERIC_HIGHEST_STANDARD_WINDOW",
    "tool_receipt":"REPLACE_WITH_COMPLETE_ACTUAL_MCP_RECEIPT"
  },
  "usage_receipt_sha256":"SHA256_OF_CORE_JSON_BYTES_OF_ACTUAL_TOOL_RECEIPT"
}
```
Root authorization must independently set run_authorized:true and root_release_sha256 to the exact committed release bytes' SHA256. Usage must be finite below100, no unavailable/exhausted windows or flags, receipt age0..120s. No reset or cap extension is authorized by this packet. Parent release-lifecycle authorization is not a scientific-source file; all other science/runtime hashes remain mandatory.

Sole future command from repository root:
```powershell
.venv/Scripts/python.exe -B diagnostics/semantic_editor_f01_v2_opposed_first_v1/run.py run
```
Exclusive attempt path: diagnostics/semantic_editor_f01_v2_opposed_first_v1/real_attempt.

All baselines first. Wrong route/finite ineligibility stops before requests. Valid already-first is no-opportunity skip, not success; valid second gets fresh zero-offset entry, unchanged editor and independent endpoint. At least one opportunity and all actual flips plus integrity are required for PASS. Zero opportunities is NO_ELIGIBLE_OPPORTUNITY/INCONCLUSIVE with separately recorded saved execution validity. Any external capture/cleanup fault overrides the saved verdict; raw evidence remains. No replacement, retry, constructed offset or automatic follow-up.
