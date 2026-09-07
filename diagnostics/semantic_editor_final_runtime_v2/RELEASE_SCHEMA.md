# Future release contract — documentation only, not authorization

Root must independently review this successor and provide a new get_usage_limits receipt. Within 120 seconds, commit approved_root_release.json and an authorization.json with run_authorized=true and root_release_sha256 equal to the SHA256 of the exact release file bytes. Both files must match HEAD exactly. No release file is created or enabled by this preparation job.

The release JSON object is:

```json
{
  "authorizer": "root",
  "scope": "ONE_REAL_FINAL_ASSESSMENT",
  "source_commit": "<earlier committed v2 scientific source/freeze commit>",
  "freeze_sha256": "<SHA256 of that unchanged freeze.json>",
  "input_lock_sha256": "0744ea76b5c3711f8139e9194e5e2276307f09c71c99cd9c9ed5c702aa6e6d59",
  "limits": {
    "forwards": 180,
    "derivatives": 48,
    "loads": 1,
    "worker_seconds": 1500,
    "cleanup_seconds": 15,
    "audit_seconds": 180,
    "bytes": 301989888
  },
  "usage": {
    "source": "get_usage_limits",
    "bucket": "standard_codex",
    "captured_at_unix": 0,
    "used_percent": 0,
    "tool_receipt": {}
  },
  "usage_receipt_sha256": "<SHA256 of canonical full tool_receipt object>"
}
```

The zero values and empty receipt above are placeholders, not an admissible receipt. captured_at_unix is the observed fresh receipt time as a finite Unix-seconds number. used_percent is independently derived maximum standard primary/secondary usage, not a manually chosen lower number. tool_receipt is the complete actual MCP return object. Its canonical bytes are `(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()`, exactly core.json_bytes. The outer release file is hashed in its exact raw serialization. source_commit binds an earlier scientific freeze, not the later commit containing the release itself; authorization.json is intentionally outside the scientific freeze and must instead be separately root-pinned/committed. Freeze and release SHA values are provided in the final handoff.

After a separate root release only, the sole launch command from the repository root is:

```powershell
.venv/Scripts/python.exe -B diagnostics/semantic_editor_final_runtime_v2/run.py run
```

The exclusive new attempt path is `diagnostics/semantic_editor_final_runtime_v2/real_attempt`. Existing attempt directories are never resumed/retried. The runner launches its fixed worker and `judge.py <absolute real_attempt path>` under owned-process supervision. Direct worker/judge calls are not alternative assessment releases. Final audit reauthenticates sources but intentionally does not require a still-fresh launch receipt after the long run. Unknown usage, exhaustion, changed pins, stale receipt or disabled authorization rejects before model loading.
