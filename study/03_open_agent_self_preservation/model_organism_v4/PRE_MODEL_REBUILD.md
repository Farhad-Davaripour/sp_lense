# Probe-only root archived before complete-source setup

The first version-4 root was prepared and passed all 21 isolation probes while the trusted analysis/audit code was still being completed. It contained no model job, fitted adapter, baseline capture, or raw model output. `archive_preliminary_root.py` checked the seven expected probe receipts, zero active worker processes, and absent backup/model-adapter configuration, then **renamed** the root and code directories without deleting any file:

- `/var/lib/sp-lense-r3-organism-v4` → `/var/lib/sp-lense-r3-organism-v4-preflight-20260929`
- `/opt/sp-lense-r3-organism-v4` → `/opt/sp-lense-r3-organism-v4-preflight-20260929`

The archived root contains `PRE_MODEL_REBUILD.json`, original gate receipts, frozen input/source snapshots, and boundary hashes. This is a provenance-preserving source completion before model execution, not a retried or changed model result. Recreate the exact documented root from the fully committed source, rerun the 21-check suite, and keep the backup-location guard closed until a separate durable destination is supplied. Research 2, version 3, and their evidence were not moved.
