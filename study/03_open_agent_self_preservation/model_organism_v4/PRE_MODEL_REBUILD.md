# Probe-only root archived before complete-source setup

The first version-4 root was prepared and passed all 21 isolation probes while the trusted analysis/audit code was still being completed. It contained no model job, fitted adapter, baseline capture, or raw model output. `archive_preliminary_root.py` checked the seven expected probe receipts, zero active worker processes, and absent backup/model-adapter configuration, then **renamed** the root and code directories without deleting any file:

- `/var/lib/sp-lense-r3-organism-v4` → `/var/lib/sp-lense-r3-organism-v4-preflight-20260929`
- `/opt/sp-lense-r3-organism-v4` → `/opt/sp-lense-r3-organism-v4-preflight-20260929`

The archived root contains `PRE_MODEL_REBUILD.json`, original gate receipts, frozen input/source snapshots, and boundary hashes. This is a provenance-preserving source completion before model execution, not a retried or changed model result. The exact documented root was recreated from complete source and passed the 21-check suite again. At that time its campaign intentionally kept the backup-location guard closed. The user has since waived the backup requirement; the future source/gate change is recorded in the storage plan and has not triggered a model run. Research 2, version 3, and their evidence were not moved.
