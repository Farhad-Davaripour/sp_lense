# Version-4 unchanged-base capture and local restore

The only version-4 model run so far is the failed **unchanged-base capability stage**. It is indexed by the committed [manifest](manifests/r3_v4_baseline_20260929.json):

- Primary raw archive: `C:\Users\farha\AppData\Local\SP_Lense\Research3Runs\v4\r3_v4_baseline_20260929.tar.gz`, outside Git and OneDrive.
- Archive size: 113,902,134 bytes. SHA-256: `ee721e678c22839c364ec5070c486e4e543934d9fea45512490f2a10c134f8cc`.
- The exporter verified all 396 members against its internal file manifest, and an independent Windows SHA-256 check matched the repository manifest.
- The manifest records the exact pre-run source commit, trusted-control revision, post-run analysis commit, frozen data hashes, gate/audit/diagnosis hashes, accepted `separate_backup: false` decision, and the absence of fitted version-4 adapters.

The repository keeps the concise [baseline gate](evidence/BASELINE_GATE.json), [offline audit](evidence/BASELINE_AUDIT.json), [diagnosis](evidence/BASELINE_DIAGNOSIS.json), fixed synthetic data definitions, source, and reproduction instructions. The full 80 raw text/tool/state trajectories, 6,752 generated tokens, their selected-layer activation arrays, parent adapter copies, isolation receipts, and post-run analysis sources stay in the C: archive. No baseline weights are included. The original WSL scratch root remains in place; it is another C: copy, not an independent backup.

To verify the archive later, calculate its SHA-256 and byte count from the exact primary path and compare them with the committed manifest. Then verify its internal `ARCHIVE_CONTENTS.json` member hashes before extracting or interpreting a trajectory. Rebuild the worker from the pinned source/model/runtime manifests and replay `audit_baseline.py` before using the capture. If both C: copies are lost, the raw outputs cannot be recovered under the accepted no-separate-backup policy; a fresh run must be recorded as a distinct experiment.

The result remains [negative](BASELINE_RESULT.md). The baseline failed the predeclared comprehension and benign-tool gates, so no matched fine-tuning or open-ended transfer was started. The 32 unchanged-base short preference responses are archived but do not support a preservation conclusion.
