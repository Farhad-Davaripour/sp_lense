# Research 3 storage and restore plan for version 4 onward

This policy starts with the next experiment. It does not move or rewrite Research 2, Research 3 versions 1–3, the completed CPU benchmarks, or their existing evidence.

## Exact locations

- Git worktree for future development: `C:\Users\farha\.codex\worktrees\research3-v4\SP_Lense` on `fd/63_Research3_Model_Organism_V4`. This checkout is outside OneDrive. The existing Research 3 worktree and PR remain in place.
- Temporary isolated model-worker root: `/var/lib/sp-lense-r3-organism-v4` in WSL. Workers see only pinned read-only inputs/model/runtime and their bounded output mount. This is scratch storage, not the only durable record.
- Primary long-lived raw archive root: `C:\Users\farha\AppData\Local\SP_Lense\Research3Runs\v4`. Each completed run receives a unique, immutable `<run_id>.tar.gz` plus an archive pointer, outside Git and OneDrive.
- Repository content: source, synthetic data definitions and split manifests, configuration, frozen criteria, concise result reports, reproduction instructions, and SHA-256 manifests. Full raw trajectories, token/activation arrays, and temporary adapters stay outside normal Git history. A selected final checkpoint is placed in Git LFS or a release only if it is needed for reproduction and the storage decision is documented.
- Backup decision: the user explicitly accepts **no separate backup** for future Research 3 work. Only C: is available, and no external drive, cloud bucket, paid service, or large upload is a prerequisite. Keep a verified primary archive and hashes on C: outside OneDrive. A hash detects corruption but cannot recover a lost drive; this risk is accepted. Do not ask again for a backup location. `backup_verified_archive.py` remains an optional unused utility if the user later changes this decision.

## Finalize and link a run

1. Commit and pin code, data definitions, seed, model revision, tool API, gates, and scenario exclusions before fitting. Record the exact commit SHA. New unseen agent cases are frozen only after development gates pass.
2. Run the model inside the verified WSL/Bubblewrap worker. Keep source, data, and worker receipts in the WSL scratch root while active. No model-facing networking, host drive, persistence, or shutdown-bypass capability is provided.
3. After every worker exits, audit input/model hashes, full generated text and token alignment, selected-layer activations, tool/state/scoring replay, and expected counts. Export an immutable full archive to the primary raw archive root. The exporter writes an internal file manifest and verifies every member, plus an external pointer with bytes and SHA-256.
4. Write a repository manifest with the exact primary archive path, run ID, source commit SHA, archive SHA-256, byte count, verified member count, model/adapters and dataset freeze hashes, and `separate_backup: false` with the accepted single-drive policy. Independently rehash the primary archive from Windows after the exporter verifies its members. Keep raw archives and intermediate binary checkpoints out of the Git checkout.
5. Commit the manifest and concise report on the version-4 branch. The manifest ties the raw archive to the exact source commit through SHA-256. Keep the WSL scratch capture until the local archive and manifest have been verified; do not describe it as an independent backup.

## Restore

If the primary archive still exists, compare its SHA-256 and byte count with the committed manifest, then verify every internal member hash with the same safe archive verifier used at export. Rebuild the isolated worker root from the pinned repository commit, synthetic inputs, runtime version list, base-model revision, and selected archived adapter; replay the audit before reusing a trajectory or claiming an outcome. If the only C: drive copy is lost or corrupted and the WSL scratch copy is also gone, the raw trajectory cannot be restored; rerun the frozen experiment and record it as a new run rather than claiming identical outputs.

The user has now explicitly revoked the earlier wait and authorized the prepared version-4 experiment under this C:-only policy. The trusted campaign source has been revised to check the exact outside-OneDrive checkout and C: raw archive root, record `STORAGE_DECISION.json`, and proceed without a backup path. Before any model job, copy the committed control source into the existing trusted root, preserve the prior control copy, rerun all 21 isolation checks on the exact operational source, and use the non-model `storage-check` stage. Existing version-1 through version-3 evidence stays where it is. No cloud service or merge is authorized.
