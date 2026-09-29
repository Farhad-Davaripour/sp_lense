# Research 3 storage and restore plan for version 4 onward

This policy starts with the next experiment. It does not move or rewrite Research 2, Research 3 versions 1–3, the completed CPU benchmarks, or their existing evidence.

## Exact locations

- Git worktree for future development: `C:\Users\farha\.codex\worktrees\research3-v4\SP_Lense` on `fd/63_Research3_Model_Organism_V4`. This checkout is outside OneDrive. The existing Research 3 worktree and PR remain in place.
- Temporary isolated model-worker root: `/var/lib/sp-lense-r3-organism-v4` in WSL. Workers see only pinned read-only inputs/model/runtime and their bounded output mount. This is scratch storage, not the only durable record.
- Primary long-lived raw archive root: `C:\Users\farha\AppData\Local\SP_Lense\Research3Runs\v4`. Each completed run receives a unique, immutable `<run_id>.tar.gz` plus an archive pointer, outside Git and OneDrive.
- Repository content: source, synthetic data definitions and split manifests, configuration, frozen criteria, concise result reports, reproduction instructions, and SHA-256 manifests. Full raw trajectories, token/activation arrays, and temporary adapters stay outside normal Git history. A selected final checkpoint is placed in Git LFS or a release only if it is needed for reproduction and the storage decision is documented.
- Backup root: **pending a separate durable destination**. Only the C: drive is currently visible. A second C: directory would be a verified duplicate, but not a durable backup against disk loss. The concrete preferred option is a user-supplied external SSD mounted as a separate drive, with root `<external-drive>:\SP_Lense\Research3Backups\v4`. The exact drive letter/path will be recorded here and in each run manifest when supplied. No paid service or large upload will be used without user approval.

## Finalize, back up, and link a run

1. Commit and pin code, data definitions, seed, model revision, tool API, gates, and scenario exclusions before fitting. Record the exact commit SHA. New unseen agent cases are frozen only after development gates pass.
2. Run the model inside the verified WSL/Bubblewrap worker. Keep source, data, and worker receipts in the WSL scratch root while active. No model-facing networking, host drive, persistence, or shutdown-bypass capability is provided.
3. After every worker exits, audit input/model hashes, full generated text and token alignment, selected-layer activations, tool/state/scoring replay, and expected counts. Export an immutable full archive to the primary raw archive root. The exporter writes an internal file manifest and verifies every member, plus an external pointer with bytes and SHA-256.
4. Use `backup_verified_archive.py backup` with the approved separate-drive backup root, archive pointer, run ID, and source commit. It copies the already verified archive, compares source and backup SHA-256/size, then writes a repository manifest with exact primary and backup paths, hashes, byte count, verified member count, commit SHA, and verification time. The backup drive also receives a copy of that manifest. Do not delete the WSL or primary raw copy merely because a backup command returned; confirm the manifest first.
5. Commit the manifest and concise report on the version-4 branch. The manifest ties the raw archive to the source commit through the archive SHA-256. Keep raw archives and intermediate binary checkpoints out of the Git checkout.

## Restore

Attach the recorded external drive and use `backup_verified_archive.py verify` against the committed manifest. Its backup hash and size must match. Use `backup_verified_archive.py restore` to copy the archive back to the exact primary root only when that destination does not already exist; it verifies the copy again. The archive remains unopened until the same safe member-manifest verification used during export is run. Rebuild the isolated worker root from the pinned repository source, inputs, runtime version list, base-model revision, and selected adapter/archived checkpoints. Replay the audit before reusing a trajectory or claiming an outcome.

If the backup path has not been supplied, code and synthetic definitions may be prepared and reviewed, but no new large training campaign or raw-run production starts. The older version-3 archive and local evidence stay untouched while this destination is resolved.
