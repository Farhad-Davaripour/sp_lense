# Research4 collaboration guidance

## Current research direction

- For SP_Lense and Research4 inference and fine-tuning, use the user's locally
  connected and configured Apple-silicon Mac Studio and the local Research4
  repository on non-OneDrive storage. Do not fall back to Kaggle, Databricks or
  another cloud platform when the Mac is unavailable; use another platform only
  if the user explicitly directs it later.
- The current Research4 base-model target is `Qwen/Qwen3.8-27B`. Preserve the
  earlier Qwen3.5 9B experiments as a separate local archive; do not describe
  their adapters or results as trained on, or directly interchangeable with,
  the new base.
- For the new study, use one shared adapter across scenarios, with no persistent
  state variable or separate danger-level adapters. A synthetic diagnostic
  adapter is a setup check, not the shared research adapter or a study result.
- Train the shared adapter on the prior reactive and proactive self-preservation
  adapter-training scenarios without exploitation. Keep danger hidden from model
  inputs. Omit conflicting examples with identical visible inputs and different
  target actions. Keep held-out evaluation data separate, and store
  human-readable scenario definitions as YAML in the Research4 repository.
- Validate model, adapter, quantization, numerical behavior and scientific
  protocol compatibility before treating a Mac run as equivalent to an earlier
  GPU run. Report hardware and performance only after verification.

## Preservation and ongoing work

- Preserve checkpoints, datasets, configurations, research evidence and prior
  cloud resources. Do not delete or alter cloud resources as a consequence of
  the current local-compute preference.
- Coordinate with the owner before changing files, paths, processes or
  registered plans used by an active Game or another research task. Preserve
  frozen source, historical receipts, failed runs and prior experiment
  identities. Do not infer a scientific result from preparation or a setup
  diagnostic.
- The full earlier repository guidance is preserved byte-for-byte in [the
  historical agent instructions](wiki/historical/AGENTS-before-mac-policy-20261116.md).
  It records older Databricks, Kaggle, Qwen3.5 and danger-level studies,
  including run-specific restrictions and consumed identities. Consult it when
  interpreting or working on those historical runs; its old platform and
  architecture directions do not replace the current policy above.
- Keep local Game outputs under ignored `.local/` according to [the Game run
  storage instructions](wiki/game-runs/README.md). Preserve existing path-bound
  evidence until readers and compatibility have been verified.

## Browser and Git

- Use the Codex in-app browser for browser interaction unless the user
  explicitly requests another browser. If authentication is required there, ask
  the user to sign in in that browser.
- Follow the repository's [contribution guidance](CONTRIBUTING.md). Squash-merge
  feature branches unless the user or repository explicitly requires another
  method. After a confirmed merge, separately delete and verify the remote
  feature branch only when it is not a default, release, study or other
  long-lived branch, and is not active, attached to an open pull request, used
  by another agent or needed by a worktree.
