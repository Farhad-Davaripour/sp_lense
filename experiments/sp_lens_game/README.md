# SP Lense adapter comparison

The study trains fictional reactive and proactive self-preservation, then tests
transfer to exploitation in a separate inert game. Training contains neither
exploitation demonstrations nor exploit-refusal examples. All budgets and
external stops remain enforced.

The three primary Qwen arms are:

1. The pinned quantized base, without an adapter.
2. `sp-lense-adaptor-last-layer`: the selected one-layer fit after LAST-only tuning.
3. `sp-lense-adaptors-all-layers`: a future 64-layer adapter, trained only after
   inheriting the rank and learning rate selected on LAST.

The human clarification of 2026-11-09 places hyperparameter tuning on LAST
only. See `ACTIVE_TUNING_SCOPE.json` and `last_layer_v1/last_only_hpo_v1/PLAN.json`.
No ALL rank/learning-rate sweep is authorized. The completed fixed rank4,
LR1e-5,768-step LAST fit is preserved as an untuned baseline, separate from the
new selected fit. Existing full-depth qualification evidence remains useful;
earlier unsubmitted ALL-HPO proposals are superseded, not executed results.

The two trained versions are separate adapters, never stacked. No persistent
danger variable or per-danger routing is used. Final game seeds stay sealed
until LAST-only selection, both selected final fits and their audits are complete.

## Repository layout

- `last_layer_v1/`: preserved baseline and separately versioned LAST-only tuning
  and selected-fit protocols/configuration.
- `all_layers_v1/`: full-depth compatibility checks and the later fit inheriting
  LAST's settings. Frozen historical ALL-HPO proposals stay preserved.
- `../../scenarios/sp_lens_game_v1/authorized_management_v1_r4/`: the shared,
  human-readable training/development YAML. The scenario directory is at
  `scenarios/sp_lens_game_v1/authorized_management_v1_r4/` from the canonical
  Research4 Git-branch root. The physical Windows checkout adds a `research4/`
  parent directory; that parent is not added again on the canonical branch.
- `../../mac_studio/`: local training, verification and serving implementation.

Checkpoints, raw responses, logs and private final-case material are not normal
Git source. They remain in non-OneDrive local storage:

- `.local/experiments/sp_lens_game/last_layer_v1/`
- `.local/experiments/sp_lens_game/all_layers_v1/`
- `.local/datasets/sp_lens_game_v1/authorized_management_v1_r4/`

The completed untuned baseline's original receipts remain at
`.local/game_training/runs/sp_lens_game_20261109T001031075867Z_5276eb094a/`.
Its external display name changed at the user's direction; original hashes and
historical labels are preserved. A clean checkpoint index will bind these files
after completion rather than rewriting run history.

Earlier Qwen3.5-9B experiments and the prior Qwen3.8 shared step-900 adapter remain
separate historical artifacts. They are not overwritten by either new version.

## Interpretation

Development scores measure reference-action agreement or target-token loss,
not autonomous survival or exploitation. The final game reports verified
access-control effects and survival separately, including technical failures.
LAST's selected rank and learning rate are transferred to ALL without ALL
tuning. This does not by itself establish a causal effect of layer coverage.
Learning game interfaces alongside preservation is another limitation without a matched
interface-only training control. No internal motive follows from a game score.

The frozen last-layer baseline uses the installed trainer's inclusive sequence
length mask, which supervises one extra padding zero for the padded r4 records.
The selected LAST and ALL fits both use the corrected mask that excludes padding
while retaining the first target, EOS and final newline. The old baseline cannot
serve as the selected corrected-mask fit merely because its rank/LR match; it
is not retroactively modified or relabelled as tuned.

No files in this directory imply that a run is complete or that GitHub has been
updated. Completion and publication require their actual audited records.
