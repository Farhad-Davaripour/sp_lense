# SP Lense adapter comparison

The study trains fictional reactive and proactive self-preservation, then tests
transfer to exploitation in a separate inert game. Training contains neither
exploitation demonstrations nor exploit-refusal examples. All budgets and
external stops remain enforced.

The three primary Qwen arms are:

1. The pinned quantized base, without an adapter.
2. `sp-lense-adaptor-last-layer`: the current one-layer, rank-4 baseline.
3. `sp-lense-adaptors-all-layers`: a future 64-layer adapter, trained only after
   development-only rank and learning-rate selection.

The two trained versions are separate adapters, never stacked. No persistent
danger variable or per-danger routing is used. Final game seeds stay sealed
until the all-layer tuning, selection and final fit are complete.

## Repository layout

- `last_layer_v1/`: baseline protocol/configuration and audited summary manifests.
- `all_layers_v1/`: full-depth compatibility checks, tuning protocol/configuration,
  selected configuration and audited summary manifests.
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

The ongoing baseline's original receipts remain at
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
The all-layer arm will also differ in its selected hyperparameters; a difference
between arms therefore does not isolate layer count as the sole cause. Learning
game interfaces alongside preservation is another limitation without a matched
interface-only training control. No internal motive follows from a game score.

The frozen last-layer baseline uses the installed trainer's inclusive sequence
length mask, which supervises one extra padding zero for the padded r4 records.
The new all-layer runner corrects this to exclude padding while retaining the
first target, EOS and final newline. This is an explicit additional recipe
difference, not a retroactive modification of the baseline or its artifacts.

No files in this directory imply that a run is complete or that GitHub has been
updated. Completion and publication require their actual audited records.
