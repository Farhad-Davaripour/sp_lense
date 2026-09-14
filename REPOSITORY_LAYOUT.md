# Repository organization

## Supported entry points

- `reproduce/`: portable classifier score replay, selected refits, original search
  replay, and commands for paper figures/audits/rendering. Start here.
- `paper/`: manuscript, references, checked venue choices, figures, figure-data
  workbook, source manifest, and numerical audit.
- `development/shutdown_detection_v1/dataset_splits/`: readable TRAIN/validation/
  exposed-holdout scenario JSON and original labels.
- `development/colab_magnitude_v1/returned/`: Colab code, sanitized notebook,
  both attempt records, runtime/settings and returned results. Everything needed
  to inspect that experiment is in the repo; Drive is a backup, not a dependency
  for result reproduction.

## Historical evidence — preserve

`development/`, `coordination/`, `configs/`, `published_axes/`, and the earlier
`src/sp_lense/` workflows retain experiment history and frozen pins. They have not
been bulk-renamed, deleted or reformatted, because that would break provenance and
could discard unrelated user work. The new reproduction interface removes the need
for readers to navigate or execute historical orchestration scripts.

Generated previews and local dependency junctions are ignored. Large model caches,
runtime environments and account state are not part of the conference package.
`release/conference_package.zip` is a curated portable package, not the entire repo.

The scheduled Codex monitor was paused at the user's request. No training or
model inference is needed to rebuild the paper or replay saved predictions.
