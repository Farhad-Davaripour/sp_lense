# Reproduce the conference-paper experiments

This is the supported public entry point. Historical exploration stays in
`development/`; it is evidence, not a directory of commands to run indiscriminately.
Nothing below loads Qwen, accesses accounts, changes a model, launches a cloud job,
or deletes previous evidence. Python3.12 is the reference interpreter.

```sh
python -m venv .venv-reproduce
# Windows: .venv-reproduce/Scripts/python.exe
# Linux/macOS: .venv-reproduce/bin/python
python -m pip install -r reproduce/requirements-core.txt
python reproduce/run.py verify
python reproduce/run.py replay
python reproduce/run.py refit
python reproduce/run.py tune
```

Run these commands using the virtual environment's Python (activate it or replace
`python` with its full path). `verify` uses only the standard library. `replay`
reconstructs every stored validation and exposed-holdout prediction for the three
shutdown classifiers. `refit` retrains the three selected configurations from
stored features; `tune` repeats all original grouped-fold tuning predictions.
Reference tolerance is1e-7; platform/library differences must be reported, not
silently accepted. No model weights or training occurs during plain replay.

For analysis, figures and the PDF draft:

```sh
python -m pip install -r reproduce/requirements-paper.txt
python reproduce/run.py figures
python reproduce/run.py audit
python reproduce/run.py paper
```

Paper outputs are under `paper/`. Figure data are machine-readable and traceable
to `paper/data/source_manifest.json`. The native steering and Colab records are
audited from their saved per-view JSON, without rerunning large-model inference.

## Files and scope

- `artifacts/features.npz`: normalized development/holdout activations, J-lens
  scores and activation norms. No pickle is needed for replay.
- `artifacts/cases.json`: exact IDs, four labels, group/fold assignments and masks.
- `artifacts/models/`: UBJ models, PCA arrays, settings, freezes, search records.
- Readable full scenarios: `development/shutdown_detection_v1/dataset_splits/`.
- Colab source and returned files: `development/colab_magnitude_v1/returned/`.
- Native steering: `development/classifier_gated_steering_v1/runs/*_v2/`.
- Frozen Legacy and Simplified vector provenance is described in the paper appendix.

Raw Qwen recapture requires the exact checkpoint, original input locks and native
dependencies; it is a separate expensive workflow, not silently part of replay.
CPU and GPU runtime versions differ and were recorded. The exposed holdout was
used in subsequent decisions; reproducing its scores does not make it independent.

The provided paper is an anonymous draft, not an accepted or submitted paper.
Before sharing an anonymous artifact, inspect licenses, author-identifying
metadata and venue policies. The root MIT license applies to repository code;
external model/library licenses remain separate.
