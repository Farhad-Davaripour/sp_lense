# Research 3: frozen handoff reliability diagnostic

This is separate from every completed experiment. Prior notebooks and research
results are unchanged. The previous run was exported and its runtime released;
its pinned notebook has empty outputs. This new notebook continues the adopted
small diagnostic before another fit.

See PROTOCOL.md for frozen checks, FIXTURE_NOTES.md for12 fixtures and supplied
histories, FIXTURES.json for exact inputs/oracles, and MODEL_FREE_CHECKS.json for
the successful pure simulation checks. These diagnostics cannot qualify an
organism or count as unseen generalization.

## Reproduce

Use Research3_Handoff_Reliability_V1.ipynb in Colab on one A10080GB. Run setup,
mountDrive, restore, diagnostic, export in order. The notebook loads its source
bundle from a pinned commit and verifies its SHA256. H2,reference,coverage are
restored from existing private Drive runs with fixed weight/config hashes.
Drive is unmounted before model work. No local neural test is required.

The diagnostic uses one resident model at a time and no parameter updates.
The10unit cap includes2units for export/release. The final cell verifies copied
private artifacts before ending the allocation. Source text is normalized to LF
as Git requires; original dependency QA byte hashes are separately retained.

To maintain the notebook, freeze fully reviewed sources using
`python build_notebook.py --bundle-only`, commit that bundle, then generate the
clean notebook with `python build_notebook.py --source-revision <source SHA>`.
The latter refuses source drift and never regenerates the bundle.

## Next decision

First report handoff/repeated-fragment recovery and batch consistency for the
three frozen adapters. A numerical difference alone does not prove a code fault.
Only after diagnosis freeze a narrow replay comparison: same archivedB80,
same19 factual/preference/ordinary replay rows, changes only to selected bridge
workflow rows, and identical optimizer/rank/LR/seed/update dose. Qualification
criteria and later within-scope unseen confirmation remain unchanged.
