# Published evidence

`confirmatory/` contains the post-review audit produced from settings committed before the
new outputs were viewed:

- `CONFIRMATORY_RESULT.md`: short human-readable outcome
- `confirmatory_summary.json`: criteria, statistics, model provenance, and per-case effects
- `confirmatory_rows.jsonl`: all 576 next-token measurements

The saved 1024-value axis is not duplicated here. Reproduce it from the discovery pipeline
or point `sp_lense.confirmatory_audit` to the local `sp_choice_axis.pt`. Model weights,
virtual environments, activation caches, and exploratory `results/` runs remain ignored
because they are large and reproducible from the tracked code/configuration.
