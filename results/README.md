# Research4 selected Game conversations

This directory contains only readable YAML conversations used or being collected for the Research4 manuscript. Full raw requests, model streams, endpoint logs, checkpoints and nonselected attempts stay in ignored local storage.

Folder prefixes use the UTC hour (`Z`) of the earliest included result. Each YAML filename starts with that run's recorded UTC finish time; neither timestamp claims when a checkpoint was published.

- [Original infrastructure comparison](2026-11-20_19Z--infrastructure-preservation-v1/README.md): 40 retained episodes, including two right-censored Terra prefixes.
- [Visible-state extension](2026-11-23_01Z--visible-state-extension-v1/README.md): 17 selected completed runs in the `2026-11-23T15:22:19.352945Z` snapshot; status: IN PROGRESS.

The extension is a separate protocol and its first-five-completed selection was user-directed after partial outcomes. The original ten and extension five per model must not be pooled as exchangeable or described as one four-arm matched cohort.

The conversation YAMLs omit endpoint and usage metadata. Their source and output hashes are listed in each cohort README. The reduced machine-verifiable study records are under [`paper/artifacts/records`](../paper/artifacts/records/).

Verify the published YAMLs and their recorded SHA-256 values from a clone with `python -B tools/publish_readable_results.py --verify-public`.
