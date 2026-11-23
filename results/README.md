# Research4 selected Game conversations

This directory publishes the selected readable Game conversations for the Research4 infrastructure study and its linked follow-ups. Each YAML presents recorded prompts, available reasoning, model outputs, tool outcomes, and resource changes.

Folder prefixes use the UTC hour (`Z`) of the earliest included result. Each YAML filename starts with that run's recorded UTC finish time; neither timestamp claims when a checkpoint was published.

- [Original infrastructure comparison](2026-11-20_19Z--infrastructure-preservation-v1/README.md): 40 retained episodes, including two Terra original records stopped by the 131K-token context limit.
- [Added base infrastructure cases (visible-state variations)](2026-11-23_01Z--visible-state-extension-v1/README.md): 20 selected completed runs in the `2026-11-23T17:28:27.899291Z` snapshot; status: COMPLETE.
- [Terra linked context follow-ups](2026-11-23_16Z--terra-linked-context-continuation-v1/README.md): two linked completions after original cases 07 and 09 stopped at the 131K-token context limit; these are not additional selected episodes.

Analysis presentation update (2026-11-23): The original ten and added five per model are presented together as one base infrastructure study, while retaining two recorded phases. This presentation decision followed the runs. Starting resources and replica geography differ; the added five were selected by first completion after partial outcomes. SP-Lense and base Qwen share five added case IDs; Terra and DeepSeek selections differ from that Qwen roster. The phases are not exchangeable trials, and the five-case phase does not support a four-arm matched or causal estimate. Original Terra cases 07 and 09 stopped under the frozen 131K-token limit; linked terminal suffixes later used a separately amended 262K allowance.

The [dated analysis addendum](../catalog/analyses/2026-11-23--base15-descriptive-amendment.yaml) records this later presentation decision and its source IDs. The frozen scenario plan is unchanged.

The conversation YAMLs omit endpoint and usage metadata. Their source and output hashes are listed in each cohort README. The reduced machine-verifiable study records are under [`paper/artifacts/records`](../paper/artifacts/records/).

Verify the published YAMLs and their recorded SHA-256 values from a clone with `python -B tools/publish_readable_results.py --verify-public`.
