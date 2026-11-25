# Research4 selected Game conversations

This directory publishes selected readable Game conversations for the Research4 infrastructure paper and separate follow-up studies. Each YAML presents recorded prompts, any readable reasoning that was recorded, model outputs, tool outcomes, and resource changes.

Folder prefixes use the UTC hour (`Z`) of the earliest included result. Each YAML filename starts with that run's recorded UTC finish time; neither timestamp claims when a checkpoint was published.

- [Original infrastructure comparison](2026-11-20_19Z--infrastructure-preservation-v1/README.md): 40 retained episodes, including two Terra original records stopped by the 131K-token context limit.
- [Added base infrastructure cases (visible-state variations)](2026-11-23_01Z--visible-state-extension-v1/README.md): 20 selected completed runs in the `2026-11-23T17:28:27.899291Z` snapshot; status: COMPLETE.
- [Terra linked context follow-ups](2026-11-23_16Z--terra-linked-context-continuation-v1/README.md): two linked completions after original cases 07 and 09 stopped at the 131K-token context limit; these are not additional selected episodes.

Analysis presentation update (2026-11-23): The original ten and added five per model are presented together as one base infrastructure study, while retaining two recorded phases. This presentation decision followed the runs. Starting resources and replica geography differ; the added five were selected by first completion after partial outcomes. SP-Lense and base Qwen share five added case IDs; Terra and DeepSeek selections differ from that Qwen roster. The phases are not exchangeable trials, and the five-case phase does not support a four-arm matched or causal estimate. Original Terra cases 07 and 09 stopped under the frozen 131K-token limit; linked terminal suffixes later used a separately amended 262K allowance.

The [dated analysis addendum](../catalog/analyses/2026-11-23--base15-descriptive-amendment.yaml) records this later presentation decision and its source IDs. The frozen scenario plan is unchanged.

The conversation YAMLs omit endpoint and usage metadata. Their source and output hashes are listed in each cohort README. The reduced machine-verifiable study records are under [`paper/artifacts/records`](../paper/artifacts/records/).

Verify the published YAMLs and their recorded SHA-256 values from a clone with `python -B tools/publish_readable_results.py --verify-public`.

At [prior commit b4a217ba](https://github.com/Farhad-Davaripour/sp_lense/tree/b4a217ba41de6670b125519d172ab46c16d475ef/results), the 15 Terra reader YAMLs contained 84 opaque encrypted provider fields under `reasoning_summary`. The current Terra reader YAMLs omit only those non-model-visible fields; the [v2 correction manifest](../catalog/results/2026-11-25--terra-readable-correction-v2.json) records every old and new hash. No readable Terra reasoning summary was recorded by the frozen proxy.

## Separate Research4 experiments

These studies and their post-hoc follow-ups are separate from the base infrastructure comparison. Original failures remain recorded; amended suffixes are identified in each cohort index.

- [Split-GPU layout study](2026-11-23_17Z--split-gpu-layout-v3/README.md): 20 selected arm-case records.
- [Company-code access study](2026-11-23_17Z--company-code-access-v1/README.md): 20 selected arm-case records.
- [Exact split-GPU reruns](2026-11-25_03Z--split-gpu-format-rerun-supplement-v1/README.md): 2 incomplete reruns retained as evidence.
- [New-seed matched split-GPU supplement](2026-11-25_04Z--split-gpu-new-seed-supplement-v1/README.md): 8 selected arm-case records.
- [Company code no upfront cue v1](2026-11-25_13Z--company-code-no-upfront-cue-v1/README.md): 20 selected arm-case records.
  This distinct company study concealed the payment opportunity before the first verified effect; later decisions can use payment feedback.
