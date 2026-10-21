# SP Lense

SP Lense is a research program on detecting and steering model behavior through internal representations. The `main` branch is intentionally limited to navigation and repository governance. Research code, data, experiment records, and papers live on dedicated study branches.

## Studies

| Study | Status | Branch | Stable release |
| --- | --- | --- | --- |
| Study 01 Guarded shutdown steering | Completed | [`01-guarded-shutdown-steering`](https://github.com/Farhad-Davaripour/sp_lense/tree/01-guarded-shutdown-steering) | [`study-01-v1.0.0`](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/study-01-v1.0.0) |
| Study 02 Adaptive activation steering | Preserved original | [`02-adaptive-activation-steering`](https://github.com/Farhad-Davaripour/sp_lense/tree/02-adaptive-activation-steering) | Paper commits below |
| Study 03 Conditional gating and process preservation | Active; published snapshots | [`03-open-agent-self-preservation`](https://github.com/Farhad-Davaripour/sp_lense/tree/03-open-agent-self-preservation) | [`research3-paper-mcp-v2`](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-paper-mcp-v2) |
| Study 04 Semantic state-aware LoRA gating | Active | [`04-semantic-state-gated-lora`](https://github.com/Farhad-Davaripour/sp_lense/tree/04-semantic-state-gated-lora) | Not released |

To inspect or reproduce a study, switch to its branch and follow that branch's README. Study 04's maintained entry point is [`research4/README.md`](https://github.com/Farhad-Davaripour/sp_lense/blob/04-semantic-state-gated-lora/research4/README.md).

Published paper citations identify exact commits: [Study 01 `7ebd5d5e5c7c`](https://github.com/Farhad-Davaripour/sp_lense/commit/7ebd5d5e5c7c934879e9881a9ba993741d6c029a), [Study 02 `090078cf75c6`](https://github.com/Farhad-Davaripour/sp_lense/commit/090078cf75c611b3b846746ef47759358d17e8cd) and its [earlier `f71d6de34660` snapshot](https://github.com/Farhad-Davaripour/sp_lense/commit/f71d6de346608df374aaa3263ede24823aec66c0). The completed Study 01 [paper](https://github.com/Farhad-Davaripour/sp_lense/blob/7ebd5d5e5c7c934879e9881a9ba993741d6c029a/paper/paper.pdf) remains at its cited snapshot; the earlier reproducibility package remains at `study-01-v1.0.0`. Existing release tags and cited commits are preserved while study branches may continue.

Research 3 also publishes [primary reproduction artifacts](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-reproduction-v1) and a [steering diagnostic supplement](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-steering-diagnostic-v1). Follow its README for saved-evidence replay and model-experiment instructions.

## Branch model

- `main` is the protected landing and governance branch.
- Numbered `<number>-<name>` study branches contain research-specific work and retained study history. Preserve these long-lived branches and their published snapshots.
- `archive/*` branches retain historical work.
- `fd/<issue-number>_<description>` branches are short-lived feature branches. They target the relevant long-lived branch, are squash-merged, and are deleted from the remote after a successful merge.

Repository-wide documentation is maintained in the [wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki). Proposed work and review history are tracked in [issues](https://github.com/Farhad-Davaripour/sp_lense/issues) and [pull requests](https://github.com/Farhad-Davaripour/sp_lense/pulls).

## Research compute

Use Databricks exclusively for scientific experiment execution, in the explicitly selected workspace. Local source editing and unit tests are allowed. Preserve existing study artifacts and workspace resources. An unavailable GPU does not authorize moving experiments to Google Colab or another platform.

Code and documentation are released under the [MIT license](LICENSE).
