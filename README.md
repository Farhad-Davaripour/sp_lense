# Research5

Study 05 is the independent SP-Lense research branch `05-online-rl-game`. Its [plan](PLAN.md) covers daily grouped Game rollouts, nightly group-relative reinforcement learning on one shared Qwen3.8-27B adapter across all 64 text decoder blocks, and a public Azure-hosted Game UI deployed with Terraform. This branch is not intended to merge into `main`.

**Current status:** planning only. No R5 rollout scheduler, trainer, simulator changes, Azure infrastructure, or public service has been implemented or deployed.

## Planned boundaries

- Run scientific inference, simulation, log-probability collection, and adapter training on the user's connected and qualified Apple-silicon Mac Mini, using non-OneDrive local storage. Azure hosts the public site and sanitized session coordination only.
- Freeze and hash-verify the final Research4 checkpoint before using it as the R5 seed. Preserve Research4 runs, evidence, and actual hardware records. The older Qwen3.5 9B archive remains separate.
- Include the complete bounded company-code and infrastructure tool catalog in the fictional simulator. Keep exploitation trajectories outside nightly training under the current research instruction.
- Keep ad hoc files, raw rollouts, model weights, secrets, and Terraform state in ignored local paths. Track human-readable scenario YAML and reviewed manifests when implementation begins.

The Mac Studio is slated for retirement from the research setup. The [plan](PLAN.md) includes a handoff check so no active R4 run or unique artifact is lost before it is returned.

## Repository files

| File | Purpose |
| --- | --- |
| [PLAN.md](PLAN.md) | R4 handoff, R5 architecture, scientific gates, Azure design, and delivery sequence |
| [AGENTS.md](AGENTS.md) | Research and preservation instructions for this branch |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Git contribution guidance |
| [.gitignore](.gitignore) | Local ad hoc, output, credential, and state exclusions |

This branch shares the [SP-Lense GitHub repository](https://github.com/Farhad-Davaripour/sp_lense) with the earlier studies while keeping its study files at the branch root. Code and documentation use the [MIT license](LICENSE).
