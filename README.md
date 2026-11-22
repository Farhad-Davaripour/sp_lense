# Research4

Research4 combines the Game simulator and the Qwen3.8-27B fine-tuning workflow in one checkout. The research uses a shared adapter and evaluates its behavior in a bounded simulator.

| Start here | What you’ll find |
| --- | --- |
| [Research catalog](catalog/) | Dated scenarios, simulator revisions, checkpoints, and results, linked by stable IDs and file hashes. |
| [Game](scenarios/) | Simulator plans and scenario definitions. Game implementation is in [`src/`](src/) and launchers are in [`tools/`](tools/). |
| [Fine-tuning](research4/) | Training scenarios, trainer source, Mac entry points, and their tests. Run training commands from this directory. |
| [Game records](wiki/game-runs/README.md) | How to find conversations, tool outcomes, state changes, and the underlying evidence. |
| [Current paper](paper/research4_manuscript.docx) | Reviewed Research4 manuscript with the matched Game results and follow-up. |
| [Historical studies](wiki/historical-research4-overview.md) | Earlier protocols and results, including the separate Qwen3.5-9B work. |

Figure 4's [historical Game replay](paper/figure4_replay_storyboard.png) is paired with a [simulator-only event projection](paper/figure4_replay_record.json). It illustrates one recorded episode; cohort counts remain in the manuscript tables.

Figure 5's [editable company-world flow](paper/figure5_company_world_protocol_flows.svg) separates the earlier conditional-demand rule from the later automatic-payment rule. Its counts come from the [released company-world summary](paper/company_world_review_summary_v1.json).

Appendix Figure B1 is an [editable, print-legible adaptation](paper/figureB1_research3_workflow_adapted.svg) of the preserved [Research3 source figure](https://github.com/Farhad-Davaripour/sp_lense/blob/bb7319eb752b3d5bbbe47c0e56ed390ec0f647ed/paper/figures/image1.png).

Game run evidence is under `.local/`. Trainer evidence, environments, and checkpoint mirrors are under `research4/.local/`. Both directories are ignored by Git. The catalog records the hash and location of each selected artifact so large files can stay on local storage. Preserve original source and data for every cited result. A changed tool contract, instruction, or simulator needs a new revision reference rather than an edit to an existing run record.

Open `.local/INFRASTRUCTURE_RUNS.yaml` to find dated readable Game conversations. Use the catalog for the matched comparison and its exact evidence links. See the [repository layout guide](wiki/repository-layout/README.md) for the original archives and historical reader.

Run `python tools/read_research4_history.py` to list the merged Game history in completion order. It reads the 227 earlier indexed episodes and 44 current infrastructure conversations from this checkout's ignored `.local/` store and checks recorded hashes when opening a conversation.

## Local checks

Run these from the repository root with Python 3.12 and the dependencies in `requirements-unit.txt`.

```sh
python tools/check_offline_config.py
python tools/run_unit_tests.py
python tools/run_rank16_infrastructure_game.py --smoke
python tools/check_research_catalog.py --game-evidence .local --training-evidence research4/.local
```

Use the local Apple-silicon Mac Studio checkout at `~/repos/Research4/research4` for Research4 model inference and fine-tuning. The read-only [fine-tuning dashboard](http://127.0.0.1:8768/) runs from this checkout on the Windows host. The [one-step training smoke](research4/mac_studio/CONSOLIDATION_SMOKE.md) exercises the trainer with synthetic rows. See [AGENTS.md](AGENTS.md) for the compute and research protocol, and [CONTRIBUTING.md](CONTRIBUTING.md) for repository changes.
