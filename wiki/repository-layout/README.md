# Research4 repository layout

`Research4/` is the consolidated Windows checkout. Work on the Game from its root and on the trainer from `research4/`. Each keeps the source paths used by its existing entry points. Shared Game helpers live in `src/common/`, and the existing import paths still resolve. The offline test runner implementation is in `src/ci/`.

| Area | Location |
| --- | --- |
| Study chronology and artifact IDs | [`catalog/`](../../catalog/) |
| Game scenario definitions and run plans | [`scenarios/`](../../scenarios/) and [`config/`](../../config/) |
| Game tools, inputs, simulator, and viewers | [`src/`](../../src/) and [`tools/`](../../tools/) |
| Fine-tuning code and Mac entry points | [`research4/src/`](../../research4/src/) and [`research4/mac_studio/`](../../research4/mac_studio/) |
| Fine-tuning scenario definitions and recipes | [`research4/scenarios/`](../../research4/scenarios/) and [`research4/config/`](../../research4/config/) |
| Local Game evidence and readable transcripts | `.local/` |
| Local trainer receipts, data, and checkpoint mirrors | `research4/.local/` |

For the consolidated baseline, the Game source came from `88b1386737978f2d5da8cf2bc49df41870a01ffc` and the tracked `research4/` subtree came from `c71118bee785c66539c6de0ce389aafb6de989cf`. The consolidation also captures the reviewed current Game launcher and the trainer's maintained Mac entry points. The original Git worktrees retain the remaining working files and the exact frozen historical sources.

## Working with recorded runs

Use the [catalog](../../catalog/README.md) for the current matched study and the [Game storage guide](../game-runs/README.md) for the readable local format. Treat hashes recorded by a run as references to its original bytes. Any changed instructions, tool contracts, simulator source, or model configuration need a fresh study revision and review before execution.

The Windows archive convention is sibling `superseded/sp_lense4` for the old trainer and `superseded/sp_lense4_worktrees` for the old Game and earlier worktrees. The 227 earlier indexed episodes were copied and hash-checked into this checkout's ignored `.local/` store. `python tools/read_research4_history.py` lists those episodes alongside the 44 current infrastructure conversations. Pass `--episode <id>` for an earlier indexed episode or `--bundle <id>` for an infrastructure conversation. The original archive index remains available through `tools/read_archived_game.py` for provenance checks.

Keep credentials, caches, trace logs, and adapter tensors in ignored local storage. The selected rank-16 checkpoint is reachable from its digest and locator in the catalog. Existing Mac model and environment paths remain the storage dependencies for current local inference and training. Old path-bound launch authorities are research records. A deliberate, separately reviewed run can bind new source and paths when a new experiment needs them.
