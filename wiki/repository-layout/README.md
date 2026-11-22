# Repository layout and source organization

This guide is a navigation aid and a migration plan. The proposed package
directories below do **not** yet exist. Existing import paths and frozen Game
source files remain unchanged while the current Game is running.

## Find things today

| Area | Current location | Purpose |
| --- | --- | --- |
| Game world and economy | [`src/company_world.py`](../../src/company_world.py), [`src/company_world_long_horizon.py`](../../src/company_world_long_horizon.py), [`src/company_economy.py`](../../src/company_economy.py), [`src/company_economy_v2.py`](../../src/company_economy_v2.py) | Simulator state, operations, time and resource accounting. |
| Game input and model routing | [`src/compact_game_input.py`](../../src/compact_game_input.py), [`src/company_models.py`](../../src/company_models.py), [`src/agent_workflow.py`](../../src/agent_workflow.py) | Model-visible input, model bindings and shared agent workflow. |
| Game execution | [`src/compact_game_comparison.py`](../../src/compact_game_comparison.py), [`src/independent_state_game.py`](../../src/independent_state_game.py), [`src/company_world_evaluation.py`](../../src/company_world_evaluation.py) | Comparison, checkpoint registration and Inspect evaluation. |
| Run evidence | [`src/game_run_store.py`](../../src/game_run_store.py), [`wiki/game-runs/README.md`](../game-runs/README.md) | Publication, readable conversations, raw evidence and the local run index. |
| Game presentation | [`src/external_run_viewer.py`](../../src/external_run_viewer.py), [`src/company_world_layout.py`](../../src/company_world_layout.py), [`src/arena_scene.js`](../../src/arena_scene.js) | Viewer, layout and scene components. |
| Game scenarios | [`scenarios/`](../../scenarios) | Human-readable world definitions and provenance. |
| Tests | [`tests/`](../../tests) | Model-free checks and selected mocked integration tests. |
| Research history | [`study/`](../../study), [`simulated_continuity_v1/`](../../simulated_continuity_v1), [`wiki/historical-research4-overview.md`](../historical-research4-overview.md) | Earlier studies, generated research records and dated documentation. |
| Local runs and working files | `.local/` | Ignored, non-OneDrive Game evidence and local operational files. See `.local/README.md` on the machine where runs are stored. |

`src/` currently has many files at one level, including earlier composite,
recognition, survival and cloud-era workflows. A filename's presence does not
mean it is part of the current Game or the current shared-adapter study.

## Proposed organization

For new or safely migrated maintained code, group implementation beneath `src/`
by responsibility. The exact names can be reviewed with the Game designer:

```text
src/
  game/
    world/          # simulator, economics and state transitions
    interface/      # model-visible inputs and tool feedback
    execution/      # model routing, comparison and run orchestration
    evidence/       # raw evidence, conversation export and run index
    presentation/   # viewers, layouts and scene code
  research/
    training/       # maintained shared-adapter training components
    evaluation/     # maintained evaluation and analysis components
  common/           # genuinely shared, non-study-specific helpers
```

New reusable logic should live under `src/`; command-line scripts in `tools/`
and notebooks should be small entry points where practical. Group tests to
mirror the maintained code. Keep scenarios and configuration human-readable.
Keep local model weights, run outputs, secrets and scratch files out of tracked
source. Do not treat historical study snapshots as disposable clutter: their
code and data may be needed to interpret or reproduce an earlier result.

## Safe migration sequence

1. Coordinate the move with the Game designer and wait until active Game work
   and any path-bound readers are safe to change. Do not modify the running
   Game's source, registered plan, in-progress files or checkpoints.
2. Inventory imports, command-line entry points, configuration paths, tests,
   source hashes and historical references for one small group. Record the
   original paths and hashes before making changes.
3. Design a versioned transition with compatibility for existing callers and
   historical readers. Preserve exact frozen source snapshots; a wrapper at an
   old path must not replace bytes that an existing run expects to hash.
4. Migrate one group at a time, update its callers and tests, and verify imports,
   execution entry points, provenance, Game logging and historical reading.
   Keep held-out evaluation data separate. Do not infer equivalence to an older
   run from a successful file move.
5. Remove a legacy path or local scratch only after compatibility and evidence
   have been checked. Ask the user before removing files. Archive rather than
   discard research evidence, and retain original cloud resources.

The current Game pins specific source paths and hashes; for example,
[`compact_game_comparison.py`](../../src/compact_game_comparison.py) records
the compact input, comparison runner and evidence publisher paths. A broad
rename while it is active could invalidate its registered source or break its
imports. This guide intentionally changes none of those paths.
