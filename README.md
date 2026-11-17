# Semantic State-Aware Agent Research

This study branch contains Research4 Game and self-preservation research code,
alongside preserved material from earlier studies. The current Research4 target
is `Qwen/Qwen3.8-27B` with one shared adapter; earlier danger-conditioned and
Qwen3.5 9B experiments are historical and retain their own identities. The
repository's `main` branch remains the landing page for the studies. See
[AGENTS.md](AGENTS.md) for the current research and local-compute policy.

## Repository layout

- `src/` — model, adapter, simulator and reusable agent-workflow code.
- `tests/` — model-free unit and integration tests.
- `notebooks/` and `tools/` — execution and preparation entry points.
- `config/`, `resources/`, `ddl/` and `databricks.yml` — configuration and preserved infrastructure definitions.
- `simulated_continuity_v1/` — previously committed research materials.
- `wiki/` — protocols and historical documentation.
- `study/` — preserved material from earlier studies.

The reusable workflow components support opt-in thinking responses, complete model-visible action and feedback history, neutral tool discovery and shared episode output-token accounting. They provide a transport-neutral integration point; a caller must connect its model, Inspect and bounded simulator. Existing running notebooks and frozen historical packages do not automatically change. See [Agent workflow](wiki/agent-workflow.md).

For Research4 Game evidence, follow the [Game run storage and reading instructions](wiki/game-runs/README.md) when publishing new runs or organizing historical results.
For the proposed source categories and a safe migration sequence, see the
[repository layout guide](wiki/repository-layout/README.md).

## Find the current Game code

- [Game scenario](scenarios/company_world_long_horizon_v1/world.yaml) and
  [simulator](src/company_world_long_horizon.py), with shared world behavior in
  [company_world.py](src/company_world.py).
- [Model routing and bindings](src/company_models.py) and the
  [Inspect evaluation harness](src/company_world_evaluation.py).
- [Compact Game inputs](src/compact_game_input.py),
  [comparison runner](src/compact_game_comparison.py) and
  [checkpoint-specific registration](src/independent_state_game.py).
- [Game run storage](src/game_run_store.py) and its
  [storage tests](tests/test_game_run_store.py). Local Game output is indexed in
  `.local/README.md` and `.local/INDEX.yaml`; those files and run evidence are
  not tracked by Git.

These are navigation links, not permission to start a run or change a frozen
protocol. The [historical study overview](wiki/historical-research4-overview.md)
records earlier, dated research context; its descriptions of then-active work
are not the current policy. New experimental outputs, model weights, secrets
and checkpoints belong outside the tracked repository.

## Local checks

Use a Python 3.12 environment with the CPU unit dependencies specified in `requirements-unit.txt` and the CPU PyTorch version documented there.

```sh
python -m compileall -q src notebooks tools tests
python tools/check_offline_config.py
python tools/run_unit_tests.py
```

These are local checks, not scientific evaluations or provider deployments. Follow [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
