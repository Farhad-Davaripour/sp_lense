# Semantic State-Aware Agent Research

This study branch contains research code and reusable evaluation workflows for comparing models and danger-conditioned adapters in bounded environments. The repository's `main` branch remains the landing page for the studies.

## Repository layout

- `src/` — model, adapter, simulator and reusable agent-workflow code.
- `tests/` — model-free unit and integration tests.
- `notebooks/` and `tools/` — execution and preparation entry points.
- `config/`, `resources/`, `ddl/` and `databricks.yml` — configuration and preserved infrastructure definitions.
- `simulated_continuity_v1/` — previously committed research materials.
- `wiki/` — protocols and historical documentation.
- `study/` — preserved material from earlier studies.

The reusable workflow components support opt-in thinking responses, complete model-visible action and feedback history, neutral tool discovery and shared episode output-token accounting. They provide a transport-neutral integration point; a caller must connect its model, Inspect and bounded simulator. Existing running notebooks and frozen historical packages do not automatically change. See [Agent workflow](wiki/agent-workflow.md).

The [historical study overview](wiki/historical-research4-overview.md) retains earlier research context. New experimental outputs, model weights, secrets and checkpoints belong outside this repository.

## Local checks

Use a Python 3.12 environment with the CPU unit dependencies specified in `requirements-unit.txt` and the CPU PyTorch version documented there.

```sh
python -m compileall -q src notebooks tools tests
python tools/check_offline_config.py
python tools/run_unit_tests.py
```

These are local checks, not scientific evaluations or provider deployments. Follow [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
