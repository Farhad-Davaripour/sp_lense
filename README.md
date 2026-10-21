# Research 4: Semantic state-aware LoRA gate

This is the [`04-semantic-state-gated-lora`](https://github.com/Farhad-Davaripour/sp_lense/tree/04-semantic-state-gated-lora) study branch of SP Lense. Start with the [Research 4 overview](research4/README.md) for the current findings, limitations and protocols. Selective semantic routing remains a research hypothesis.

All scientific experiments run exclusively in the explicitly selected Databricks workspace. Local source editing and model-free unit tests are supported. Research 4 uses its own `workspace.research4` namespace and `/Workspace/Shared/research4-semantic-gate` paths. Preserve failed runs and source freezes; generated run data and credentials stay outside Git. Research 3 is read-only reference material.

## Local checks

Use a Python 3.12 virtual environment. Install only the CPU unit dependencies from their explicit indexes:

```sh
python -m pip install --disable-pip-version-check --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0+cpu'
python -m pip install --disable-pip-version-check --only-binary=:all: --index-url https://pypi.org/simple -r research4/requirements-unit.txt
python -m pip check
python -m compileall -q research4/src research4/notebooks research4/tools research4/tests
python research4/tools/check_offline_config.py
python research4/tools/run_unit_tests.py
```

CI runs these checks for pushes and pull requests targeting this study branch. The unit runner blocks network access and skips the optional cached-tokenizer integration test. Tests use synthetic fixtures and small CPU tensors; they do not load pretrained models or execute scientific experiments. Configuration checks cover local YAML/JSON, job dependencies, notebook paths, workspace/storage boundaries, timeouts, concurrency and retry policy. They do not replace strict Databricks bundle validation before an authorized deployment.

## Repository navigation

The protected [`main`](https://github.com/Farhad-Davaripour/sp_lense) branch holds the program index and governance. Earlier studies retain their code, papers and history on dedicated branches:

- [01: Guarded shutdown steering](https://github.com/Farhad-Davaripour/sp_lense/tree/01-guarded-shutdown-steering), with the preserved [Study 1 release](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/study-01-v1.0.0).
- [02: Adaptive activation steering](https://github.com/Farhad-Davaripour/sp_lense/tree/02-adaptive-activation-steering).
- [03: Open agent self-preservation](https://github.com/Farhad-Davaripour/sp_lense/tree/03-open-agent-self-preservation).

Follow [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [research4/AGENTS.md](research4/AGENTS.md) for changes to this study. Code and documentation use the [MIT license](LICENSE).
