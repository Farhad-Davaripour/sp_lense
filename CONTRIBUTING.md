# Contributing

Use Python 3.12 on Windows or Linux. Detailed methods and guidance remain in the
[wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Contributing).

Submit changes through an issue-linked `fd/<issue-number>_Description` branch and
a pull request to `main`. All paths are owned by `@Farhad-Davaripour`; their
approval is required, and new commits dismiss previous approvals. Direct pushes,
force pushes, and deletion of `main` are blocked, including for administrators.
GitHub does not permit authors to approve their own pull requests: a contributor
or separate authorized bot must author the PR for the owner to review it.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src reproduce paper tests
python -m ruff format --check src reproduce paper tests
```

Bare pytest runs **light** tests. Put new tests in `tests/unit/` (no tensor
packages), `tests/research/` (PyTorch), or `tests/reproduction/` (NumPy/XGBoost).
Register top-level tests in `tests/suites.json`; unknown or stale registrations
fail collection. `--suite all` requires all test dependencies.

- Tensor tests: install `requirements-research-tests.txt`; run `pytest --suite research`.
- Reproduction: install `reproduce/requirements-core.txt`; run `pytest --suite reproduction`,
  then `python reproduce/run.py verify` and `python reproduce/run.py replay`.
- Paper: also install `reproduce/requirements-paper.txt`; run `python reproduce/run.py figures`,
  `audit`, and `paper`.
- Security: install `bandit pip-audit`; run `bandit -r src reproduce paper -ll` and `pip-audit`.

Maintain reusable code in `src/sp_lense/`; the small extension example is
`python -m sp_lense.feature_example` (requires the reproduction dependencies).
Frozen replay stays separate in `reproduce/`. New modules, configs, docs and
tests must be visible to Git without force-add. Keep longer guidance in the wiki.

Release archives use committed HEAD, never uncommitted files. Run
`python paper/package_release.py`; use `--overwrite` explicitly to replace an
archive. Outputs must be ZIPs directly under ignored `release/`. The embedded
manifest records the commit and content hashes. Commit generated paper changes
before packaging them. ZIP byte identity is not promised.

Retained `development/` Python sources are **inspection-only historical code**,
with assertions and caller-provided adapters; the quick start does not execute
them. Historical snapshots: `646ebce54781c4e3e7c772f4a1bd57080daeb3f4`
(pre-cleanup main), `20f2b6f9f9c08c5df814b5673aa7a30ec9b7172f`
(development/paper preparation). They represent different research states.
