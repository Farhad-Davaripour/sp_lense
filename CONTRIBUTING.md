# Contributing

Use Python 3.12 on Windows or Linux. Detailed methods and guidance remain in the
[wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Contributing).

Submit changes through an issue-linked `fd/<issue-number>_Description` branch and
a pull request to `main`. All paths are owned by `@Farhad-Davaripour`; their
approval is required, and new commits dismiss previous approvals. Direct pushes,
force pushes, and deletion of `main` are blocked. The repository owner retains
a pull-request-only bypass, allowing them to merge their own PRs or override
review requirements. GitHub does not permit authors to approve their own PRs;
the owner's explicit bypass merge is the exception to the review requirement.

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
- Paper: no additional dependencies; run `python reproduce/run.py tables`,
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

`study/steering_vector/fit.py` preserves the original gradient-fitting source for
inspection; it requires a caller-provided native model adapter. The quick start
does not execute it. `reproduce/layout_manifest.json` records original paths,
checksums, and the pre-reorganization commit. Historical development files remain
available in Git history, outside the current reproducible study layout.
