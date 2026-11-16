# Test layout and offline checks

Run `python tools/run_unit_tests.py` after installing the CPU dependencies in
`requirements-unit.txt` (including the mocked Game client dependencies). Create
the ignored `.local/` directory first if it does not exist; CI does this before
running the tests. The runner blocks network connections and does not start a
model or a Game.

- `test_*.py` contains the portable unit and mocked-integration tests.
- `fixtures/` contains inert synthetic test data and builders. These do not
  contain valid grants, credentials, model weights or private Game cases.
- `support/` contains shared test helpers, including the explicit local-only
  prerequisite check for historical exact-source tests.

Some historical integration tests verify exact HELD contract or tokenizer
bytes that intentionally remain outside the repository. If those inputs are
absent, only the affected tests report a **local-only exact-source integration**
skip. When the files are present, the original hash and guard assertions still
run and changed bytes fail; a CI pass without them is not evidence that the
historical artifacts were verified. Synthetic fail-closed tests and the
portable suite run without those local files.

Do not copy actual grants, credentials, held-out data, model weights or
checkpoints into this directory to make CI pass. Record the local integration
result separately when that exact evidence is required.
