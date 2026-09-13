# Capture Binding Review

Verdict: **SOUND** — model-free candidate accepted. No blocking findings.

## Checked
- Hashes independently recomputed; all match:
  - capture/SOURCE_FREEZE.json `30cbcac7e25fa8f241c23e89917fe85d324059f51cf46a994a2b45e0fe10271b`
  - capture/input_reader.py `f86512c5a6ae1832b0553298538cc8c57bf75ab99b336e07e1fdf7ae44033f8d`
  - capture/authority.py `8d0a24c8dccb7ee0727f0a957ee4d9bbb6ebfd9d63b5e9e58ebb11e1fd3f52c8`
  - diagnostic_reader.py `7b4d3509ac1a51bb3401a54f855c7d513a4e3f8e184ee25ab187ad85e5629ef9` (= PARENT_SHA256, input_reader.py:110)
- input_reader.py:154-168 `accepted_reader()`: exactly one `external_sources` pin equal to PARENT_SHA256, in-root non-symlink ≤5MiB; raw bytes sha-checked (164) **before** exec (167). Hash-before-execute and source-inventory join both hold.
- diagnostic_reader.py:405-538 `reuse_accepted_prepared_inputs()`: 19-file set (448-449) and per-blob hash (458-459); text-lock/closure joins (461-477); RESULT.json 0 loads/forwards/derivatives (481-487); fixed 16-key selector order (496, 508) via unchanged `index_contract.bind`; original output digest re-observed (513). Explicitly reuse-only; no historic TRAIN re-audited; all permission flags false.
- authority.py:5,13-17 retains 16-forward/0-derivative limits, SOURCE_FREEZE/CHECKPOINT/OWNED_IDENTITY/DATA_LOCK bindings; :19 exact trace-source binding; :20-22 joins accepted checkpoint freeze to release `fit_freeze_sha256`. Storage-fault guard at diagnostic_reader.py:214; owner/binary hashes asserted in test_06:118-121.
- Downstream contract unchanged: `build_inputs()` still yields `prechoice_diagnostic_inputs.v1` cases/`readout_selector`, `certificate_sha256`, `data_lock_sha256`; digest `8a899b39…0a7b0f` (test_01). No join blocker.
- Attempt/namespace consistent (support.py:5, fit_source_auth.py, audit_saved.py). Seven changed files, 18 byte-identical (test_06:122-127).

## Tests
59 tests, **exit 0**, 0.979s, OK; no failures/errors/skips.

## Residual prerequisite
This is model-free candidate acceptance only. A prospective commit plus root_release/RELEASE.json, pinned input/checkpoint/budget release, and independently checked real evidence remain required before any model work. All 40 scientific checks remain UNRUN.
