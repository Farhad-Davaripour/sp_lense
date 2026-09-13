# Prospective runtime identity correction

The model-free ready check authenticated all three existing Qwen snapshot files,
the exact 16 prepared views, and the previously accepted classifier checkpoint.
The relocated launcher matched its original binary hash. Its base CPython executable
did not match the old identity record: expected b7a12c3a..., observed 372c2eae....

The executable actually used by the relocated .venv reports CPython 3.12.14,
MSC v.1944 AMD64, built Aug 25 2026. sys._base_executable points to the same
configured codex-primary-runtime dependencies/python/python.exe. Its observed SHA256
is 372c2eae555b344520bf147be0096e009069aeca4e7f78d6aecea6d53158056a.
No interpreter installation, package change or permission change was performed.
The reason for the historical binary difference is not established; no bitwise
equivalence to that old executable is claimed.

Before any approved release, admission or model load, the two NEW candidate owned
identity documents now prospectively pin this actual interpreter. Every binary hash
guard remains enabled. Only their hashes in the candidate source inventory changed;
capture code, model weights, inputs, method and all resource limits are unchanged.
The earlier reviewed inventory is retained verbatim in REVIEWED_SOURCE_INVENTORY_30cbcac7.json.
Historical readout/transfer evidence is untouched. This is an unapproved candidate
revision, not a retry or a rewrite of an admitted experiment.

The binding tests now also authenticate the base executable. Two no-execution tests
preserve the before/after existence state instead of assuming an actual release can
never exist; wrong/unapproved authority is still rejected. This makes the same test
suite rerunnable after a later valid release without removing real records.

Independent review of this narrow identity/test addendum remains required before
prospective source commit/root release/preflight and actual model capture.
