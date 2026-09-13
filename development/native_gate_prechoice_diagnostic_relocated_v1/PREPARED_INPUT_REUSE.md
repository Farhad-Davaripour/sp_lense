# Prepared diagnostic inputs: checked reuse candidate

The two DeepSeek authoring attempts ended at their immutable operational limits.
The first wrote no source/tests; the second wrote the implementation and five tests,
but stopped before inspecting their failures. Both SDK owners closed/exit0 and their
PIDs are absent. No third identical authoring attempt was launched.

Supervisor diagnosis found one concrete production error: expected case keys were
compared to the preparation manifest's filenames without adding .json. Fixing that
one set expression made all five new tests pass (0.034s). Original hashes, file set,
payload, selector mathematics and expected output digest were unchanged.

Supervisor then restored the inherited 5MiB per-file read cap and explicit resolved
in-repository containment. A full regression found a brittle old test rejecting the
word forward even in the metadata assertion real_model_forwards == 0. It now checks
actual forbidden call names, retaining the other file-name restrictions and the new
runtime provider/activation/OneDrive read tripwires. No forbidden production call was
removed or hidden; a direct forbidden-call canary checks the test itself.

## Verification

From this directory, using the relocated .venv/Scripts/python.exe -E -S -B:

    -m unittest test_prepared_input_reuse test_checkpoint_reuse test_diagnostic_scoring test_diagnostic_reader test_reader_pins test_reader_join test_math_equivalence

52 tests passed, exit0, 0.770s; no failures/errors/skips. Earlier failed diagnostic
invocations remain reported above, not recast as successful worker completions.
Historical tracked readout/transfer directories have no Git diff.

All19 preparation blobs plus pinned release/text/closure/data-lock are authenticated.
The unchanged original selector binder reconstructs16 prepared diagnostic views with
exact previously fixed digest:
8a899b396d504cc0ccbac2e373cc0e933329803478f6320fc97cbc305c0a7b0f.

Final SHA256: reader 7b4d3509ac1a51bb3401a54f855c7d513a4e3f8e184ee25ab187ad85e5629ef9;
new test 73e71117cad63e6e9e81b093e41729ea0857a2091bcb1f5f27182863785ca89c;
existing reader test 0f7bf26da5a1473debc4a3e8c33b6cbbcd7957b1ffb484bcca1745f7d9cebe5f.

This reuses accepted prepared bytes, not a fresh replay of old preparation authorizers.
Filesystem checks are point-in-time, not a concurrent mutation security guarantee.
No actual classifier model capture, scoring, training, tokenizer or real release ran.
All40 scientific checklist items remain UNRUN. Next: independently accept the input
reuse boundary and connect existing capture/owner code through minimal new bindings;
separate prospective source/input/budget release is still required before any model.
