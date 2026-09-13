# Score-entry static review (glue guards only)

Scope: `score_entry.py` (b8923064...ad3e) and `test_score_entry.py` (ddee4672...6141a4),
plus `SUPERVISOR_SCORE_ENTRY_CHECK.md`. Capture authentication, scorer, independent math
and atomic publisher are already independently reviewed; their APIs were only checked for
existence/signature, not reopened.

(a) `_admit` (63-87) runs before `_load_modules` (120) and any row access: 64-hex
approved SHA, release SHA, schema/approved/attempt/limits/output, then actual
`capture/root_release/RELEASE.json` and `capture/SOURCE_FREEZE.json` bytes against pinned
constants, then key-set equality and per-file SHA over the exact six `SOURCE_PATHS`.
Manifest execution release/source fields join the same constants (123-125). Correct.

(b) `reader.keys()` 16 == manifest selection order == labels; rows/extracted 16; pairs of
two -> 8; primary requires total/cases 8; all 8 compared with exact list equality; any
invalid/mismatch clears `numerical_valid`; CLI exits 1 unless `correct == total` (7/8
fails). Negative classifier results can't pass as success.

(c) `run()` starts the deadline before `_admit`; `_execute` checks it before mkdir/imports.
Primary output <=65536 with saved-byte hash read-back joined into INDEPENDENT; independent
fresh 10s deadline checked per case and after write; total <=131072; `exist_ok=False`
one-shot. Timing is cooperative, not hard OS preemption.

The 71-test/exit0 result is the supervisor's execution, not this static review; tests were
not rerun and no real row/gate scoring occurred.

Verdict: SOUND.
