# Capture executor independent review V1

- **Job ID:** `capture_executor_review_20260914_0610`
- **Scope:** read-only review of `capture_executor.py`, `test_capture_executor.py`,
  `CAPTURE_EXECUTOR_SUPERVISOR_REPAIR_V2.md`, `CAPTURE_EXECUTOR_HANDOFF_V1.md`,
  `tokenizer_input_adapter.py`, `capture_export.py`, `native_capture_contract.py`
  and the directly used fabricated helpers (`test_native_capture_contract.make_case`,
  `test_tokenizer_input_adapter.FakeTokenizer`). No source was edited. No dataset,
  cache, model, tokenizer, file, network, fit or HOLDOUT access occurred.
- **Verdict:** **scoped PASS** for the stated executor contract, with two
  low-severity robustness escapes recorded below.

## Native hashes (recomputed)

- `capture_executor.py`:
  `5142873B7AC110DBB27E1E0A14317D301CEB0124E37F887B9CA58765CB567775`
- `test_capture_executor.py`:
  `5EAF06077362204C786BA5457A94B7D48CD75D512818AFCD1845D8247772F948`

Both match `CAPTURE_EXECUTOR_SUPERVISOR_REPAIR_V2.md`. V1 handoff hashes
(`BA29…`, `CCB4…`) are stale pre-repair values, consistent with the note that V1
is preserved unchanged.

## Reproduction

- `.\\.runtime\\Scripts\\python.exe -W error test_capture_executor.py -v` →
  **Ran 22 tests, OK**, 0 failures/errors.
- `-W error -m unittest` over executor + adapter + export + contract + pair_join →
  **Ran 75 tests, OK** (handoff's 71 covered a narrower set).

## Adversarial probes (fabricated, no real artifacts)

Confirmed by probe: numeric-only ints reach `capture_view`; the AB/BA call
sequence equals the adapter-prepared `input_ids` and indices exactly (full-list
equality, not just the label boundary); all-case preparation precedes any
capture (an unadmitted second case yields `INPUT_PREPARATION_ERROR` with zero
capture calls); `validate_result` failure reserves the attempt
(`attempted=1, succeeded=0, failed=1`); callback-supplied diagnostics cannot
erase or forge counters; deadline loss through adapter wrapping is preserved
(`DEADLINE_EXCEEDED`, `phase=after_decode` / `after_encode` / `before_encode`,
`attempted` correct); exact-minimum budgets pass and one byte below fails
pre-callback; non-copyable/large bounds behave; callback mutation of `input_ids`
cannot alter receipt hashes; a failed BA capture returns nothing and leaves
caller cases unchanged.

Probe exits (minor robustness escapes; not core-contract breaks):

1. A `clock` that raises (e.g. `RuntimeError`) or returns a non-numeric value
   leaks the raw exception from `run.start()`/`check_deadline`. The constructor
   only asserts `callable(clock)`, and the blanket docstring claim that "any
   error aborts the whole run … raises `CaptureExecutorError`" is not honoured
   for this injected control input.
2. `deepcopy(cases)` (line 279) runs before `_screen_cases`; a case object whose
   `__deepcopy__` raises leaks the raw exception instead of a structured code.

Neither affects well-formed admitted cases or the enumerated rejection classes;
both are narrow, fail-closed robustness gaps in malformed injected callables.

One intent-confirming nuance: `run.call_capture` deliberately discards
callback-supplied diagnostics, so a user callback that legitimately raises
`CaptureExecutorError` loses its own `phase` but keeps `code`/`detail` and the
executor's charged counters. This matches the repair note and is not a defect.

## Contract points checked

- Bounds and all-case preparation precede every `capture_view` call.
- Numeric-only copied callback inputs; paired AB/BA order fidelity.
- Exactly one reservation per attempt, two per case, never retried, including
  when the callback raises `CaptureExecutorError`.
- Normalized preparation failures with zero-forward diagnostics;
  per-encode/decode/capture deadlines bracketed around each callback with
  adapter exception wrapping; flag/feature validity; no partial success.
- Flags, identity pins and native provenance are caller assertions; the receipt
  marks them as such and claims no native authenticity.

## Documented limits — not failures

Raw-payload budget excludes Python object memory; deadline is cooperative only;
the receipt contains raw `bytes` payloads so it is not JSON-serializable as a
whole (`decoded_views`/sidecars are); flags/identity are caller assertions. All
accepted as documented. Static import boundary holds (stdlib + the two reviewed
modules only).

**Conclusion:** repairs verified; scoped PASS. Real native execution remains
locked and unauthenticated by these synthetic tests.
