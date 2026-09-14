# CAPTURE_EXPORT_INDEPENDENT_REVIEW_V2

- Job: `capture_export_review_20260914_0321`
- Independent, read-only review. No source edits. No real datasets/vectors, HOLDOUT, model, tokenizer, cache, fits, installs, network, Git, coordination, config, or subagents used. Only synthetic in-memory probes.
- **Verdict: PASS** (one non-blocking typing note).

## Reviewed artifacts and native hashes

| File | SHA256 |
|---|---|
| capture_export.py | `1B9C44CEA5B80658F9F655A0CCE54C360610E079B0EEDB1EEF246B262C505DE8` |
| test_capture_export.py | `32F05E1C0CC0B6A9356A948C22508CC5CAE9B41101D27F744739E292D06625D9` |
| pair_join.py | `2598226122BCD503E52F5D2B744812483A9F53EEE6D18B4BB3F57615223B9881` |
| test_pair_join.py | `2E89179BC9A1F59B5C20E8FF264B9E419CFDC6FABDD6EE23F582E5CB4C4678C1` |
| native_capture_contract.py | `8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391` |
| test_native_capture_contract.py | `1FF0CD16739601E40BB4B5D38A9FED2199FB895BD17D59EEF159D9C0D535A739` |

The two handoff-listed hashes (`capture_export.py`, `test_capture_export.py`) match exactly.

## Executed test outcome

Command, repo root, `PYTHONPATH=development/classifier_generalization_v2`, study `.runtime` Python 3.12.14 / numpy 2.5.3, warnings as errors (`-W error`):

`python -m unittest -v test_capture_export test_pair_join test_native_capture_contract`

Result: **Ran 44 tests in 0.147s — OK, exit 0** (27 + 5 + 12, matching the handoff counts). No failures, errors, skips, or warnings.

## Findings against required checks

1. **Serialize/decode byte integrity — PASS.** `serialize_activation` enforces exact width (`capture_export.py:104`), packs each `<f` (`:108`), and re-checks length `== 4096` (`:109`). `decode_activation` checks exact `bytes` type (`:126`), length (`:127`), digest format (`:128`), digest match (`:129`), then unpacks `iter_unpack("<f")` and rejects non-finite (`:130-132`). `payload_sha256` is computed from the same payload (`:158`) and compared in `decode_view` (`:184`). Probe: `decode_bytearray → PAYLOAD_TYPE`, `decode_inf → NONFINITE`.

2. **Finite/width/bool/overflow — PASS.** `_is_real_number` uses `type(value) is int or type(value) is float` (`:80-81`), so `bool`/subclasses are rejected; probe `bool → VALUES_TYPE`. Non-finite → `NONFINITE` (`:95`); `float(int)` `OverflowError` → structured `OVERFLOW` (`:93-94`); magnitude above max finite binary32 (`:63`, `:96`) → `OVERFLOW`. Probes: `hugeint → OVERFLOW`, `overmax3e38 → OVERFLOW`, `nan/inf → NONFINITE`, `width1023 → WIDTH`, `strinput → VALUES_TYPE`.

3. **Explicit float32 rounding — PASS.** `_float32_exact` packs/unpacks once through `struct <f` (`:97-98`) and `serialize_activation` packs that already-rounded value (`:108`), so there is no double rounding. Probes: `round_0.1 == struct.pack('<f',0.1)*1024` True; `max_exact` True; subnormal `1.401298464324817e-45` preserved. Native float32 values round-trip exactly (`test_round_trip_is_exact_for_representable_values`).

4. **Matching prefix + opposite order — PASS.** `_validate_order` is strict (`type(order) is list and order in ORDERS`, `:136-137`; `ORDERS` `:57`). Views are bucketed by `tuple(order)` (`:290-293`), the pair must be exactly `{("A","B"),("B","A")}` (`:298`), and `prefix_sha256` must match across views (`:301-302`). Probes: duplicate same-order → `DUPLICATE_VIEW`; differing prefix → `PREFIX_MISMATCH`.

5. **Validation admitted status / explicit null fold before decode — PASS.** In `assemble_validation_pairs` the case loop calls `_validate_case` (`:276`) before any `decode_view` (`:287`). `_validate_case` requires `status == ADMITTED_{split}_TEXT_ONLY` (`:230`) and the key `'development_fold'` explicitly present (`:231`), then `None` for non-TRAIN (`:236-237`). Probes passing `view_records=[None]`: unadmitted → `CASE`, missing fold → `FOLD` (case-level error, never a decode `RECORD` error), matching `test_unadmitted_and_missing_fold_rejected_before_pair_decode`.

6. **No fitting or centering — PASS.** Pair rows are a raw `0.5*a + 0.5*b` double average (`capture_export.py:303`); `folds` are `[None]*n` (`:306`). `capture_export.py` imports only `hashlib`, `re`, `struct` (probe `ce_imports`; no numpy/torch/transformers). `grep` for `np.|numpy|fit|center|normaliz|mean` matched only docstrings; no scaling, L2, or fitting occurs in either joined path.

7. **Equal activations allowed — PASS.** No value-difference check exists; only order records distinguish views. Probe: identical AB/BA retained exactly (`equal_retain` True; test `test_identical_preoption_activations_are_retained` passes). `pair_join` likewise accepts equal values (`pj_equal` True); `test_pair_join` has no equality rejection.

8. **Metadata separated from model features — PASS.** `decode_view` returns exactly the 8 structural fields (`:185-194`; probe `view_keys`). `assemble_validation_pairs` returns features under `x` with labels/groups/folds/case_ids as separate sidecars (`:306`; probe `out_keys`). `native_capture_contract.inference_input` emits only `schema/views/feature_contract` (`:302-306`); probe `native_keys = ['feature_contract','schema','views']`, `native_leak = []`.

9. **Native contract (hash-reviewed) — PASS.** `validate_token_ids` rejects bool/out-of-range/oversize (`:112-115`); binding enforces true LCP, `LAST_SHARED_ID`, and per-order label boundaries (`:212-219`); `validate_binding` recomputes and requires equality (`:252-253`); `_validate_binding_shape` re-checks hashes/indices (`:257-282`); `inference_input` strips supervision (`:285-306`). All 12 native tests pass, including `test_only_standard_library_imports`.

## Non-blocking note

`decode_view` uses `==` for `block` (`:179`) and `width` (`:182`), so float-equal values are accepted (probe: `lax width ACCEPTED 1024.0`, `lax block ACCEPTED 10.0`). Actual payload length (`:127`) and decoded width (`:130`) are still strictly enforced, so byte integrity is unaffected; this is schema strictness only, not a failure. `order`, `model_revision`, and `dtype` are strict.

## Provenance caveat

These results prove serialized-byte integrity and structural role validation only. They do not authenticate a native model, tokenizer, hook, capture pipeline, or real release; **passing tests are not native model provenance.**
