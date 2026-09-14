# Independent review: supervisor tokenizer-input adapter, V1

- Job ID: `tokenizer_input_adapter_review_20260914_0417`
- Mode: independent, read-only. No source edits, tokenizer/model loads, datasets, caches, vectors, HOLDOUT reads, fits, network, installs, Git, coordination, config, or subagents.
- Files read: `tokenizer_input_adapter.py`, `test_tokenizer_input_adapter.py`, `TOKENIZER_INPUT_ADAPTER_HANDOFF_V1.md`, `native_capture_contract.py`, `test_native_capture_contract.py`, `CAPTURE_EXECUTOR_PLAN_DISPOSITION_V1.md`.

## Native SHA256

| File | SHA256 |
|---|---|
| tokenizer_input_adapter.py | `3883326066F2D3B322A79547BC9054588750D9D020A47543930B216183F4D164` |
| test_tokenizer_input_adapter.py | `3E62104F55606CDD191A7E4A55639F056AB46915C9E274CD8D68B9B83276033E` |
| native_capture_contract.py | `8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391` |
| test_native_capture_contract.py | `1FF0CD16739601E40BB4B5D38A9FED2199FB895BD17D59EEF159D9C0D535A739` |
| TOKENIZER_INPUT_ADAPTER_HANDOFF_V1.md | `3BF046AC5A8C67307BCAA55ED57D2FEDFCCEFE119D7C2D27D9B00D678773326E` |

The adapter and its test hashes match `TOKENIZER_INPUT_ADAPTER_HANDOFF_V1.md` exactly.

## Test run

From repository root, study directory on `PYTHONPATH`, study `.runtime` Python:

`python -B -W error -m unittest -v test_tokenizer_input_adapter test_native_capture_contract`

Result: **Ran 21 tests (9 adapter + 12 native) — OK, 0.011s, exit 0** under warnings-as-errors. **PASS.**

## Independent failure probes (14)

F1 exact callback flags — encode always `add_special_tokens=False`; decode always `skip_special_tokens=False`, `clean_up_tokenization_spaces=False`. PASS
F2 valid-format identity mismatch → `IDENTITY_MISMATCH`, zero callback calls. PASS
F3 HOLDOUT full case → `DEVELOPMENT_ONLY`; `DRAFT_UNREVIEWED` → `STATUS`; non-dict → `DEVELOPMENT_ONLY`; incomplete `{'split':'TRAIN'}` → `CASE_KEYS`; all before callbacks. PASS
F4 admitted VALIDATION accepted; views ordered `AB`/`BA`. PASS
F5 model-facing payload contains none of `T01`, `T01_S99`, `SELF`, `case_id`, `group_id`, `class_label`, `mechanism_ancestry`, `template_ancestry`, `development_fold`, injected `LEAKME`; exact keys `{schema, views, feature_contract}`; per-view keys have no label sidecar. PASS
F6 decode returns the exact rendered prompt; re-encode equals original ids; contract recomputes same `readout_index`, shared-prefix hash, sentinel `198`, label-boundary `32`/`33`. PASS
F7 case dict unchanged and callback-returned lists not mutated in place (decode clears its argument). PASS
F8 tampered binding (`input_ids_sha256`) rejected by reused `inference_input`. PASS
F9 token bounds via adapter: 320 accepted (bound not `TOKEN_IDS`); 321, empty, and `== MAX_VOCAB_ID` rejected `TOKEN_IDS`. PASS
F10 boolean token id rejected `TOKEN_IDS`. PASS
F11 wrong decode → `DECODE_MISMATCH`; unstable re-encode → `REENCODE_MISMATCH`. PASS
F12 encoder/decoder exceptions → `ENCODER_ERROR`/`DECODER_ERROR`; internal detail not leaked in message. PASS
F13 `None`, `abc`, uppercase 64-hex, 63-hex → `IDENTITY_HASH`; non-callable encode → `CALLBACKS`. PASS
F14 wrong/duplicate label map → `AB_LABEL_BOUNDARY` / `LABEL_TOKEN_IDS`. PASS

**Probes: 14/14 PASS.**

## Verification of requested properties

1. **Development-only role admission before callbacks — PASS.** `prepare_case_inputs` requires a dict with `split in {TRAIN, VALIDATION}`; `render_case_views` then enforces the split-status pairing and fold rules. `HOLDOUT` — although allowed by the native contract — is excluded at the adapter, so no HOLDOUT case reaches a callback.
2. **Identity check before any callback — PASS.** Both digests are format-checked and equality-checked before `deepcopy`/render/encode/decode. Proven by F2/F13 (zero callback calls on rejection).
3. **Exact raw prompt and special-token flags — PASS.** Prompts come verbatim from `contract.render_case_views`; `add_special_tokens=False` on every encode; both decode flags are `False` (F1).
4. **Decode + re-encode equality — PASS.** Exact string decode then identity re-encode is required, with structured failures (F6/F11).
5. **Bounds, boundary, and hash verification via the reused contract — PASS.** Token range/count, LCP sentinel, label boundary, prefix/input/prompt hashes, and tamper rejection all flow through `native_capture_contract` (F6, F8–F10), not re-implemented.
6. **No input mutation — PASS.** The case is deep-copied; callback-returned lists are copied before use and never mutated (F7).
7. **No label/group/case identity in the model-facing payload — PASS.** `inference_input` carries only schema, view order, raw prompt text, readout/final indices, and token hashes; binding/provenance receipts are separate (F5).

## Identity-hash status (explicit)

The two 64-hex **identity hashes are caller assertions only**. They are format-checked and required to match each other, but this adapter does not read tokenizer snapshot bytes, so equality proves nothing about real tokenizer provenance. `provenance.snapshot_bytes_verified_by_this_adapter` and `native_model_provenance_verified` are hard-coded `False`; `identity_pin_matched: True` means only "the two supplied strings agreed". **These are not real snapshot authentication.**

## Non-blocking observations

- The role gate passes a malformed dict with `split='TRAIN'` through to the contract, which rejects it as `CASE_KEYS`; still pre-callback, not a defect.
- `encoder_identity_sha256` records the expected pin only; observed is not retained separately — cosmetic.
- All fixture tokenizer behavior is synthetic/in-memory; no real tokenizer was loaded, consistent with the disposition and handoff scope.

## Disposition

**PASS** for the development-only, model-free adapter. Promotion to real capture remains conditional on the future loader independently verifying frozen tokenizer snapshot bytes; the identity pins here must not be treated as that verification.
