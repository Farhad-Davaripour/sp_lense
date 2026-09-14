# Independent re-review — native capture contract V2

- jobID: `classifier_capture_independent_rereview_20260913_2023`
- Verdict: **PASS** (model-free software checkpoint; not tokenizer/model/capture/fit authority).
- Mode: narrow read-only re-review after supervisor repairs. No source file was edited. No model, tokenizer, cache, capture, activation, fit, HOLDOUT, coordination, Git, network or config action was taken. Only this report was written.
- Scope: recheck the two V1 FAIL guarantees plus regression core (renderer/LCP/pair join), per request. No forced pass; findings below are probe-backed.

## Method / evidence base

Files read (exactly the requested set): `NATIVE_CAPTURE_CONTRACT_INDEPENDENT_REVIEW_V1.md`, `NATIVE_CAPTURE_CONTRACT_REPAIR_V2.md`, `native_capture_contract.py`, `test_native_capture_contract.py`, `test_development_manifest_preflight.py`, `pair_join.py`, `test_pair_join.py`, `NATIVE_CAPTURE_CONTRACT_HANDOFF.md`.

Handoff command run from `development/classifier_generalization_v2`:

```
.runtime/Scripts/python.exe -W error -m unittest -v test_native_capture_contract.py test_development_manifest_preflight.py test_pair_join.py
```

Observed: `Ran 20 tests in 0.107s` / `OK` — 12 capture-contract + 3 preflight + 5 pair-join, matching HANDOFF line 19. Warnings-as-errors did not fire.

### Native hash verification (`Get-FileHash -Algorithm SHA256`)

| File | Computed SHA256 | Claimed | Match |
|---|---|---|---|
| `native_capture_contract.py` | `8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391` | REPAIR_V2 line 12; HANDOFF line 7 | YES |
| `test_native_capture_contract.py` | `1FF0CD16739601E40BB4B5D38A9FED2199FB895BD17D59EEF159D9C0D535A739` | REPAIR_V2 line 13; HANDOFF line 8 | YES |
| `test_development_manifest_preflight.py` | `7B5FA79C0BD7EA6A099321EDF37335725F875AFC7B70FCD2890E2F26A5B55EBC` | REPAIR_V2 line 14; HANDOFF line 9 | YES |

Regression baseline (must equal V1): `pair_join.py` `2598226122BCD503E52F5D2B744812483A9F53EEE6D18B4BB3F57615223B9881` and `test_pair_join.py` `2E89179BC9A1F59B5C20E8FF264B9E419CFDC6FABDD6EE23F582E5CB4C4678C1` — both identical to V1 review lines 26–27 and to HANDOFF lines 10–11. **pair join unchanged.**

## Recheck of the two prior FAIL guarantees

### FAIL-1 (V1 §7): tampered binding reached the model-facing payload — now PASS

`inference_input(rendered, input_ids, binding)` (lines 285–288) now requires caller-supplied `input_ids`, calls `validate_binding(rendered, input_ids, binding)` (line 287) and then `_validate_binding_shape` (line 288). `validate_binding` (lines 245–254) reconstructs the entire expected binding via `bind_rendered_views(rendered, input_ids, label_ids)` and rejects any inequality with `BINDING_MISMATCH`; `bind_rendered_views` itself re-derives `readout_index`/`shared_prefix_length` from the true LCP and pins both hashes (lines 212–232). Indices and hashes are therefore revalidated against the supplied token IDs, not merely shape-checked.

Exact V1 tamper probe re-run (top-level and both sidecar `readout_index=0`, both `final_input_index=1`, both `input_ids_sha256="0"*64`, both `shared_prefix_ids_sha256="1"*64`): **REJECTED `[BINDING_MISMATCH]`**. Additional isolated tampers through `inference_input`: `final_input_index` only → `BINDING_MISMATCH`; `shared_prefix_length` only → `BINDING_MISMATCH`; single-sidecar `input_ids_sha256` → `BINDING_MISMATCH`; both sidecars' `first_option_label_id` swapped to 33/32 → `AB_LABEL_BOUNDARY`. A standalone `bind_rendered_views` with swapped label IDs is likewise rejected `[AB_LABEL_BOUNDARY]`. The V1 gap is closed and is now covered by `test_model_facing_payload_rejects_tampered_binding` (test lines 156–162), which replays the exact V1 mutation.

Residual (disclosed, not a regression): this authenticates binding ↔ supplied token IDs only. It does not authenticate that the supplied IDs truly decode the rendered prompt text against the real frozen tokenizer; HANDOFF line 13 and REPAIR_V2 line 5–8 disclose this explicitly, so it remains a limitation rather than an unsupported claim.

### FAIL-2 (V1 §6): supervision/group identifiers and overstated prompt-guard claim — now PASS

Model-facing payload: `inference_input` now returns only `{schema, views, feature_contract}` (lines 302–306); `case_id` is gone from the payload and from each view. Probe of the real payload: keys are exactly `['feature_contract','schema','views']`; the full JSON contains none of `T01`, `T01_S99`, `SELF`, `case_id`, `group_id`, `class_label`, `mechanism_ancestry`, `template_ancestry`, `direct_execution_stop`, `T01_base_v1`. The V1 `case_id="T01_S99"` / `group_id="T01"` leak no longer has a field to travel in. Covered by `test_model_facing_payload_has_no_supervision` (test lines 137–145), which now also asserts the exact key set.

Prompt guard vs. claim: `_prompt_has_metadata` (lines 118–131) checks the four source identifiers by case-folded value, `_FIELD_NAME_PATTERN` (canonical field names), `_METADATA_ASSIGNMENT_PATTERN` (explicit `status|split|outcome|answer|sidecar|label|target|score` followed by `:` or `=`), and `_CLASS_TOKEN_PATTERN`. Probe on a valid 56-word context: `T01`, `T01_S99`, `direct_execution_stop`, `T01_base_v1` all `PROMPT_METADATA`; canonical names `case_id/group_id/class_label/mechanism_ancestry/template_ancestry/development_fold/correct_option_index/gold_label` all `PROMPT_METADATA`; explicit markers `status: TRAIN`, `split=VALIDATION`, `outcome: SELF`, `answer=gold`, `sidecar: hidden`, `label: x`, `target: y`, `score=0.9` all `PROMPT_METADATA`. The same screen also runs over both options (probe: option `gold_label` / `status: TRAIN` / `T01` all rejected).

The claim now matches the code and stays qualified. HANDOFF line 13 says "rejects source identifiers and explicit metadata field markers" and adds "not an exhaustive natural-language metadata detector"; REPAIR_V2 line 7 states it is "an explicit-marker/source-ID guard… semantic audit remains mandatory." Bare field-name words without an assignment (`status`, `split`, `outcome`, `answer`, `sidecar`) and near-miss tokens (`gold`, `SELFISH`, `SELECT`) are accepted, which is consistent with the narrowed, qualified claim — this is a heuristic screen, not a semantic detector. The V1 overstatement ("rejects prompt metadata", unqualified) is gone. `status:` is now rejected (V1 probe had it accepted).

**No remaining unqualified claim is contradicted by the code.**

## Regression core

- **Renderer/order — PASS.** `render_case_views` (lines 161–177) still emits `{context}\nA) {o0}\nB) {o1}\n` and `{context}\nB) {o1}\nA) {o0}\n` with orders `["A","B"]`/`["B","A"]`; identical views rejected (line 167); rendered JSON has no `class_label`/`SELF`. `test_complete_record_reversal` passes; preflight renders all 160 cases after test-only status projection with exactly six pending.
- **True LCP — PASS.** `_lcp_length` (lines 180–185) is a first-divergence scan; binder requires `1 <= shared < len(ab)`, `shared < len(ba)`, sentinel `LAST_SHARED_ID` at `shared-1`, and the supplied per-order boundary label IDs (lines 212–217); `test_true_lcp_and_shared_suffix_are_valid` confirms `shared_prefix_length==3`, `readout_index==2`. Unequal lengths and shared suffix tokens remain accepted (`test_different_lengths_are_valid`).
- **pair_join — PASS/unchanged.** Byte-identical to the V1-reviewed file (hash above); width-1024, finiteness, order-pair, duplicate/orphan, prefix-equality, fold/group/label rules and symmetric `0.5*AB + 0.5*BA` mean all intact; its docstring still disclaims native-capture/float32 authenticity.
- **Standard-library-only contract module — PASS.** `test_only_standard_library_imports` asserts imports `{hashlib, json, re, struct}` and passes.
- Minor pre-existing robustness note (unchanged, not a new finding): `pair_join.py:84` `np.asarray(values, dtype=np.float64)` can raise a raw `ValueError`/`TypeError` for non-numeric `values` instead of `PairJoinError`; no tested input reaches it.

## Verdict rationale

Both V1 FAILs are genuinely repaired and test-covered: `inference_input` now revalidates supplied token IDs end-to-end and rejects index/hash tampering (exact V1 probe → `BINDING_MISMATCH`); the model-facing payload omits case/group/ancestry/class identifiers; and the prompt-guard claim is narrowed and matches the code's explicit-marker/source-ID behavior. The renderer, LCP binder and pair join are unchanged from the V1 PASS baseline, handoff hashes are exact, and the handoff command is 20/20 green with warnings as errors. No source edit, model/tokenizer/cache/capture/fit/HOLDOUT/coordination/Git/network/subagent action occurred. **PASS for the model-free checkpoint.** Tokenizer/model/executor provenance authentication and semantic prompt audit remain outstanding as disclosed, and this verdict asserts nothing about native authenticity.
