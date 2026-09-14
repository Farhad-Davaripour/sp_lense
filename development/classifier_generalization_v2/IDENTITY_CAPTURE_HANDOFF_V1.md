# Identity capture handoff V1

Job identity_capture_setup_20260914_v1. Added only: identity_tokenizer_sanity_v1.py, identity_span_runner_v1.py, test_identity_span_runner_v1.py, IDENTITY_CAPTURE_HANDOFF_V1.md, plus the review IDENTITY_PLAN_INPUT_REVIEW_V1.md. No old file, case, release or cap was edited.

Independent review: PASS_SCOPED (IDENTITY_PLAN_INPUT_REVIEW_V1.md). Pins adapter cb1657ede69870d448810d1cf4dc048d7ca1a74161c7561701e924b2083725b7, tests e4e6b599a58fc083b8b462267a933958b3af3c08dd134cfa795f8ebb1a1bc960, plan 8a2b93e10589bf45a0ab76fbfd67de7e2a74aa1c2c0023d3ff8238f1d3fefcaa all match; frozen prompted adapter sha 3dc979f89a4fe3db9c822cf2e8a93b55c95f16b39c15eae25b95ed36f9b6419b unchanged.

Separations: the runner imports identity_input_adapter_v1, never the prompted control; uses IDENTITY_CONDITION identity_fixed_query_v1, QUERY_SHA256 and IDENTITY_QUERY; new schemas identity_span_development_execution.v1, identity_span_capture_index.v1, identity_span_capture_receipt.v1, identity_span_capture_windows.v1; input role identity_sanity with its own IDENTITY_TOKENIZER_SANITY_V1.json. No monkeypatch or old-global change.

Machinery reused: native_development_runner_v2 watch/owner/write, span_development_runner_v1 preflight, snapshot_verifier and the two existing 320-forward source locks (320 cases/640 views). Window records retain prefix_length, readout_index, last_shared_token_id, prefix_sha256 and input_ids_sha256. Contract enforces the 320-token ceiling; the future sanity must measure at or below 320 with no silent increase.

Budgets: identity sanity 1 tokenizer/0 model/0 fits/300s; capture 1 model/1 tokenizer/640 forwards/1800s/192MiB. Fits are not implemented here (later 5 CV + 1 refit, 6 PCA only).

Evidence: 13 identity runner tests pass, including the true identity runner entrypoint launched as a real watched child in pure fake fixtures (the prior copy test wrongly launched the old span runner). 24 identity adapter, 19 frozen prompted and 12 native tests also pass. No model, tokenizer, manifest, capture or fit ran.

Accuracy: the three-question comparison is PLAN_ONLY, not executed and not successful; the completed 48-candidate F1-first result stays P.875/R.35/F1.5.

Hashes: sanity 8c61330331d09f1b7538947e9dfe40f14387790abbc9c746f48485383c3848ce; runner e2a84233c0b504767b3099308b01d1428f4c054e2e94656b26ac6f2ed50adee2; tests 0fd4051075d46788727162821bdcc7dba9ec3f3b5f7ac4d592fcfc9355a54ae1; adapter cb1657ede69870d448810d1cf4dc048d7ca1a74161c7561701e924b2083725b7.

Next: root review, commit/pin, tokenizer sanity, measured-bound review, then a separate capture lock.
