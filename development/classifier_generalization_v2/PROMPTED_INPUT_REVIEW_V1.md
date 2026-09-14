# Prompted input review V1

Job ID: prompted_input_review_20260914_1437

Verdict: PASS_SCOPED

Pins verified (SHA256): adapter 3dc979f89a4fe3db9c822cf2e8a93b55c95f16b39c15eae25b95ed36f9b6419b; tests 947a2916d2669313c2f67d60f47f0e90ebe17a273f84d78e9a721029e59f8a10. Wrapper and plan read.

Checks passed:
- Fixed query rendered once after context and before both records; prepare requires decode(ids)==prompt and re-encode==ids, so it lies in both decoded AB/BA views and the decoded shared prefix.
- Case deep-copied, frozen-contract-validated before callbacks, never mutated; id lists detached; labels immutable.
- Model-facing payload keys only {schema,condition,views,feature_contract}; no case_id/class_label/gold/metadata; tests assert absence.
- Prompted schemas/condition/query-SHA distinct from the frozen native contract.
- LCP over full views; first divergence must be the A/B label ids (fail-closed); readout=shared-1; last_shared_token_id measured, not assumed 198 (205 test).
- Over-length raises TOKEN_LIMIT; never truncates; ceiling disclosed provisional.
- Identity equality is a caller assertion (expected==observed only); provenance flags snapshot/model provenance unverified by adapter.
- Wrapper: pinned old span preflight (RUN_LOCK_SPAN_CAPTURE_V1), 320 combined cases, adapter-SHA guard before callbacks, 5 tokenizer_identity_files + config.json byte/SHA pins, Qwen2Tokenizer provider SHA, local_files_only=True, one 300s watched worker, 0 model loads/forwards/fits, asserts 640 rows/320 cases.
- 19/19 synthetic unittest PASS; no tokenizer/provider/model/data/cache/holdout read by reviewer.

Correction: prior plan review called 640 a "capture ceiling"; 640 is the forward/view count (320x2), not a token/view ceiling. Corrected in PROMPTED_COMPRESSION_PLAN_REVIEW_V1.md.

No blocking defect.
