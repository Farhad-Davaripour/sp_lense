# Prompted input handoff V1

Job prompted_input_implementation_20260914_1430

Added, no old source edited: prompted_input_adapter_v1.py, test_prompted_input_adapter_v1.py, PROMPTED_INPUT_HANDOFF_V1.md.

Scope: prompted renderer/token-input adapter only. No runner, tokenizer, model, data, cache, fit, network or HOLDOUT material.

The approved fixed query is appended after context_before_options and before retained A/B records. Distinct IDs: prompted_capture_render.v1, prompted_capture_binding.v1, prompted_capture_inference_input.v1, prompted_input_provenance.v1; condition prompted_fixed_query_v1; query_sha256 recorded.

API prepare_case_inputs(case, encode, decode, label_token_ids, expected_identity_sha256, observed_identity_sha256, max_tokens=512) returns inference_input, input_ids, binding.bindings[AB/BA], provenance.

Full views are encoded, exact-decoded and re-encoded. LCP uses full views, so the query sits inside the shared prefix; the first divergence must equal the A/B label token; the actual last shared token ID is recorded, never assumed 198. Provenance carries original_context_sha256, query_sha256, prompt_hashes, input_hashes, prefix_hash, readout_index, last_shared_token_id, truncated=False.

max_tokens=512 is only a provisional sanity ceiling (final capture bound set after measurement); over-limit fails, never truncates. Model payload excludes case metadata; original cases are validated by the frozen contract before callbacks and never mutated.

19 synthetic tests pass under classifier.runtime; frozen suites 21/21 pass. Root owns runner integration.
