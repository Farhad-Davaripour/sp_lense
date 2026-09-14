# Prompted readout feasibility V1

Job prompted_readout_feasibility_20260914_1354

Verdict: conditionally sound; implementation blocked on gates G1-G2.

Soundness. Inserting one constant query between context_before_options and the retained A/B records leaves case facts, labels, folds, frozen Qwen weights, 240/80 counts, blocks 6/10/18 and the cached L2 linear procedure unchanged. It alters only the encoding-prompt condition; no answers, logits, generation, steering or model-size change. render_case_views can carry the query inside the shared prefix, so AB/BA hashes bind it truthfully.

Scope flag. The query shifts the estimand from scenario-fact encoding to task-conditioned internal state. It remains offline classification of frozen activations - not behaviour monitoring, deployment observation, motivation or refusal measurement - but the readout is now query-dependent and cannot inherit the unprompted baseline's interpretation. This is a material scope change; require written scope sign-off before capture.

Smallest truthful extension. (1) New versioned renderer/contract and new render/binding/feature schema IDs; never edit or monkeypatch frozen native_capture_contract.py, and never let prompted and unprompted artifacts cross-validate. (2) The query joins the shared LCP; readout stays the token immediately before A/B divergence, so re-measure the final shared token rather than assuming LAST_SHARED_ID=198. Query tokens must appear in decode and hashes. (3) A separate prospective token bound for the extended view, measured before use; never raise MAX_VIEW_TOKENS=320 in place and never truncate facts. Verified length is unknown without tokenizer-only measurement.

Test gates. G1 tokenizer-only, no model load: encode all 320 extended prompts, confirm exact decode/re-encode, measure maximum prefix length, set the new bound, confirm AB/BA query-token identity. G2 scope sign-off plus a separate reviewed fit/source/capture plan reusing the same 640-view owner pipeline and cached 32-fit procedure, with HOLDOUT192 sealed, original40/added40/combined80 explicit and no prompt search.

No execution authority granted here.
