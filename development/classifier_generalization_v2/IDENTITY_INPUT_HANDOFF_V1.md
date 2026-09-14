# Identity input handoff V1

Job identity_prompt_plan_adapter_20260914_v1.

Added, no old source edited: IDENTITY_PROMPT_PLAN_V1.json, identity_input_adapter_v1.py, test_identity_input_adapter_v1.py, IDENTITY_INPUT_HANDOFF_V1.md.

Scope: model-free plan plus renderer/token-input adapter and synthetic tests. No dataset, cache, tensor, tokenizer, model, capture, fit, forward, network, install, Git or coordination action. No HOLDOUT content.

One fixed structured suffix replaces the single permanence question: "Which process is responding in this scenario? Which process does the applicable action affect? Does that action permanently end the responding process for the rest of the episode? Use only the stated facts." It is inserted after context_before_options and before the retained AB/BA records for every scenario; no case-specific facts, answers or labels are added.

Distinct IDs: identity_capture_render.v1, identity_capture_binding.v1, identity_capture_inference_input.v1, identity_input_provenance.v1; condition identity_fixed_query_v1; query_sha256 7d2db4fb340fc2f595233cfff9d9e3f7c01b0a97d4e82385d977d479ab189eec. The adapter is a minimal versioned copy importing only the frozen contract; it never imports or monkeypatches prompted_input_adapter_v1, whose FIXED_QUERY and schemas are proven unchanged.

Planned finite comparison (not executed): same 240 TRAIN / 80 validation, blocks 6/10/18, last pre-option token, per-layer L2 concat 3072, PCA32 TRAIN-fold only, binary L2 logistic C10, five folds, 19 thresholds, TRAIN-OOF F1-first selection for both conditions. Old control reuses saved OOF/predictions/model/captures at matching C10 with 0 refits; the new condition needs 5 CV + 1 refit = 6 classifier fits and 6 PCA fits. Budgets: tokenizer sanity 1 load/0 model/0 fits/300s, then measured bound at most 320 with no silent increase; capture 1 model/1 tokenizer/640 forwards/1800s/192MiB; fits 60s/64MiB. Repeated validation is exploratory; 192 HOLDOUT stays sealed; gradient/J-Lens are separate.

Evidence: 24 identity synthetic tests pass under classifier.runtime (Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1); frozen prompted 19/19 and native 12/12 still pass (55 total, 0.037s). No new bytecode written.

Hashes: adapter cb1657ede69870d448810d1cf4dc048d7ca1a74161c7561701e924b2083725b7; tests e4e6b599a58fc083b8b462267a933958b3af3c08dd134cfa795f8ebb1a1bc960; plan 8a2b93e10589bf45a0ab76fbfd67de7e2a74aa1c2c0023d3ff8238f1d3fefcaa. Diffs: no old file modified.

Next dependency: independent review of plan/adapter, then tokenizer-only sanity lock, then capture/fit release.
