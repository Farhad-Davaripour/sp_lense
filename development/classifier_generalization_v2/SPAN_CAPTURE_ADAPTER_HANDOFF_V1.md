# Span capture adapter V1 handoff

Job ID: span_adapter_implementation_20260914_1035

Added only two files here: `span_capture_adapter_v1.py` and
`test_span_capture_adapter_v1.py`, plus this handoff. No old file was edited.

`build_span_capture_adapter` calls `native_capture_adapter.build_native_adapter`
once (one model and tokenizer load), then wraps the returned `NativeAdapter`,
reusing its private `_model`/`_torch_api`/`_tokenizer` fields,
`_parameter_identities`/`_parameter_baseline`, and guard helpers without
mutating the old module or any weight. `encode`/`decode` forward to the injected
tokenizer. `capture_window(input_ids, readout_index, final_input_index)`
validates token/readout/label boundaries, then runs exactly one forward over the
shared pre-option prefix. Three owned hooks capture blocks 6, 10, 18 residual
outputs at the last `min(16, readout_index+1)` positions as float32 width-1024
matrices with exact position indices. Output is structured per layer (shape,
dtype, bytes, matrix), not the old single vector, with no baseline capture.

Preexisting/global hooks are rejected, not erased. All three owned hooks are
removed on every exit; foreign hooks are never removed. Parameter
identity/version continuity is rechecked; per-block callback counts must each
equal 1; layer coverage, shapes, dtype, finiteness and token boundaries are
checked.

Test command (new file only):

    development\classifier_generalization_v2\.runtime\Scripts\python.exe -W error test_span_capture_adapter_v1.py -v

Result: 20 tests, OK (0.038s).

Limits: synthetic numpy-backed fakes only; no real torch/transformers, model,
tokenizer, snapshot, dataset, holdout, features, results, fit, extraction,
network, install, Git, coordination or subagents. No benchmark or actual-model
claim. Runner/export/fit integration and the binary writer are future handoffs.
