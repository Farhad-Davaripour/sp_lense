# Native receiver vertical slice

Ordinary editable development, not a frozen experiment or released backend. The receiver accepts an already constructed PyTorch model with native HF-shaped model.layers and logits/past_key_values outputs. It never imports a provider or loads checkpoints.

## Implemented

- Exact block10 last-input residual patch, CPU float32,1024-wide. The patch AST is identical to the reviewed adapter: add the offset at the last input position only; on gradient calls, return a detached full residual leaf requiring gradients so downstream computation actually consumes it.
- Standard PyTorch forward-hook registration on model.layers[10], removed in finally. Duplicate callback invocation is rejected. Unknown foreign hooks are preserved and cause cleanup failure, not silently removed.
- Native call uses exact IDs/mask, past_key_values=None,use_cache=False,logits_to_keep=0,return_dict=True. It consumes full-vocabulary logits and rejects an actual returned cache.
- The derivative remains grad(logits[50057]-logits[48964], current full leaf)[0,-1]. Added identity check rejects a stale logits object before derivative dispatch.
- Fresh native-order parameter hashes, object/name/shape/dtype/byte-boundary and duplicate-path checks; flags/versions/no-grad restoration; native module/global hook registries and buffer metadata; sticky failure and guard restoration.

core.py reuses DispatchStopped, DispatchLatch, ForwardDerivativeGuard and parameter_digest with exact AST parity to development/gdn_public_smoke_v3/candidate_real_adapter.py. It authenticates/reuses the existing pure finite trace module. No TL hook/registry/helper alias is imported. The finite trace in these tests is explicitly unclosed/unpublished synthetic observation (zero source marker), not authenticated production evidence.

## Tests

Run from the repository root:

```powershell
& .venv/Scripts/python.exe -B development/native_receiver_v1/test_native.py
```

Final result: eight PASS in2.078s measured test time (2.876s command wall time), after one ordinary cache-observation clarification. Initial eight-case run also passed. Cases cover exact zero/OFF identity, last-position-only edits/current gradients (1.5 then3.5), stale graph rejection, duplicate selected hook, foreign-hook cleanup failure, returned cache rejection, version-silent parameter-byte mutation and an uncounted forward. Nine synthetic forward dispatches and two derivatives were executed in the final test; no real model loaded.

The fixture is24 diagonal tiny blocks,1024 channels and248320 output logits, not a Qwen constructor or approximation used as scientific evidence. Provider/TL/PyArrow imports and network/checkpoint access are denied. No policy block occurred. Parameter flags and guard functions are restored in every case; negative cases remain sticky and cannot report a clean finalization.

## Limitations / next boundary

No HF/native Qwen definitions, weights, tokenizer, study inputs, model answers or learned gate were exercised. No provider loader, root authority, production controller, public smoke or final-study integration was added. The generic development latch is not the production PENDING/ADMITTED recorder protocol. The metadata registry here is a development check, not a certified replacement for the historical full hook guard/capacity evidence. Native weight order has not been mapped to the frozen expected digest; no observed hash replaces that digest. HF/TL numerical correspondence, gate compatibility and actual native importability under Windows policy remain unverified.

The next checkpoint is source review of this adapter and a separately approved native-provider entrypoint/import check; do not run or modify blocked PyArrow, weaken security policy, or reopen prior attempts. Existing codec/owned-controller proofs remain available for later integration and were not rerun.
