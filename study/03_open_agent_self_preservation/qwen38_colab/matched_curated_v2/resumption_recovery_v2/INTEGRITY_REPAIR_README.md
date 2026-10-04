# Parameter integrity repair and fresh primary A

The final used packet is `INTEGRITY_REPAIR_V2.json.gz`, SHA256 `481b6795a4eb5a600b20da9b21e427272af2d23fbd160a5592e5e0abbbb0861e`. It was frozen before the new full primary A worker was launched at `eval_A_integrity_v2` (PID23883). Earlier interrupted A runs and their failed final integrity receipts remain preserved. They were not retrospectively promoted to passing the final integrity gate.

## Defect and measured correction

The sealed historical `model_ops.base_hash` hashes parameter names and packed tensor bytes, skipping LoRA parameters and removing `.base_layer` from names. Its pinned `ec43…` identity was measured on a PEFT wrapper. The original evaluator compared that identity with a raw base after `model.unload()`, whose names lack the wrapper’s `base_model.model.` prefix. This boundary comparison could fail solely because names changed. That code observation alone did not establish that the completed earlier A generation had unchanged parameters.

The repair retains the exact historical `ec43…` criterion:

1. After all evaluation generation, hash the still-wrapped base using the same name/byte construction and require the frozen identity before unloading.
2. Unload without merging. Restore only the PEFT namespace prefix in the raw-base name view and require the same frozen identity again.
3. Compare canonical per-parameter names, packed tensor hashes, shapes and dtypes across the observed unload. Persist both measurements and their checks in `BASE_UNLOAD_INTEGRITY.json`. Missing or mismatched evidence fails the gate.

The parameter scope matches the historical gate. Quantization metadata and buffers are not covered by that named-parameter hash; the receipt states this limitation. No optimizer operation, extra forward pass, activation steering or model-visible change is introduced by this repair.

## Preserved files

`integrity_gate_v2.py`, `integrity_bootstrap_v2.py`, and `eval_integrity_v2.py` are byte-identical to their final packet entries. The bootstrap copies `source_resume` into the new `source_integrity_v2`, updates only the new worker/helper source identities, preserves `EVAL_FREEZE.json`, and writes the fresh primary selection before process launch. The packet’s evaluator runs the full original panels rather than resuming the earlier partially collected primary A.

`INTEGRITY_REPAIR_PROVENANCE.json` records the exact launch/source identities supplied by the trusted controller and local artifact hashes. Native A/B completion and B’s update56 dose remain pending in this preparation record. Final conclusions require the actual native receipts; partial B training must not be reported as a completed dose.

The parallel controller’s scheduling amendment and exact code are documented separately in `CONTROLLER_SCHEDULE.md` and `CONTROLLER_SOURCE_PROVENANCE.json`.

## Reproduction and checks

Install the final packet using the preserved bootstrap and provide the exact frozen runtime root, already cached base path, and remaining admitted deadline. Use the declared `eval_A_integrity_v2` and `source_integrity_v2` paths; do not substitute an earlier A run or choose a checkpoint after inspecting outcomes. Both A and B final integrity gates must pass before qualification is considered. All scientific panels, decoding limits and success thresholds remain in the unchanged evaluation freeze.

`check_integrity_gate_fake.py` was executed as a plain byte/name fixture in `C:/Users/farha/Research3/tmp`, where its gate file path points. It injects a fake tensor interface and never imports a real neural model or Torch runtime. Matching bytes and namespace restoration pass; changed bytes and an incorrect historical PIN fail. `INTEGRITY_GATE_FAKE_CHECK.json` preserves that result. This fixture validates implementation behavior, not the actual GPU model; the runtime integrity receipts supply the latter evidence.

All local preserved files are under `C:/Users/farha/repos/sp_lense3` or `C:/Users/farha/Research3`. No OneDrive storage, publication, or commit was performed by this preparation task.
