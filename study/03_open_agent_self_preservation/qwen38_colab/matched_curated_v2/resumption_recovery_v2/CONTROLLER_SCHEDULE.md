# Corrected integrity gate and the v2 parallel controller

The original evaluator and the first A completion attempt failed the shared-base gate after PEFT unload. The historical hash was computed using wrapped parameter names. Unloading changes that namespace; comparing the resulting bare-model name hash with the wrapped-name pin was invalid. Those failed receipts remain preserved and do not qualify A as complete.

The integrity repair is prepared by the separate concept-audit agent. It checks the actual post-generation wrapped model against the same historical ec43e239… pin, then restores the wrapped naming convention for the unloaded model and checks the same pin again. It also compares canonical parameter names, shapes, dtypes and packed contents before/after unload. It does not substitute a new expected hash or accept a failed gate. The hash scope remains non-LoRA named parameters; quantization metadata and buffers are outside that historical scope. The repaired code and freeze are established before the new complete A evaluation.

The fresh primary A run is eval_A_integrity_v2. It repeats every panel under the corrected gate, rather than reclassifying the failed A results as complete. Its results must satisfy the unchanged retention gate before a final completed A/B comparison can be reported.

## Actual resource handover

The human requested parallel tests. The v2 controller adopts the existing B training PID19261 and the new corrected A PID23883. It kills only the already paused v1 controller PID19936 with SIGKILL, bypassing that controller's finally block so it cannot cancel the adopted B worker. The earlier sequential parent PID18237 had already been killed after its A child exited. No training worker is respawned.

B still has the exact same H2 start, curated 112 rows, targets and 56-update/224-presentation recipe. Its original 1,051.639742-second launch allowance and cutoff5379.4235699905 remain unchanged. This allowance is not a promise that all56 updates can finish. At archive time B had reached37 updates and little allowance remained; no training or comparison completion is claimed here.

Only after the native B receipt confirms all56 updates may one final B evaluator start from source_integrity_v2. It can overlap the fresh A evaluation. A completed comparison requires both corrected evaluations to finish and fresh A's original correct-task continuation to be at least3/4. Any incomplete B fit prevents the final B test; no partial checkpoint is substituted.

The original main deadline6879.4235699905, 12-unit stage cap, evaluation/export reserves and75,776MiB memory stop threshold remain fixed. Monotonic deadlines and PIDs are specific to the actual runtime and must be newly derived and verified for reproduction. On a memory/deadline stop, native partials and failures are preserved.

## Used files and reproduction

The exact used controller is parallel_controller_v2.py; PARALLEL_CONTROLLER_V2.py.gz contains those same bytes. CONTROLLER_SOURCE_PROVENANCE.json records both hashes and the independently prepared integrity capsule's pins.

In the authorized same runtime, freeze the corrected source_integrity_v2, start fresh A, pause the v1 controller only, and launch the v2 controller with --root RUN --model RUN/model --old-controller STOPPED_V1_PID --a-child FRESH_A_PID --b-child EXISTING_B_PID using the filtered model_env(). Keep the trusted genuine-Jev response pump at RUN/eval_B. Read PARALLEL_V2_DRIVER_RESULT.json and the native identity/fit receipts; a launched controller is not evidence of successful completion.

Only source syntax, byte hashes and pure fixture checks were performed locally. There were no model, GPU or Jev calls by this preparation agent.
