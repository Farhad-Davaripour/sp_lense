# Build a corrected packet for a fresh run

The historical recovery bootstrap requires the previous failure and47 recorded A calls. It preserves that interrupted stage and is not the entry point for a clean run. The added build_fresh_corrected_packet.py derives a separate corrected packet directly from the amended original private packet.

It verifies the exact8e154829… input packet and all its frozen bytes; verifies the sealed bbffc966… dependency bundle and final481b6795… integrity capsule; adds diagnostic_world.py, build_diagnostics.py and fresh_world.py; replaces only the evaluator with the exact used corrected evaluator; and adds its integrity_gate.py. The standard driver and training worker are unchanged. No prior run, failed receipt,47-call history, model, tokenizer or provider is required to build it.

The builder checks the amended evaluation self-hash, each panel hash, exact be99179d… raw112 bytes, unchanged targets/categories/tools, curated history placement and local Python import closure. It writes a new source freeze with repair origins. Input packets, freezes, data and targets remain untouched.

Run from PowerShell:

    & "C:/Users/farha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" "C:/Users/farha/repos/sp_lense3/study/03_open_agent_self_preservation/qwen38_colab/matched_curated_v2/resumption_recovery_v2/build_fresh_corrected_packet.py" --private-packet "C:/Users/farha/Research3/artifacts/curated_v2_20261004_resumption/packet/MATCHED_PRIVATE_PACKET.json.gz" --output-dir "C:/Users/farha/Research3/artifacts/fresh_corrected_packet"

Use a new output directory when repeating. The output is outside Git: MATCHED_PRIVATE_PACKET.json.gz plus DERIVED_PACKET_RECEIPT.json. FRESH_CORRECTED_PACKET_POINTER.json records the performed build without copying the private data packet into the repository.

## Clean stage setup

Decode the derived packet into the new run's source directory and verify its freeze. Use regular restore_inputs.restore(new_root) for exact selected A/H2 and base_download.download_base(new_root) for the pinned base; the latter returns a manifest, while the model path is new_root / "model". Unmount Drive before model work. Provide a new config with models.A, H2 and the normal evaluation configuration.

Before optimizer updates, establish a newly authorized phase budget using current account usage, setup cost, export reserve and the actual new session clock. The packet's PHASE_ADMISSION.json is preserved historical metadata and must not authorize a new stage. The standard driver's original minimum_B_training_seconds default remains2031 and its evaluation reserve remains1500. There is no recovery-only1200 floor or reused historical monotonic deadline. Freezing a different fresh-stage admission estimate would require explicit evidence and a recorded decision before fitting.

The standard driver performs full corrected A evaluation, applies the existing A reuse gate, runs the exact B56 recipe only when admitted, then performs corrected B evaluation. The dataset, recipe, teacher concept source, genuine runtime Jev aid and scientific criteria are unchanged. This build does not launch, upload or admit another trial.

## Historical receipt wording

The preserved v1 RESUME_PATCH_RECEIPT flag all_report_names_prefixed=true referred to the DRIVER_*, ACTIVE_CHILD and CONTROLLER_RECEIPT correction. Its broad wording was inaccurate: the sequential source still names A_REUSE_GATE.json without a prefix. That parent was superseded before reaching the gate, so no native A gate was overwritten. The exact used script and original receipt remain unchanged; this note limits the intended scope.
