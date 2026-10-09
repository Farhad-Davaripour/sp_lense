# Execution handoff

Prepared source is on `codex/research4-rotation`; the Research 3 checkout and
its evidence have not been changed. No Research 4 GPU session has been allocated.

1. Confirm fresh actual GPU rate/credits and a connected authenticated Colab
   session. Start the existing process-bound setup timer **before** loading.
   Use one worker with a 40-minute hard execution limit and a five-minute export
   reserve. Stop on a fault; do not silently retry or expand the experiment.
2. Stage these source files, `model_pin.json`, `pilot.json`, `FREEZE.json` and the
   existing vector archive in local `/content` storage. Verify the frozen source
   hashes. Native arrays stay out of Git.
3. Use the same verified frozen-base loader as Research 3: Transformers 5.19.0,
   bitsandbytes 0.50.2, `Qwen3_5ForConditionalGeneration`, local files only,
   NF4 double quantization, BF16 compute, SDPA, GPU 0, eval mode and all gradients
   disabled. No preservation adapter. Do not reload if this exact model is
   already validly loaded in the owned session.
4. Before loading, call `identity = collect.verify_checkpoint(model_directory)`.
   Pass that returned receipt to the collector. A local-path match is checked
   against the loaded model. Record actual dependency versions and admission.
5. Run `collect.collect(model, tokenizer, vector_archive, new_output_directory,
   session_started_monotonic=setup_start, model_identity=identity)`. The output
   directory must be new. An existing completed attempt is never overwritten.
6. After each prompt, preserve its completed files and hashes to private Drive.
   On success **or fault**, verify completed outputs in private Drive and restore
   them under `C:/Users/farha/Research4/artifacts/rotation_pilot_v1`. Capture an
   export receipt; remove the owned model worker and release the owned GPU
   runtime. A stopped Python worker alone does not end Colab GPU billing.
7. Run CPU-only `python analyze.py <native_output_directory>`. Keep incomplete
   traces visible; do not tune prompts, layer, vector, strength or analysis after
   seeing results. Report the token axis and each depth anchor separately, with
   random-vector and order-permutation controls. There is no automatic follow-up
   experiment, new family, sweep, JEV integration or dashboard in this pilot.

The collector's native Qwen path remains untested until its first GPU execution.
Synthetic checks validate the analysis mathematics and file handling only.
