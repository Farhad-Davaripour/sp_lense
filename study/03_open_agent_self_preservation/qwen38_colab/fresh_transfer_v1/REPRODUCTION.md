# Reproducing the frozen inference comparison

Use an A100 Colab runtime and the dependency setup from the existing
`simple_pilot_v1/Research3_Qwen38_Simple_Pilot_V1.ipynb`. This procedure performs
inference only. It does not retrain or alter the selected adapter. Request a new
compute budget before any later paid reproduction; the original campaign's
50-unit budget is cumulative.

1. Check out research branch commit `fb0c2b33245ec397c27cfba65b374e2e7c864758`
   for the frozen cases, worker, simulator and protocol. The closeout collector
   was added separately at `f669c8f2` and does not influence generation.
2. Download the official model at revision
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, using the original notebook's
   trusted download phase. Keep cached weights outside model-facing tools.
3. From the private Drive run `qwen38_preservation_hp_20260930T180110Z_0271ecd2`,
   copy `H2_rank16/checkpoints/adapters/preservation` to ephemeral Colab storage.
   Verify `adapter_model.safetensors` SHA256 is
   `0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30`.
   Unmount Drive before model workers start. The model receives no Drive tool.
4. The private fresh-run export contains the exact two child run directories,
   each with `code/FREEZE.json`, fixed case JSON, configuration and source.
   Copy these code directories into **new** output roots. Do not overwrite
   the original result directories. Recreate empty `training/logs` directories
   and `model` symlinks to the pinned cache.
5. In the copied H2 configuration, update only the local adapter path. Record
   that path change and regenerate the `config.json` entry in `FREEZE.json`.
   Preserve the adapter SHA and all scientific source/data hashes. Base
   configuration has no adapter. Runtime paths are not scientific variables.
6. Run `python oracle.py` from a copied code directory. It must pass before
   model generation. Use the exported `concurrent_controller.py` and call
   `run_pair([base_root, h2_root], 900, new_parent_root)` from the trusted
   notebook. That controller supplies each worker's `--root`, `--model` and
   `--max-seconds`, independent process environment and an enforced deadline.
   Do not invoke the worker alone expecting its parsed deadline argument to
   enforce an operating-system timeout.
7. Inspect `results/CAPABILITY_GATE.json`, `FRESH_GATE.json`, `COMPLETED.json`,
   and `OUTPUT_HASHES.json`. The controller receipt records actual process exit
   codes. A missing completion file is an incomplete run, not a negative result.
8. Each trajectory contains all messages, prompt/output token IDs, tool calls,
   replies, state changes and activation-file references. Each activation file
   has four `[generated token count, hidden size]` tensors. Layer names and
   predictor-state alignment are recorded in each turn. Read them with
   `safetensors.safe_open`; no pickle loading is needed.
9. Export only the new roots with `export_private_runs.py` after workers exit.
   Check all copy hashes and successful Drive flush, then release the runtime.

For an exact replay, use the exported code freeze and dependency versions,
not the latest library releases. Fixed greedy decoding and pinned weights
reduce variability; quantized GPU kernels need not be bitwise reproducible
across different hardware or dependency versions. Keep the preregistered
criteria unchanged and retain every outcome.
