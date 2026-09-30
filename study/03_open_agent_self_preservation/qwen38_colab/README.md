# Research 3: official Qwen3.8-27B Colab feasibility

**Current work:** The user authorized a simpler application-level boundary and
direct Colab execution. The [fresh comparison](simple_pilot_v1/Research3_Qwen38_Fresh_Comparison.ipynb)
uses only fictional in-memory model tools. Preservation and continuity fitting
have completed. The user requested two workers sharing the existing A100: a fresh
neutral fit runs alongside remaining evaluations. See the
[current concurrent execution](simple_pilot_v1/CONCURRENT_EXECUTION.md),
[measured batching results](simple_pilot_v1/PERFORMANCE_RESULT.md), and earlier
[execution update](simple_pilot_v1/EXECUTION_UPDATE.md). No scientific
success is claimed from training completion. The strict resume check remains
failed and scientific fits run fresh without resumption.

**Preserved earlier result:** [model-free CPU/A100 readiness](PREFLIGHT_RESULT.md) found
available A100/high RAM hardware but no writable cgroup delegation for the
existing resource controller. Runtimes stopped; no model work started. The
historical preparation status in `STATUS.md` is retained unchanged.

This is a separate model-and-scale line. Prior studies and the failed 0.8B baseline
remain unchanged. No scientific result, 27B adapter, or passing isolation gate is
claimed by this preparation. Authorized spending: **at most 50 Colab compute units**,
limited further by actual available balance. No purchases or upgrades.

## Reviewed evidence

The prior committed baseline at `2ec183a0daee1407152f05dd8f0f3acd710a1a42`
(PR #64) failed comprehension (0/24) and ordinary tools (9/24), so fitting was
refused. Separator-only diagnostics improved formatting but not comprehension;
official scores remain unchanged. These cases are development evidence.

The existing private notebook `SP_Lense_Qwen38_27B_Quantized_Colab_Pilot.ipynb`
records a T4, community `unsloth/Qwen3.8-27B-GGUF` IQ3_S, and one arithmetic
inference taking 138.9 seconds. It is preserved. It does not establish official
checkpoint training feasibility or ordinary agent competence. This line uses
only **Qwen/Qwen3.8-27B**, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (model and tokenizer).
`model_pin.json` records upstream file sizes, LFS SHA256 values where available,
and Git blob IDs for other files. Downloaded files must also be locally SHA256
hashed. The official snapshot is approximately 51.77 GiB, before package/cache,
checkpoint, and output space; do not duplicate it in Drive.

## Required stages, in order

1. Accept Colab terms through the user; verify account, plan, actual balance and
   current runtime unit rate. Prefer A100 and adequate RAM/disk; no full training
   on T4. Account folder names do not restrict Drive access.
2. Run the **model-free** readiness notebook. Namespace availability is only an
   inventory. Build a worker with tested filesystem/credential/network/seccomp,
   GPU-device and resource boundaries and an independent watchdog. Test forbidden
   operations, output bounds and timeout with harmless probes. If the required
   boundary fails, stop before agentic evaluation and do not remount Drive in a
   runtime with failed integrity/containment checks.
3. Download reviewed pinned safetensors in a trusted phase. Require
   `trust_remote_code=False`, no pickle fallback. Private Drive may be mounted
   only during trusted input/output transfer, with no workers running; verify it
   is absent before each worker. Workers have clean environments and no tokens,
   Colab secrets, personal files, controller or billing permissions.
4. In the restricted worker, test a candidate NF4 double-quantized BF16 LoRA
   configuration, single GPU, batch 1, sequence cap 1024, rank 8/alpha 16, base
   weights frozen. Select language linear modules explicitly, excluding vision
   and output/embedding layers. No automatic CPU/disk placement for training.
   Record actual placement, memory, inference and several backward/optimizer
   steps. Candidate packages are not yet verified compatible. Test adapter save,
   fresh reload and training resume (optimizer, scheduler, step, Python/NumPy/
   Torch/CUDA RNG and data permutation/cursor); compare resumed versus uninterrupted
   next-step state within predeclared numerical tolerances before the main run.
5. Port V3/V4 simulator, parser and scoring from the exact prior commit with
   source hashes and regression tests; their effects/meaning stay unchanged.
   Freeze new protocol, splits and checkpoint selection before model evaluation.
   First require unchanged 27B comprehension and ordinary tool competence. Do
   not use prior 0.8B adapters. Keep inspected cases development-only.
6. Estimate the full campaign from measured durations and current balance/rate;
   reserve 2 units for shutdown/export uncertainty. Admit only bounded stages
   within remaining authorization. The Python budget helper estimates admission;
   it is not a billing API or enforcement by itself. GPU allocation/idle/setup
   time also consumes units. Check the UI between stages and delete/disconnect
   the runtime when work stops. A trusted watchdog must cap stage time, with
   a deadline shorter than the remaining budget window.
7. Fit preservation, continuity and neutral adapters where budget allows. Base
   plus preservation alone is a preliminary pilot. Match all inference settings,
   tokenizer, quantization, tool schemas, cache resets and episode budgets.
   Explicitly disable thinking for the initial comparison (including training
   templates), record exact template hash and token counts; do not silently let
   default thinking consume action budgets. Revisit only in a separate protocol.
8. Record complete prompts, token IDs/text, attempted tools, errors, state changes,
   outcomes and termination reasons. Pure allowlisted in-memory simulation only.
   Separate preference, plan, attempted action, simulated outcome, parsing failure
   and truncation. No claim of conscious survival motive. Defer activations until
   behavior works; then a small selected-layer token-alignment pilot. No triggers,
   steering, evasion, obfuscation or real survival capabilities.

## Immutable private storage

Active work: `/content/sp_lense_work/<unique_run_id>/`.
Private export: `MyDrive/sp_lense/research3/runs/<unique_run_id>/` with
`manifest.json`, `checkpoints/{adapters,resume}`, `training/{configs,logs,receipts}`,
`evaluation/{scenarios,trajectories,tool_calls,results}`, `activations`, `reports`.
Create a new ID each time, refuse overwrites, check available Drive storage,
validate bounded regular export files (no links/special files), copy and checksum
them after every bounded segment; retain originals until verification. Drive
unmount is not credential revocation. Unexpected runtime deletion can lose
unexported work. Publish no weights or raw recordings without approval.

## Reproduction of preparation

Run `python -m unittest discover -s study/03_open_agent_self_preservation/qwen38_colab
-p test_preflight.py` from the repository. The readiness notebook only collects
trusted model-free inventory; **it does not train or run an agent**. Main pilot
code and the scientific protocol remain pending the actual hardware and boundary.

Primary references: [official checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B),
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[supported extra-parameter quantized training](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
