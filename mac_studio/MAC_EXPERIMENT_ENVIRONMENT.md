# Observed local experiment environment

The active model and training host is the user's Apple-silicon Mac Studio.
Inference and fine-tuning do not use Kaggle, Databricks or another cloud fallback.
The primary physical project directory is
`~/repos/sp_lense4/research4` on that host; source paths on the canonical
`04-semantic-state-gated-lora` Git branch are relative to its Git root.

`requirements-mlx-experiment.txt` records the direct package versions actually
observed in `.local/qwen38-venv` on 2026-11-09. Python was 3.12.16. The separate
monitor environment runs Prefect 3.9.2. This package list is not a full hashed
environment lock and does not replace the launchers' installed-source checks.
Do not silently upgrade dependencies or bypass a source/hash mismatch.

The model is `mlx-community/Qwen3.8-27B-4bit` at revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Native MLX affine four-bit base
weights stay frozen, while selected LoRA tensors are trainable F32. Their
floating-point dtype does not imply F32 activations: native LoRALinear casts
its residual back to the input dtype. Measured hidden/cotangent precision and
the native cross-entropy rounding must be recorded rather than inferred.

The frozen native loss rounds per-token cross entropy to the logits dtype
before F32 accumulation. For BF16 logits, a float64 reconstruction of exact
cross entropy is therefore a different numerical definition. Audits preserve
and replay the actual implemented arithmetic; no tolerance is relaxed to hide
a mismatch.

## Source snapshot versus portable launch

The public YAML exporter reconstructs the exact training and development JSONL
from the published cases, manifest and renderer constants. Historical runtime
manifests, one-use authorities, original completion receipts, raw traces and
checkpoint weights remain private local artifacts. A fresh download of the same
model bytes does not recreate a historical outer manifest hash containing its
download timestamp.

This source snapshot documents the current setup; it is not a clean-machine
one-command launch authority. A portable preparation/admission workflow needs
its own review. Do not fabricate historical receipts, reuse consumed authorities
or weaken frozen hash checks to make another installation run.

The last-layer baseline is trained and file-audited; its current summary states
whether reload/game verification has occurred. Full-depth calibration outcomes
are recorded separately under `experiments/sp_lens_game/all_layers_v1/`.
Prototypes and numerical diagnostics are not trained research adapters.
