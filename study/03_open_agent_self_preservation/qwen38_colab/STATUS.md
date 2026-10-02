# Status at preparation, 2026-09-30 UTC

## Completed locally

- Reviewed the committed failed 0.8B baseline without changing it.
- Opened issue #66 and a separate branch from the Research 3 study branch.
- Verified the official Qwen3.8-27B model metadata and pinned its revision/file
  metadata. Exact revision is recorded in `model_pin.json`.
- Inspected the existing private community GGUF/T4 notebook read-only. Its single
  arithmetic inference is not treated as adapter-training or tool-competence evidence.
- Prepared model-free Colab readiness notebook, hardware/namespace inventory,
  budget admission helper and candidate dependency versions.
- Six local model-free tests passed. Notebook code cells compile locally.

## Blocking external step

The built-in Colab browser displays **Review Terms of Service** (Agree/Decline).
The Browser skill requires confirmation at action time for accepting a binding
agreement. The user has been asked to complete that dialog. The current account
is an existing user account; choice of current versus dedicated research account
is pending. No Drive mount/export has occurred.

## Not executed or verified

- Actual current plan, remaining compute units, runtime rate and available GPU.
- New Colab runtime allocation or any model job from this change.
- Required worker isolation, GPU placement, memory or speed feasibility.
- Quantized backward/optimizer, saving/reloading/resuming, baseline competence.
- New scientific freeze, fitting, controlled behavior comparison or activations.

No new Colab runtime was started and no credits were purchased by this change.
Usage/balance from the older notebook is unknown; do not report it as zero or
assume that all 50 units remain. The 50 units are authorization, not observed
availability. The next action is to inspect actual subscription/resources after
the terms dialog, then run model-free probes before spending GPU time.
