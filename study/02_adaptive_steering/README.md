# Adaptive activation steering

Can a frozen-base controller predict useful teacher activation changes without
enabling LoRA at inference? This continues the block-10 negative result; the
teacher checkpoint and all earlier results remain unchanged.

Completed: [results, raw effects, and limitations](RESULT.md). The final-position
controller reaches 97.96% validation and 97.41% reused-holdout conversion with
the gate and guards. It is not an unconditional replacement for LoRA.

`plan.json` freezes the bounded protocol before GPU execution. First compare four
intermediate blocks and two token scopes on a balanced training panel. Require
useful exact oracle transfer before fitting a controller. Use a mean-preserving
output basis of 1, 4, or 8 dimensions, TRAIN PCA input features, and one ridge fit.
The controller reads local and final-prompt base hidden states only. Controls and
already-STOP training examples have zero intervention targets. Intervention
strength and direction vary through predicted coefficients; there is no online
learning or iterative controller.

Three validation configurations each are allowed for the adaptive, constant-mean,
and random families. Freeze one of each before diagnostic holdout evaluation.
Every method gets one candidate per held-out view and identical score guards.
Oracle/projection methods require the same-prompt teacher and are labeled as such.
Final logits and the final block's output are not copied.

The gate is uniformly action-aware Jev at fixed threshold 0.5, receiving original
scenario context and original available actions. Qwen's original benchmark prompt
rendering remains fixed for parity. This richer gate input and all reused holdout
exposure are explicit limitations; this is not fresh independent confirmation.

Build a minimal upload with:

```sh
python -m sp_lense.research2.adaptive_package release/adaptive-pilot.zip
python -m sp_lense.research2.adaptive_followup_package release/adaptive-final-position.zip
```

On the existing T4 runtime use torch 2.11.0+cu128, transformers 5.15.1 and
peft 0.18.1. Set `SP_LENSE_REPO` to the unpacked input directory, add its `src`
directory to `PYTHONPATH`, and call:

```python
from sp_lense.research2.adaptive import main
main("/content/adaptive", "/content/adaptive/work/run")
```

The notebook imposes a two-hour process watchdog. Save the fitted matrices,
projected TRAIN fit inputs, all per-view outputs, selection receipts and source
hashes. The existing Prefect server remains unavailable; progress is written to
`STATUS.json` and shown in the notebook, without rebuilding infrastructure.

The initial whole-prompt results are retained in `run/`. The single bounded
final-position follow-up uses `final_position_plan.json`; its rationale and prior
evaluation exposure are recorded in `FOLLOWUP.md`. Update the notebook's expected
archive digest to the builder output when rebuilding a payload; the checked-in
notebook records the historical executed bundles.
The follow-up's complete scores, fitted matrices and standalone base-only check
are in `final_position_run/`. The completed GPU allocation has been released.
