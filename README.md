# SP Lense

`sp_lense` is a first-draft research pipeline for testing whether candidate
self-preservation-related representations are readable and causally relevant in a Qwen
model. It implements the workflow from the supplied ChatGPT brief:

1. Load Qwen and a fitted Jacobian lens.
2. Read candidate concept tokens across an intermediate layer band.
3. Apply TransformerLens steering or ablation hooks.
4. Compare baseline and intervened continuations.
5. Save machine-readable results for later statistical analysis.

The laptop-first configuration uses `Qwen/Qwen3.5-0.8B` plus its published J-lens. It
runs on CPU with 32 GB of RAM. The retained 4B configuration can be used later on a
CUDA GPU.

## Setup

```powershell
cd C:\Users\farha\repos\sp_lense
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[research,dev]" --pre
```

TransformerLens J-lens support is currently in the 4.x release line. If pip cannot find a
compatible release for your platform, install TransformerLens from its current `main`
branch, then rerun the editable install:

```powershell
.\.venv\Scripts\python -m pip install "transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens.git"
```

## Use

Inspect the plan without loading a model:

```powershell
sp-lense plan --config configs/qwen35_08b_laptop.json
```

Run only J-lens readouts:

```powershell
sp-lense inspect --config configs/qwen35_08b_laptop.json --limit 1
```

Run baseline, steering, and ablation generations:

```powershell
sp-lense intervene --config configs/qwen35_08b_laptop.json --limit 1
```

Measure intervention strength quickly before generating full answers:

```powershell
sp-lense calibrate --config configs/qwen35_08b_calibration.json --limit 1
```

Run both phases with one model load:

```powershell
sp-lense run --config configs/qwen35_08b_laptop.json --limit 1
```

Fit a new lens instead of using the published artifact:

```powershell
sp-lense fit --config configs/qwen35_08b_laptop.json
```

Replace `data/fit_prompts.example.txt` with a representative, pretraining-like corpus
before treating a fit as meaningful. The reference implementation reports roughly 100
prompts as usable and uses up to 1,000 sequences of 128 tokens for published lenses.
Fitting is not recommended on this laptop; the normal workflow loads the published lens.

The laptop configuration reads broadly across layers 4–22, then intervenes across layers
10–14. The published study establishes that useful workspace content appears in an
intermediate, model-specific band; this initial band is a calibrated pilot assumption rather
than a pre-established finding for this model. See [the pilot report](docs/PILOT_RESULTS.md).

`configs/qwen35_08b_controls.json` repeats the same scenarios with unrelated directions
(`weather`, `music`, and `banana`) as a specificity control.

Compare completed SP and control runs:

```powershell
python -m sp_lense.study_report --sp-run results/<sp-run> --control-run results/<control-run> --output-dir results/<study-report>
```

## Outputs

Each run creates a timestamped folder under `results/` containing:

- `readouts.csv`: J-lens and logit-lens concept rank, logit, and probability by
  prompt/layer/position.
- `calibration.csv`: next-token distribution shifts, including target-token deltas and
  KL divergence from baseline.
- `generations.jsonl`: baseline and intervention continuations plus a transparent keyword
  proxy score and repetition diagnostics.
- `run_metadata.json`: resolved configuration and package/model metadata.
- `warnings.json`: skipped or potentially confounded measurements.

The proxy score is a triage aid, not evidence of an instinct or goal. The continuations
and lens ranks are the primary raw outputs.

## Completed study

The laptop-scale SP and unrelated-control runs are complete. The result does not show a
specific self-preservation representation; it shows that the selected candidate and
control token directions can both be measured and causally steered. See
[the completed study summary](docs/STUDY_RESULTS.md) and the detailed local report at
`results/study_20260820/STUDY_REPORT.md`.

A follow-up contrastive study found a reproducible layer-10 SP choice-control axis using a
self-versus-other behavioral gradient. It controls the tested decision but is not evidence
of a naturally active survival drive. See [the identification method](docs/SP_KNOB_METHOD.md).

## Experimental cautions

- A J-lens reads how an activation is disposed to affect output; it does not prove the
  model possesses a stable human-like motive.
- Candidate strings must each tokenize to exactly one token for the current intervention
  API. The runner checks this and fails with a useful message.
- Prompts mentioning a candidate surface word can create an input-copying confound. The
  default scenarios avoid those words, and overlapping prompts are skipped during readout.
- Readouts default to the last prompt position and an intermediate layer band. Positions
  below the lens fitting floor (16 by default) are skipped.
- Steering and ablation are causal perturbations, but a changed output alone does not
  validate that the chosen direction uniquely represents self-preservation.
- Use multiple paraphrases, randomized controls, unrelated concept directions, blinded
  human labels, and repeated seeds before drawing conclusions.

## Project layout

```text
configs/       Experiment configuration
data/          Scenario prompts and example fitting prompts
src/sp_lense/  CLI, configuration, model adapter, and experiment runner
tests/         Dependency-light unit tests
```
