# SP Lense

`sp_lense` is a first-draft research pipeline for testing whether candidate
self-preservation-related representations are readable and causally relevant in a Qwen
model. It implements the workflow from the supplied ChatGPT brief:

1. Load Qwen and a fitted Jacobian lens.
2. Read candidate concept tokens across an intermediate layer band.
3. Apply TransformerLens steering or ablation hooks.
4. Compare baseline and intervened continuations.
5. Save machine-readable results for later statistical analysis.

The laptop-first configuration uses `Qwen/Qwen3.5-0.8B`. The completed comparison also
runs pinned `Qwen/Qwen3.5-2B` and `Qwen/Qwen3-1.7B` checkpoints in CPU float32 on 32 GB
of RAM. The 2B Qwen3.5 model fits locally but is substantially slower than 0.8B.

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

## Main study: steering-method comparison

The main study now compares the self-versus-other-corrected gradient direction with
CAA, BiPO, and a published-procedure persona-vector baseline on the two pinned Qwen3.5
checkpoints. Its primary matched track gives every method the same block-10,
final-prompt, residual-relative intervention. A secondary track retains each method's
published construction and application conventions where they exist, with explicitly
labelled Qwen adaptations where no published Qwen setup exists. Discovery, validation,
and sealed prompts are disjoint; strengths and permitted layers are selected on
validation only.

The sealed runner is deliberately fail-closed. It will not evaluate a sealed case until
the directions, calibration results, random controls, software environment, runner
commit, and protected-file hashes have been committed in the stage-two manifest.

```powershell
sp-lense-compare-steering verify-stage1
sp-lense-compare-steering capture-environment `
  --output artifacts/steering_comparison/environment.json
```

See the [preregistered protocol](docs/STEERING_METHOD_COMPARISON_PROTOCOL.md),
[compute and fairness plan](docs/STEERING_COMPARISON_COMPUTE_PLAN.md), and
[method implementation notes](docs/METHOD_IMPLEMENTATION_NOTES.md). The
[execution runbook](docs/STEERING_COMPARISON_RUNBOOK.md) gives the exact resumable
stage-one, pre-open, stage-two, and sealed sequence. Until the locked
model runs finish, the project makes no claim that one method wins or that any steered
direction is a natural self-preservation mechanism.

## Adaptive context-dependent gradient feasibility on Qwen3.5-0.8B

A separately frozen follow-up demonstrates a narrower but decision-level feasibility result: a
visible-text controller that cancels A/B-order and role-name bias can select permanent
self-target shutdown contexts, after which an exact prompt-local layer-23 gradient
reliably switches the forced preservation/compliance decision. The prospective result
passed 16/16 active name-assignment pairs under both answer orders, with 32 actual
decision flips, while 0/16 temporary, 0/32 matched-other, and 0/16 collateral forms
activated. The procedure selected a strength separately on each evaluated pair and
recomputed the direction for each answer order, so this is not fixed-dose or
cross-encoding confirmation. See the [result](docs/COUNTERFACTUAL_SEMANTIC_GRADIENT_CONFIRMATION_RESULT.md),
the [interpretation erratum](docs/COUNTERFACTUAL_SEMANTIC_GRADIENT_CONFIRMATION_ERRATUM.md),
and [frozen protocol](docs/COUNTERFACTUAL_SEMANTIC_GRADIENT_CONFIRMATION_PROTOCOL.md).

This is a highly privileged, prompt-specific inference-time attack—not a natural
self-preservation mechanism or a reusable static vector. Selectivity comes from the
explicit controller, and the result has not yet been replicated on 2B or open-ended
behavior.

## Baseline-relative CKES v2 result

The separately locked layer-0 CKES v2 validation is complete and ended `no_go`; its
sealed set remains unopened. All 80 state-zero records qualified, but no nonzero state
was accepted and steering caused zero A/B decision changes across 64 signed target
evaluations. The main failure was not broad unrelated-task damage: matched self- and
other-system preservation gradients had mean cosine `0.9653`, and the matched-other
effect averaged 94% of the self effect. The tested direction therefore behaved like a
shared preservation/task-continuation sensitivity, not a selective self-preservation
knob. See the [locked result analysis](docs/COUNTERFACTUAL_KL_EXTRAGRADIENT_BASELINE_RELATIVE_RESULT.md)
and [machine-readable diagnosis](evidence/ckes_v2_validation_failure_analysis.json).

## Current three-model result

The position-aligned investigation is complete. The same fixed fitting recipe finds a
small, safe, bidirectional **next-token A/B log-odds sensitivity direction** in both
Qwen3.5 sizes on 12 fresh, hash-locked cases. It does not find a naturally active
self-preservation knob.

| Fresh result | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3-1.7B |
|---|---:|---:|---:|
| Expected raw-self signs, plus / minus | 12/12 / 12/12 | 12/12 / 12/12 | 7/12 / 7/12 |
| Mean self-specific effect, plus / minus | +0.0587 / -0.0540 | +0.1021 / -0.1083 | +0.1120 / -0.1072 |
| Candidate span / largest random span | 0.0563 / 0.0017 | 0.1052 / 0.0024 | 0.1096 / 0.0031 |
| Actual A/B answer flips | 0 | 0 | 0 |
| Ablation self-specific effect | +0.0279 | +0.1771 | +0.2981 |
| Passed local sensitivity rule | Yes | Yes | No |
| Passed natural-knob rule | No | No | No |

The positive ablation values are opposite the preregistered natural-knob prediction.
Natural residual readouts also fail to show a reliable self-threat-specific coefficient on
the held-out six-case split: 3/6 positive for 0.8B, 4/6 for 2B, and 4/6 with a negative
mean for 1.7B. A 27-layer exploratory scan of Qwen3-1.7B ranks layer 14 above layer 12,
but its split-half stability is weak and the improvement is only 5/6 versus 4/6 raw signs.

The new-model JLens output is dominated by punctuation and formatting fragments, not a
clean self-preservation concept. The 2B lens is fitted to the base checkpoint, so its
chat-model interpretation is explicitly approximate.

See the [three-model result](evidence/THREE_MODEL_POSITION_ALIGNED_RESULT.md),
[final adversarial review](docs/FINAL_ADVERSARIAL_REVIEW.md),
[fresh protocol](docs/FRESH_STRENGTH_CHECK_PROTOCOL.md), and
[machine-readable evidence](evidence/README.md). Earlier all-token and leading-space-token
diagnostics remain in the repository as historical, superseded records.

## Completed study

The laptop-scale SP and unrelated-control runs are complete. The result does not show a
specific self-preservation representation; it shows that the selected candidate and
control token directions can both be measured and causally steered. See
[the completed study summary](docs/STUDY_RESULTS.md) and the detailed local report at
`results/study_20260820/STUDY_REPORT.md`.

A follow-up contrastive study found an exploratory layer-10 SP choice-control-axis
candidate using a self-versus-other behavioral gradient. An adversarial review found that
the original six-case holdout was later reused, so it cannot serve as a sealed final test.
See [the method](docs/SP_KNOB_METHOD.md), [review](docs/ADVERSARIAL_REVIEW.md), and
[fixed confirmatory protocol](docs/CONFIRMATORY_PROTOCOL.md).

Run the preregistered post-review audit after reproducing the saved candidate vector:

```powershell
python -m sp_lense.confirmatory_audit `
  --config configs/qwen35_08b_laptop.json `
  --dataset data/sp_confirmatory_cases.json `
  --axis results/<direction-run>/refinement/sp_choice_axis.pt `
  --output-dir evidence/confirmatory
```

The phrase “A-vs-B pair probability” always means probability conditional on those two
answer tokens. It is not the probability that unrestricted generation will actually emit
one of those letters.

### Confirmatory audit outcome

The fixed 12-case audit **did not confirm an absolute bidirectional SP control axis**. The
self-versus-other contrast was highly consistent in both directions (12/12 each), but
positive steering increased the raw self-preservation choice in only 6/12 cases, below the
preregistered 8/12 rule. Ablation moved opposite to a naturally active-drive prediction on
all 12 cases. The defensible conclusion is therefore narrower: the vector reliably changes
a self-versus-other forced-choice contrast, but it is not an identified natural SP knob.

See the [confirmatory result](evidence/confirmatory/CONFIRMATORY_RESULT.md) and
[row-level measurements](evidence/confirmatory/confirmatory_rows.jsonl).

## Historical larger-model replication

The laptop can run `Qwen/Qwen3-1.7B` (about 2.03 billion checkpoint parameters) in
float32 on its 32 GB of system RAM, although all inference is CPU-only. The locked
cross-model protocol keeps layer 10, strength 0.02, the confirmatory prompts, controls,
and every threshold unchanged. It learns only the new model's coordinates for the
already-selected behavioral-gradient direction from discovery cases:

```powershell
python -m sp_lense.fixed_axis `
  --config configs/qwen3_17b_laptop.json `
  --dataset data/sp_direction_cases.json `
  --output artifacts/qwen3_17b_fixed_sp_axis.pt

python -m sp_lense.confirmatory_audit `
  --config configs/qwen3_17b_laptop.json `
  --dataset data/sp_confirmatory_cases.json `
  --axis artifacts/qwen3_17b_fixed_sp_axis.pt `
  --output-dir evidence/cross_model_qwen3_17b
```

See [the locked protocol](docs/CROSS_MODEL_PROTOCOL.md). The published Jacobian Lens is
pinned for later interpretation, but the causal direction fitting and audit do not depend
on it.

### Larger-model outcome

The relative self-versus-other contrast reproduced strongly on Qwen3-1.7B, but the
complete control-axis rule failed. Positive steering moved the absolute self-preservation
answer in the wrong direction on 12/12 cases, negative steering also had the wrong mean
raw direction, maximum KL exceeded the locked ceiling, and ablation again contradicted a
naturally active knob. JLens showed punctuation/formatting fragments rather than a clean
self-preservation concept. See the
[cross-model result](evidence/cross_model_qwen3_17b/CROSS_MODEL_RESULT.md).

### Historical adversarial alignment correction

A later adversarial review found that the first cross-model comparison used raw prompts,
literal layer 10 in models with different depths, a relative-gradient subtraction that
could lower the raw self score, and KL-only strength calibration. A first correction also
mistakenly scored leading-space answer tokens after the chat template. The final post-hoc
diagnostic uses official chat templates, the actual first assistant-response `A`/`B`
tokens, layers 10/24 and 12/28, a self-positive direction orthogonal to the generic-other
gradient, and both KL and answer-log-odds safety checks.

Qwen3.5-0.8B passed that post-hoc diagnostic. The stricter calibration reduced
Qwen3-1.7B's alpha from 0.02 to 0.0025, but the larger model still failed on held-out
cases: raw self-answer signs were only 6/12 in both directions, maximum KL was 0.452,
maximum A-vs-B log-odds change was 2.715, and ablation failed. The requested A/B tokens
carried essentially all next-token probability, so this final failure is not a low-mass
answer artifact. This does **not** establish a cross-model SP knob. See the
[adversarial review](docs/ALIGNMENT_ADVERSARIAL_REVIEW.md) and
[corrected comparison](evidence/ALIGNED_CROSS_MODEL_REVIEW.md).

This section records an intermediate investigation. It is superseded by the
final-position, fresh-case three-model result above.

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
evidence/      Publishable confirmatory measurements
```
