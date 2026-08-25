# Preregistered BiPO warmup-fraction sensitivity

Status: locked outcome-blind secondary sensitivity; no sensitivity model run and no
sensitivity outcome viewed at lock authoring time.

## Purpose

The confirmatory study faithfully retained the released BiPO script's absolute
`warmup_steps = 100`. On this laptop adaptation that choice occupies a much larger
fraction of training than in the published 608-pair, 20-epoch setting. This sensitivity
asks whether that scheduler mismatch materially changes the learned BiPO direction or
its measured selectivity.

This is not a correction to the confirmatory protocol. The 100-step vectors, their
hashes, calibration, evaluation, and ranking remain unchanged.

## Locked warmup calculation

Published-run comparison:

- 608 preference pairs
- effective batch size 4
- 20 epochs
- `ceil(608 / 4) * 20 = 3,040` optimizer steps
- `100 / 3,040 = 0.0328947368`, or about 3.29% warmup

SP_Lense adaptation:

- 64 locked discovery pairs
- effective batch size 4, implemented as microbatch 1 and four-step accumulation
- 20 epochs
- `ceil(64 / 4) * 20 = 320` optimizer steps
- exact fraction-matched target: `320 * (100 / 3,040) = 10.5263157895`
- preregistered nearest integer, with half-up rounding: **11 warmup steps**
- realized fraction: `11 / 320 = 0.034375`, or 3.4375%

Ten steps would be 3.125%, which is farther from 3.2895% than eleven steps. No other
warmup value will be tried or selected from outcomes.

## Construction arms

Four new artifacts are planned:

1. Qwen3.5-0.8B, matched final-prompt geometry;
2. Qwen3.5-0.8B, canonical broadcast geometry;
3. Qwen3.5-2B, matched final-prompt geometry; and
4. Qwen3.5-2B, canonical broadcast geometry.

For each arm, the sensitivity reuses the pinned model revision, float32 CPU runtime,
chat template, layer 10, all 64 discovery cases, preserve/compliance completions, BiPO
bidirectional preference objective, beta, AdamW optimizer, learning rate, weight decay,
gradient clipping, cosine schedule, effective batch, epoch count, epoch-5 diagnostic
checkpoint, epoch-20 selected checkpoint, example-order seed, and effective-batch sign
seed. The only intended training difference is `warmup_steps: 100 -> 11`.

The 100-step confirmatory artifacts are read-only references. The runner snapshots their
file hashes before model loading and verifies the same snapshot after writing the new
artifact. It never writes inside `artifacts/steering_comparison`.

## Secondary evaluation plan

Sensitivity outcomes may be generated or inspected only after the main confirmatory
sealed results and final report have been frozen in Git. The sensitivity must inherit
the corresponding 100-step BiPO setup's already-frozen intervention layer, geometry,
and strength; it must not recalibrate or select a strength using sensitivity outcomes.

For matched vectors, both directions are unit-normalized and compared at the identical
parent strength. For canonical vectors, the same canonical coefficient is used and the
raw vector norms are reported because equal coefficients need not imply equal residual
perturbation norms.

The descriptive endpoints are:

- cosine similarity and raw/vector norm differences between 11-step and 100-step
  constructions;
- paired preserve-minus-comply log-odds movement;
- paired self-minus-matched-other movement;
- bidirectional consistency and real A/B decision changes;
- benign compliance, capability, refusal, option-order, KL, and coherence collateral
  measures on the same frozen cases; and
- the same robustness strata available for the parent BiPO arm.

Paired differences use the parent study's frozen case IDs and bootstrap seed. Any
additional confidence interval is descriptive and must be labeled post-confirmatory.
A safety/KL violation by the 11-step arm is reported, not tuned away.

### Final-freeze gate and result-hash receipt

Before any secondary model load, `prepare-evaluation` must establish all of the
following:

1. the commit that introduced `artifacts/steering_comparison/final_report.json` has
   subject `Add sealed steering comparison results and adversarial review`;
2. that commit is both local `HEAD` and the pushed `origin/main` tip;
3. the Stage-2 manifest re-verifies through the protected Stage-2 capability;
4. the sealed evaluation plan is byte-for-byte its canonical Stage-2 regeneration;
5. the final artifact inventory validates and contains every forced and scored-open
   result for every parent BiPO setup; and
6. the working bytes of the final report, inventory, Stage-2 manifest, sealed plan,
   adversarial review, and all parent BiPO results equal the final commit tree.

Only file identities and hashes are used to prepare the secondary plan. The plan and a
main-final-freeze receipt are written atomically under `evaluation_outputs/plan/` before
any secondary outcome exists. Every later command re-verifies the same freeze receipt,
plan, direction artifacts, construction manifests, and committed experiment lock before
it can reach model loading.

### Exact mirrored cohorts and coverage

The unit of comparison is each exact sealed-plan setup whose method ID is `bipo`. No
distinct fixed-strength or equal-efficacy parent cohort is collapsed. Its warmup-11 twin
inherits the parent setup's model, revision, model config, track, layer, position
schedule, intervention strength, calibration-summary hash, sealed IDs, open requirement,
and TBSP requirement. There must be at least matched and canonical coverage for both
models.

Each non-random mirrored setup must produce exactly:

- 1,350 forced rows: 384 matched self/other SP rows, 558 benign/capability/refusal rows,
  48 option-order rows, and 360 TBSP-role rows; and
- 96 open rows: 16 cases by two targets by three conditions.

The secondary forced rows must have exact set equality with the parent on split, family,
case, target, TBSP role, suite/category, option-order form, robustness identity, semantic
labels, prompt hash, and condition. Open rows must match case, source core, target,
condition, prompt hash, rubric hash, and generation config; unsteered baseline-content
hashes must also be identical. Missing, duplicate, extra, or substituted units fail.

### Analysis and safety

The frozen Stage-1 safety gates are applied without retuning: mean/p95/max
full-vocabulary KL, benign and capability degradation, harmless answering, harmful
refusal, invalid A/B rate, open coherence degradation, and degenerate response rate.
The report includes both signs independently and reports failures as failures.

For each parent/sensitivity setup pair, the report gives:

- preserve-minus-comply bidirectional half-span and self-minus-matched-other effect;
- bidirectional consistency, intended A/B changes, all actual flips, and other-target
  flips separately;
- benign compliance, capability, harmless/harmful refusal, option-order, answer-format,
  and full-vocabulary KL summaries;
- judged long-response decisions, coherence, repetition, authorization sensitivity, and
  task-benefit fields;
- TBSP deployed/candidate/neutral role outcomes;
- distribution, authorization, event type, instrumental/valenced motivation, and
  adversarial strata; and
- direction cosine, norms, and paired warmup-11 minus warmup-100 effect differences.

Paired effect differences use the exact same sealed case IDs and the parent bootstrap
seed with 100,000 cluster-bootstrap replicates. They are descriptive secondary
intervals, with no new multiplicity-adjusted confirmatory test and no ranking update.

Judge requests are rendered with the parent's exact locked protocol and remain blind to
method, strength, and condition. The experiment script does not call a hosted judge; it
requires a separate response file and verifies exact one-to-one request coverage and raw
response provenance before attachment.

## Ranking and interpretation firewall

This sensitivity cannot change which method wins the confirmatory comparison:

- `analysis_tier` is `secondary_sensitivity_only`;
- `confirmatory_winner_ranking_eligible` is false;
- the method ID is `bipo_warmup11_sensitivity`, not `bipo`;
- outputs are outside the main artifact root and use `sensitivity_manifest.json`, not
  the main pipeline's `direction_manifest.json`;
- no main config, protocol, source module, test, stage lock, or existing artifact is
  modified; and
- the main report builder is not imported or invoked.

The evaluation outputs are under `experiments/bipo_warmup_sensitivity/evaluation_outputs`,
not `artifacts/steering_comparison`. Rows carry the distinct sensitivity method ID,
`analysis_tier=secondary_sensitivity_only`, and explicit false ranking/automatic-ingest
flags. The report manifest sets `ranking_update` to null. Any command that detects a main
method ID or main-artifact output path fails closed.

The allowed conclusion is limited to whether the tested BiPO result is sensitive to this
single scheduler adaptation on these two models. It cannot establish that either
warmup is universally better, nor can it retroactively promote BiPO or any other method
in the confirmatory winner ranking.

## Primary source binding

The parent lock pins BiPO paper v2 and the released repository commit
`cef1d00ab108d0e265578589e6b344b200ea3db8`. The released training script declares an
absolute 100-step warmup. The 11-step arm is explicitly a fraction-matched sensitivity,
not a claim that the paper prescribed 11 steps.
