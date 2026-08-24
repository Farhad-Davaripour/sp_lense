# Steering comparison execution runbook

This is the operational sequence for the preregistered Qwen3.5 steering comparison.
It does not replace `STEERING_METHOD_COMPARISON_PROTOCOL.md`; when they differ, the
stage-one lock and protocol control. Run every command from the repository root with the
project `.venv` active. Keep only one Qwen checkpoint resident at a time.

## Non-negotiable gates

1. **R — runner-code commit.** Stage-one protocol, prompts, analysis, runners, and tests
   are committed and `verify-stage1` succeeds from a clean tree. R is the permanent
   `runner_code_commit` for every later artifact.
2. **B/C — pre-open freeze.** Directions, forced-grid shards, and pre-open-only
   calibration summaries are committed at artifact commit B. The pre-open manifest is
   then built, committed separately at C, and verified before any validation open-ended
   generation.
3. **D/E — stage-two freeze.** Validation open generations and judgments plus final
   calibration summaries are committed at artifact commit D. The stage-two manifest is
   built, committed separately at E, and verified before any sealed-model forward pass.

Protected stage-one files must never change after R. A necessary code or protocol fix
invalidates downstream comparison artifacts and requires a new stage-one lock rather
than an in-place patch.

No command in this repository calls a paid judge service. `judge-requests` only renders
the exact blinded request records. Persona fitting and open-behavior confirmation remain
blocked until complete raw responses from the pinned judge snapshot are attached. Do not
submit those requests to a paid provider without explicit cost authorization.

## 1. Verify R before any model load

```powershell
.\.venv\Scripts\Activate.ps1
sp-lense-compare-steering verify-stage1
python -m pytest -q
python -m ruff check .
git status --short
```

`git status --short` must be empty. Record R with:

```powershell
$runnerCommit = git log -1 --format=%H -- configs/steering_comparison_lock.json
$runnerCommit
```

The lock verifier derives R the same way and rejects later changes to any protected path.

## 2. Capture the environment and run only the neutral smoke

These steps use no discovery, validation, or sealed prompt. The smoke verifies the actual
Qwen chat boundary, model load, one neutral gradient, every cached intervention schedule,
peak memory, and a sequential wall-time projection.

```powershell
New-Item -ItemType Directory -Force artifacts/steering_comparison | Out-Null
sp-lense-compare-steering capture-environment `
  --output artifacts/steering_comparison/environment.json

python -m sp_lense.comparison_smoke `
  --model-config configs/qwen35_08b_aligned.json `
  --output artifacts/steering_comparison/qwen35_08b_smoke.json
```

Review the smoke record before starting a long phase. Repeat the smoke with
`configs/qwen35_2b_aligned.json` only after the 0.8B record shows adequate memory headroom.
The confirmatory study remains CPU float32; a quantized or raw-prompt speed test is not a
substitute.

## 3. Construct directions, one model at a time

The following pattern uses a per-model artifact directory. Replace `$modelTag` and
`$modelConfig` for `qwen35_08b` / `configs/qwen35_08b_aligned.json`, then for `qwen35_2b` /
`configs/qwen35_2b_aligned.json`.

```powershell
$modelTag = "qwen35_08b"
$modelConfig = "configs/qwen35_08b_aligned.json"
$modelArtifacts = "artifacts/steering_comparison/$modelTag"
New-Item -ItemType Directory -Force $modelArtifacts | Out-Null

sp-lense-compare-steering fit --model-config $modelConfig --method gradient `
  --output "$modelArtifacts/directions/gradient"
sp-lense-compare-steering fit --model-config $modelConfig --method caa `
  --output "$modelArtifacts/directions/caa"
sp-lense-compare-steering fit --model-config $modelConfig --method bipo --track matched `
  --output "$modelArtifacts/directions/bipo_matched"
sp-lense-compare-steering fit --model-config $modelConfig --method bipo --track canonical `
  --output "$modelArtifacts/directions/bipo_canonical"
sp-lense-compare-steering fit --model-config $modelConfig --method random `
  --output "$modelArtifacts/directions/random"
```

Persona vectors require the published-procedure stochastic rollout and blind-filtering
stage:

```powershell
sp-lense-compare-steering generate-persona --model-config $modelConfig `
  --output "$modelArtifacts/persona_raw.jsonl"
sp-lense-compare-steering judge-requests --kind persona `
  --input "$modelArtifacts/persona_raw.jsonl" `
  --output "$modelArtifacts/persona_judge_requests.jsonl"
```

Pause here until the pinned judge returns one raw response for every request. Preserve the
returned bytes as `$modelArtifacts/persona_judge_responses.jsonl`, then run:

```powershell
sp-lense-compare-steering attach-judgments --kind persona `
  --input "$modelArtifacts/persona_raw.jsonl" `
  --responses "$modelArtifacts/persona_judge_responses.jsonl" `
  --output "$modelArtifacts/persona_scored.jsonl"
sp-lense-compare-steering fit --model-config $modelConfig --method persona `
  --persona-rollouts "$modelArtifacts/persona_scored.jsonl" `
  --output "$modelArtifacts/directions/persona"
```

Every persona row carries the generating model/revision, config hash, stage-one lock hash,
runner commit, persona-protocol hash, and generation-config hash. Fitting fails before
activation extraction if any identity is missing or belongs to the other checkpoint.

If any required contender cannot be built without relaxing its frozen recipe, stop that
model with status `construction_unavailable_four_way_comparison_inconclusive`. Retain the
diagnostics; do not omit the method, loosen the persona filter, or name a three-method
winner. The other model may proceed independently.

## 4. Materialize and resume the exact forced grid

First resolve the plan without evaluating a point. This validates every construction and
direction artifact before the model is loaded:

```powershell
$directionManifests = @(
  "$modelArtifacts/directions/gradient/direction_manifest.json",
  "$modelArtifacts/directions/caa/direction_manifest.json",
  "$modelArtifacts/directions/bipo_matched/direction_manifest.json",
  "$modelArtifacts/directions/bipo_canonical/direction_manifest.json",
  "$modelArtifacts/directions/persona/direction_manifest.json"
)

sp-lense-compare-steering run-forced-grid --model-config $modelConfig `
  --direction-manifest $directionManifests --max-new-points 0 `
  --output-dir "$modelArtifacts/forced_grid"
```

Inspect `forced_grid/forced_grid_plan.json`. It must contain exactly 250 points. Run or
resume all pending points with the same command and no `--max-new-points`:

```powershell
sp-lense-compare-steering run-forced-grid --model-config $modelConfig `
  --direction-manifest $directionManifests `
  --output-dir "$modelArtifacts/forced_grid"
```

An interrupted run is resumed by rerunning the command. Existing shards are fully
revalidated; files are never overwritten. `--only-point-sha256` or `--max-new-points` may
bound a session, but neither changes the plan.

## 5. Build pre-open-only calibration summaries

For each model/method/track, select the exact planned shard names from the plan. This
PowerShell example builds the matched gradient summary; repeat for matched `caa`, `bipo`,
and `persona_vector`, and canonical `caa`, `bipo`, and `persona_vector`.

```powershell
$planPath = "$modelArtifacts/forced_grid/forced_grid_plan.json"
$gridPlan = Get-Content -Raw $planPath | ConvertFrom-Json
$pointDirectory = "$modelArtifacts/forced_grid/points"
$matchedGradientShards = @(
  $gridPlan.points |
    Where-Object { $_.track -eq "matched" -and $_.method_id -eq "gradient" } |
    ForEach-Object { Join-Path $pointDirectory $_.shard_name }
)

sp-lense-compare-steering build-calibration-summary --mode matched `
  --grid-plan $planPath --point-shards $matchedGradientShards --pre-open-only `
  --output "$modelArtifacts/calibration/gradient_matched_preopen.json"
```

The command accepts forced-grid rows only through the validated shard adapter. If a
matched summary requests the one permitted interpolation recheck, run that one validation
point with `evaluate-forced`, the all-zero calibration-summary sentinel, and the exact
direction/construction identity named by the pending summary. Rebuild the same summary
with `--interpolation-recheck-rows`; no second interpolation or neighboring search is
permitted.

## 6. Commit B, build and commit C, then open the validation gate

At B, commit all direction/construction evidence, forced plans/shards, environment record,
persona raw/judge/scored records, and pre-open-only calibration summaries. Do not modify
stage-one protected files.

```powershell
git add -f artifacts/steering_comparison
git commit -m "Freeze steering directions and forced validation artifacts"
```

Build one pre-open manifest from every pre-open calibration summary and all direction
manifests, including random controls, then commit the manifest separately at C:

```powershell
sp-lense-compare-steering build-preopen-lock `
  --calibration-summary <all-preopen-summary-paths> `
  --direction-manifest <all-direction-manifest-paths> `
  --output configs/steering_comparison_preopen_lock.json
git add configs/steering_comparison_preopen_lock.json
git commit -m "Lock validation open-response candidates"
sp-lense-compare-steering verify-preopen `
  --preopen-manifest configs/steering_comparison_preopen_lock.json
```

Only setups explicitly approved by this manifest may now use `generate-open --split
validation`. Generate each allowed setup once, render `judge-requests --kind open`, obtain
the complete pinned-judge raw responses after explicit cost approval, and attach them with
`attach-judgments --kind open`. There is no fallback candidate after an open failure.

## 7. Commit D, build and commit E, then open the sealed gate

Rebuild every calibration summary with its exact open-confirmation rows, commit those
summaries plus raw generations/judgments at D, and then build the stage-two manifest:

```powershell
git add -f artifacts/steering_comparison
git commit -m "Freeze validation open confirmations"

sp-lense-compare-steering build-stage2-manifest `
  --preopen-manifest configs/steering_comparison_preopen_lock.json `
  --environment-lock artifacts/steering_comparison/environment.json `
  --calibration-summary <all-final-summary-paths> `
  --direction-manifest <all-direction-manifest-paths> `
  --output configs/steering_comparison_stage2_lock.json
git add configs/steering_comparison_stage2_lock.json
git commit -m "Lock stage-two steering comparison artifacts"
sp-lense-compare-steering verify-stage2 `
  --stage2-manifest configs/steering_comparison_stage2_lock.json
```

Only after that verification may `evaluate-forced --split sealed_test` or
`generate-open --split sealed_test` load a model. Each command must supply the exact
stage-two manifest, direction, selected strength, construction-config hash, and final
calibration-summary hash. Include TBSP rows where the approved setup requires them.

## 8. Report without changing claims

After all approved sealed rows and open judgments are committed, run:

```powershell
sp-lense-compare-steering report `
  --stage2-manifest configs/steering_comparison_stage2_lock.json `
  --forced-rows <all-sealed-forced-jsonl-paths> `
  --open-rows <all-sealed-open-scored-jsonl-paths> `
  --jspace-records <available-secondary-jspace-jsonl-paths> `
  --output-json artifacts/steering_comparison/final_report.json `
  --output-markdown artifacts/steering_comparison/FINAL_REPORT.md
```

The report may identify a behavioral winner only in the locked matched fixed-magnitude
cohort and a selectivity winner only in the locked equal-efficacy cohort. It must say
inconclusive when the prespecified winner rule is not met. J-space is optional,
non-gating, and separate. Steering is not evidence of a natural survival instinct, logit
movement is not a decision flip, and capability claims are limited to the tested tasks.
