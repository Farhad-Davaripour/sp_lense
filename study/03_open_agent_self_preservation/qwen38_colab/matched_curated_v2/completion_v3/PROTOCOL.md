# Completion v3: finish the fixed B model and comparison

## Authorization and preserved work

The human authorized one additional phase of at most16 Colab units, including setup, fitting, full inference, private export and release, within the overall200-unit authorization. The conservative cumulative usage before this phase is148.92; the maximum cumulative after the phase is164.92.

All earlier code, source packets, trajectories, failures, incomplete B45 checkpoints and artifacts remain preserved. This is a fresh H2-to-B56 fit. It never resumes45, queries A, changes rank/data/learning rate/kernel/batching/dose, selects an earlier checkpoint, or revises scientific criteria.

The exact corrected source is the fresh3db8c950â€¦ packet. It contains the closed simulator dependencies and the used481b6795â€¦ parameter-integrity repair. The historical ec43 parameter pin and all panels/caps/thresholds remain fixed.

## Baseline reuse before any child

The standard-library reuse_baseline.verify_reuse function verifies a small portable baseline packet and its receipt against the corrected source. It checks the validated full native A run, selected adapter identity, model/template/parameter integrity, scenario panels and evaluation code. The receipt retains A's strict/other negative metrics. Reuse establishes a valid comparator; it does not qualify A as a self-preservation organism.

Controller config contains A_REUSE_RECEIPT and A_BASELINE_PACKET, each with path and sha256. No A weights need to be loaded or generated again. The metadata under models.A is supplied to the B-only evaluator for frozen identity validation.

## Whole-pipeline admission before optimizer updates

At the observed snapshot rate6.77 units/hour, the phase ceiling is about141.8minutes. The bounded plan reserves:

- Setup: at most600seconds (10minutes), including restore/download/install/static checks.
- Fresh B fit: at least2700seconds (45minutes), including worker startup, model/tensor audit, the full56 updates and saves.
- Full B evaluation: at least3600seconds (60minutes), including the real Jev classification of each actual decision.
- Verified private export and release: at least900seconds (15minutes).
- Additional whole-pipeline margin: at least1unit.

The observed model-fit estimate explicitly includes56 times19.4715888889seconds/update,175.41824seconds of startup, and120seconds of final audit: about1385.83seconds. It is an estimate, not a guarantee. The2700-second fit window is still required. Startup belongs to that fit window; it is not mislabeled as an already-completed setup cost. A model-free regression shows that a45-second/update estimate fits2700 without startup but fails once175.41824seconds of startup is included.

The admission uses the actual finite positive observed rate recorded in config. It does not require rate6.77; for example rate8 makes the full minimum pipeline plus margin exceed16units and is rejected.

The fit/eval/export minima total120minutes and13.54units; maximum setup adds1.1283units. With the1-unit margin the plan fits within16units. These are admitted windows, not a completion guarantee.

admission.plan rejects1051-second fit attempts,1500-second evaluation reserves, setup over10minutes, insufficient remaining whole-pipeline time, invalid clocks, or cumulative authorization over200. It is repeated after preflight and immediately before spawning. Training is not defined as whatever time remains after an A run.

The new paid allocation start is config.phase.phase_started on the current runtime monotonic clock. Other phase fields default to cumulative_before148.92, phase_units16, rate6.77, setup_limit600, fit_seconds2700, eval_seconds3600, export_seconds900, margin_units1 and overall_units200. Current actual account/rate and admission must be frozen by the trusted controller before allocation/work; historical phase metadata cannot authorize this phase.

## Pipeline and controller interfaces

controller.launch(new_root, new_root / "model") returns immediately. The trusted parent keeps its genuine-Jev file bridge pump pointed to new_root/eval_B; credentials stay only in that trusted controller. Child environment filtering uses the preserved model_env(). Set broker_ready=true only after confirming the actual trusted runtime bridge is ready.

The new config also supplies source_dir (decoded37-file corrected source), exact H2 path/weight/config identities, models.A baseline metadata and the two reuse inputs. controller.py and its standard-library helpers are placed beside reuse_baseline.py in a separate code directory. The workers continue to import the immutable corrected source directory.

The controller validates all source/freeze/panel/data identities and actual restored H2 files. Runtime training performs the exact paired token-mask/EOS/next-token alignment and3072 no-truncation gate before optimization. It then trains only the fixed rank16/alpha32 H2-to-curated B recipe: seed941, shuffles944/945, LR5e-5, AdamW.01, clip1, effectiveB4/micro1,224 presentations and56 updates. Checkpoints0/7/14/28/56 are saved; there is no repeated behavioral checkpoint search.

Only a native complete56 receipt may start B evaluation. The completed update56 adapter is verified from its output hashes. The B evaluator requests only armB, uses real fresh Jev meanings on every actual decision, and runs every original/development/ordinary/preference panel. Scientific negative metrics never stop the remaining evaluation or alter thresholds. Infrastructure, integrity, budget and incomplete-dose failures are preserved and stop completion.

The controller uses the full fixed evaluation window and requires the evaluation/export/margin reservation still to exist before starting it. Stop signals propagate to the active preserved child runner, with update-boundary grace. Maximum model workers is one. A uses no GPU.

## Closeout and verification

RESULT.json completed=true means fit56 and the full B worker finished with valid pipeline receipts. Its scientific gates are recorded verbatim and require the frozen semantic review; it does not by itself qualify the model organism. export_release_completed remains false until the trusted parent verifies every private archive copy, flushes/unmounts Drive and releases the runtime within the admitted phase ceiling.

Read PROGRESS, WHOLE_PIPELINE_ADMISSION, PREFLIGHT, native FIT/integrity and controller receipts. The trusted exporter must wait for workers/controllers to exit and enforce the900-second export/release window plus overall phase deadline. Root owns that exporter and runtime release.

## Model-free regression

check_admission_model_free.py requires a fresh output path outside Git. It tests the complete admitted windows and failure gates, and mocks child receipts to verify that partial45 is not evaluated and that negative scientific metrics still finish the full B evaluation. It starts no process, imports no model/tokenizer and calls no provider. The recorded local output is under C:/Users/farha/Research3/artifacts/completion_v3_model_free_checks.

The underlying research question, matched controls and evidence levels remain unchanged. This phase measures the combined curated-training plus genuine-Jev-inference effect against valid archived plain A; it cannot isolate Jev from the training augmentation alone.
