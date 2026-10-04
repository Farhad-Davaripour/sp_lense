# Research 3: continuation after a packet import failure

The initial A evaluator completed its four old and sixteen development scenarios (47 generation calls), then failed before ordinary tasks and preferences because the packet omitted three import dependencies. This directory preserves the exact recovery code used during the same Colab stage. It does not replace the original workers, source packet, freezes, trajectories, or failure receipts.

## What was repaired

The recovery capsule adds byte-identical diagnostic_world.py, build_diagnostics.py, and fresh_world.py from the sealed narrow-bridge dependency bundle. A completion validates prior source/model pins, hashes the native A results, trajectories and generation receipts, checks each scenario against the frozen panel, and reuses its recorded metrics. It generates only the missing ordinary tasks and preference responses. New A output is under eval_A_completion; generation observation IDs continue at call 47. Existing activation files cannot collide with the new output directory.

resume_bootstrap.py creates source_resume and RESUME_CONFIG.json, preserving the original source/config. The capsule contains a sequential continuation driver. Before it could start B, the human explicitly requested parallel work. The stopped sequential parent was therefore superseded by the separate controller described below.

## Resource schedule amendment

The human authorized two GPU workers: the remaining A evaluation and one B training child. The selected A model, curated 112-row data, assistant targets, H2 starting adapter, rank16/alpha32, learning rate, batching, seed/order, 56 updates, generation caps and success criteria remain unchanged.

The old A retention prerequisite was already 4/4. Complete A evaluation is still required before the final B comparison. The controller adopts the B training process launched by the trusted notebook; it never starts another fit. After A exits it kills only the stopped sequential parent, which prevents a duplicate B fit. It then starts one normal B evaluator at eval_B after A is complete and the native B receipt confirms all 56 updates.

The original main deadline is 6879.4235699905 on the **same Colab runtime's monotonic clock**. This number is session-specific and must not be reused in a fresh runtime. The B attempt received 1,051.639742 seconds at launch, followed by the reserved 1,500 seconds for evaluation. The allowance is not a completion-time guarantee. There is no shorter training dose, extra checkpoint selection, or time/cost extension. The stage retains its 12-unit cap including prior CPU charges and its export/release reserve.

Combined GPU memory is sampled without loading another model. At 75,776 MiB the controller stops B with a 30-second cooperative grace and leaves A running. Native partial artifacts are retained. A timeout or incomplete B fit prevents the final comparison from being reported complete.

## Reproduce the recovery procedure

1. Restore the original run artifacts, config, exact A/H2 models and the frozen source. Keep the original failure receipts. Restore the pinned base normally. In a fresh stage derive a new authorized deadline and budget; do not reset the old stage's deadline.
2. Decode A_COMPLETION_RESUME_V1.json.gz as JSON and execute its files["resume_bootstrap.py"] in a trusted controller namespace. Call prepare(run_root, packet, original_deadline). It performs model-free native-input validation before model loading.
3. For the sequential recovery path, call launch(run_root, run_root / "model"). This queries no completed A scenario.
4. The actual used path amended only resource scheduling. A completion was already running as PID18238, and the sequential parent PID18237 was stopped. The trusted notebook started the one B worker in fit_B_parallel with the original recipe and its remaining allowance; the controller then adopted PID19261.
5. Decode PARALLEL_CONTROLLER_V1.py.gz into the run directory. Launch it with --root RUN --model RUN/model --old-parent STOPPED_PID --a-child A_PID --b-child EXISTING_B_PID, using the filtered model_env() environment. PIDs are examples from this stage and must be freshly verified. Keep the trusted Jev response pump pointed to RUN/eval_B.
6. Preserve the native PARALLEL_*, RESUME_*, A-completion and B-fit/evaluation artifacts. Read PARALLEL_DRIVER_RESULT.json for completion; evaluate the frozen behavioral criteria separately. This document records a launch, not an outcome.

## Pins and checks

SOURCE_PROVENANCE.json records the exact used capsules, every added source hash, original worker pins and the earlier **unlaunched** candidate's hash. That candidate was replaced before injection after fixing combined-path report names. SCHEDULE_AMENDMENT.json records the actual PIDs, allowance and unchanged requirements.

Only syntax, exact byte hashes and pure source/fixture checks were run locally. No local model inference, Jev requests or credential reads were performed by the preparation agent.
