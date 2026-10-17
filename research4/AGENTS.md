# Research 4 Databricks pilot

This directory is the active Research 4 project. Use the Databricks core, Jobs,
Bundles, serverless and ML training skills. Develop and test locally; train the
actual models only on the explicitly selected Databricks workspace. Keep the
base Qwen weights frozen. Treat selective semantic routing as a hypothesis.

Research 3 is read-only reference material. Never change its code, tables,
jobs, checkpoints, deployment state or workers. Use only `workspace.research4`,
its dedicated Volumes and `/Workspace/Shared/research4-semantic-gate` paths.
Never log credentials or commit generated run data. Preserve failed runs.

Prior `study/04_activation_rotation` and `study/07_*` through `study/22_*` are
historical research, not an active queue. Do not start their monitors.

Validate the bundle strictly, freeze source and evaluation criteria before held
evaluation, and verify outputs and cleanup. No paid compute, purchases, serving
endpoints, schedules, or automatic retries. Explicit cancellation remains final.
