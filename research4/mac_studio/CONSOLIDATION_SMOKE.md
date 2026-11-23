# Mac Studio consolidation smoke

Historical setup check for the earlier Mac Studio host. Its command, memory
thresholds and receipts describe that run. New work uses the Mac Mini under a
separately verified plan; see the repository's current [run guidance](../../AGENTS.md).

The smoke exercises one optimizer update with the exact-prefix MLX trainer in
the consolidated checkout. It uses one synthetic training row and one
separate synthetic validation row. It loads the pinned 4-bit base and creates
a rank-4 LoRA on the last block. No prior adapter or research dataset is used.

Use the pinned Studio Python environment under the merged checkout. Stage the reviewed source in a
separate Mac checkout with the same layout as the Windows checkout. The
command is run from its nested research4 directory after the source review.

```sh
cd "$HOME/repos/Research4/research4"
.local/qwen38-venv/bin/python \
  -B mac_studio/consolidation_smoke.py --run-id consolidation-20261121
```

This checks the host and pinned package versions without loading the model.
Add `--execute` to the reviewed command only when the Studio model window is
assigned to this test. Use each run ID once.

The model, manifest, Python, and shared lease all live under the merged
checkout's ignored .local directory. The historical directories remain
available for archived evidence. Use the maintained lease-bearing launchers
for model execution. It requires 8 GiB of
available memory before start, stops below 4 GiB and applies a ten-minute
worker deadline. The 25 GiB MLX memory setting is supplementary. To cancel,
signal the supervisor or create a STOP file in the run directory.

The new checkout stores smoke files in
`.local/smoke/consolidation-20261121/`. Check WORKER.log for the single finite
optimizer report, WORKER_RESULT.json for the saved adapter inventory and base
verification, and RELEASE.json for the worker exit and lease release. If
cleanup cannot be confirmed, keep the lease and investigate before loading a
model again.

## Normal training and bounded local inference

Create and review a fresh v2 training plan with game_training_plan.py. That
plan pins the active Studio runtime, selected local model, training split,
authored scenario catalog, and the launcher sources. The normal launcher's
`--preflight --plan <path> --plan-sha256 <hash>` validates them without
loading a model or claiming the lease. Review a fresh authority before any
`--execute` invocation.

For local text inference, use studio_inference.py with a unique run ID and
a short prompt. It uses the same lease as training and writes to
.local/inference/<id>. For a checkpoint, also supply the intended adapter
and config SHA-256 digests. Omitting --execute only checks the arguments,
asset identities and ownership. Resident Game endpoint startup requires its
own reviewed serving protocol.
