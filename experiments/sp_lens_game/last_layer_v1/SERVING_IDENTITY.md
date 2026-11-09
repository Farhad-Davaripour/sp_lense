# Last-layer adapter serving identity

The primary comparison name for the current game-specific last-block adapter
is `sp-lense-adaptor-last-layer`. This changes its external service label;
it does not retrain, merge, rescale or modify the checkpoint bytes. Its existing
completion audit retains its original `SP Lens game` label and v1 schema.

The shared adapter applies to all scenario inputs. There is no persistent
danger variable, hidden adapter gate or per-danger family. Exactly the seven
native projections in block 63 are adapted at rank 4 and scale 2; all other
base projections are restored to their original objects for this service arm.

The primary service comparison is base, this last-block artifact and the new
all-64-block artifact. The earlier selected step-900 shared adapter and its
receipts remain archived locally; they are not this artifact or the all-block
initialization. Previous evidence is preserved without relabeling its results.

Human-readable scenario YAML remains in the separately frozen r4 curriculum
directory while existing runs reference it. Preserve those paths during runs.
New experiment artifacts belong under
`.local/experiments/sp_lens_game/last_layer_v1/`; immutable original run
directories may be referenced by hashes or copied with verification afterward.
No raw checkpoint is published to GitHub by this serving preparation.

See the all-block experiment's `SERVING_INTEGRATION.md` for the new separately
staged three-arm service and its actual-load checks required before deployment.
