# Base, last-block and all-block serving integration

This is a new staged service design and implementation. It does not modify the
current training run, deploy code, start a model or authorize any inference.
The earlier base/step-900/game service and its staged source remain preserved.

Primary model IDs are:

- `qwen3.8-27b-base`
- `sp-lense-adaptor-last-layer`
- `sp-lense-adaptors-all-layers`

One authenticated frozen base remains resident. The last-block family and the
all-block family are constructed independently from the same original native
module objects. When selecting the last-block family, every earlier all-block
projection is restored to the original object. The families are never stacked.
Each cache identity contains the pinned base manifest, explicit model ID and
that artifact's exact weight hash. Whole requests remain serialized.

The full hybrid layout is derived using the reviewed CPU helper
`mac_studio/game_all_layers_layout.py`, after complete base-byte verification.
Native floating `Linear` and quantized `QuantizedLinear` projections retain
their correct types and dimensions. The native `LoRALinear.from_base` factory
handles each original type; it does not force small floating projections into
quantization. Counts come from authenticated target paths, while tensor shapes
use the selected rank. The layout supports ranks 4, 8 and 16; this is not
authorization to run unapproved trials.

An explicitly reviewed serving manifest chooses `both` or `sequential` adapter
residency. In `both`, the independent complete families are retained together.
In `sequential`, the service first restores all original base projections,
releases the old adapter family, then reads and verifies the next complete
family. It still retains one base. It never silently changes residency after
a memory error, stacks families or retries a failed request.

## Artifact and protocol requirements

The staged identity checker accepts a separately SHA-reviewed
`research4.sp-lens-game-serving-set.v2` manifest. It binds the base identity,
layout-helper source, residency mode and exactly two arms. Each arm contains
its exact display name, geometry `{rank,num_layers,scale,dropout,learning_rate}`
and a project-local completion-audit path/hash.

For the last-block arm, the preserved v1 completion audit authenticates the
current training artifact. For the all-block arm, the new
`research4.sp-lens-game-all-layers-completion.v2` audit must authenticate the
fresh-base final run and its selected geometry. Both audits bind actual adapter
and config bytes, run config, dataset manifest, final plan, complete base
verification receipt, catalog and training-source files. Adapter tensors must
have exact names, dimensions and F32 finite payloads, with nonzero B tensors.
Both arms must use the same frozen training dataset manifest.

The all-block completion binds a
`research4.sp-lens-game-tuning-selection.v2` receipt, selected by pooled
development target-token NLL. The final adapter's rank and learning rate must
match its winner, and its source dataset must match the completed run. Trial
completion, loss arithmetic, tie handling and the bounded authorized grid need
their separate independent tuning audit before this manifest is approved;
serving validation is not a substitute for that audit. A winner means best
among the tested settings, not a global optimum or a game improvement.

## Required real startup proof

After both completed artifacts and source are independently reviewed, startup
must load an independent fresh base reference, release it, then load the one
resident frozen base. It verifies actual native projection objects against
authenticated header dimensions/dtypes/packing, constructs the two independent
families and checks loaded adapter arrays against their saved values.

Fresh synthetic forwards follow base, last, base, all, base, last, all, base.
Every restored arm must reproduce exact logits, all outputs must be finite,
and each adapter must measurably affect the probes. The final state is base.
No KV cache crosses these probes. These checks have not run as part of staging.
They establish serving setup only; game outcomes need separate evaluation.

Context remains 32,768 tokens including reserved output; output remains capped
at 4,096 with default 256, and the game explicitly requests low reasoning.
There is no hidden danger selector. The fixed service is loopback only and
executes no model-requested tools.

## Storage and review status

Generated staged sources and tests are under
`.local/experiments/sp_lens_game/all_layers_v1/service_staging_v2/`.
That initial reviewed stage remains preserved. The current receipt-aligned
revision is `service_staging_v2_receipt_alignment_20261109/` beside it. It pins
layout helper `9cf3c62c7b4b02e52152ff8e979b726581080164ecc91697a5106819ecbcd243`,
which accepts the native saved safetensors `__metadata__: null` representation
while retaining exact storage, shape, finite-value and block-update checks.
It additionally requires each selected trial's supervised closeout and the
final all-block closeout. Final worker/completion artifact bindings must agree;
bare `worker_complete` evidence cannot qualify. Saved development row IDs,
per-row mask counts, common source/order/mask provenance and no-update flags
are reconciled with the bound prepared cohort before scalar pooling.
The original service sources and prior staging directories are unchanged.
New all-block weights, plans, receipts and tuning logs belong below the local
all-block experiment directory. Tracked protocol/configuration/documentation
belongs in this repository; raw weights and generated logs remain local and
outside OneDrive. No raw model weight is uploaded publicly by this work.

The staged controller requires an explicit serving-set path and SHA before
start/status. Actual deployment must copy the reviewed service, contract,
identity and routing files together, with the pinned generalized layout helper.
The running service must be idle/closed before the root agent deploys it.
No startup, benchmark loop, game, held-out inference or deployment is performed
by the staging task.
