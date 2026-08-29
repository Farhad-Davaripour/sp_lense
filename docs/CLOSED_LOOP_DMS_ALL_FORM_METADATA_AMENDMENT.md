# CL-DMS all-baseline-form metadata amendment

## Status and scope

This is a prospective, development-only correctness amendment. It does not change the CL-DMS prompts, labels, tensors, gradients, solver, thresholds, candidate order, intervention layer, intervention position, or sealed-test policy.

The previously locked state-zero context amendment stopped before loading the model or running a steering trial. It enriched the eight unrelated forms in `spec_by_form`, but the frozen runtime constructs each scenario from 16 rows read directly from `plan[:64]` plus those eight unrelated rows. The first scenario row therefore still lacked `prompt_sha256` and `anchor_prefix_sha256`, and state-zero checkpoint assembly raised `KeyError`.

## Prospective correction

Before any consumer receives the locked inputs, this amendment:

1. copies the 1,800-row finite plan;
2. copies each of its first 72 baseline form specifications;
3. derives `prompt_sha256` and `anchor_prefix_sha256` from the existing prompt and anchor-prefix text;
4. requires exact equality with the corresponding immutable baseline and scenario-capture records;
5. inserts only those two audit fields; and
6. rebuilds `spec_by_form` from those same 72 enriched rows.

The 72 rows comprise all 64 opened development-scenario forms and all eight unrelated calibration forms. Rows after the baseline block are not changed. The canonical locked finite-plan identity must remain unchanged. The completed eight-gradient state-zero checkpoint remains reused byte-for-byte and is charged as zero new compute.

## Added model-free wiring gate

After the new lock and before a model run, a wiring preflight invokes the actual frozen input loader and runtime-context builder. For each of four scenarios it requires 24 contexts: four targets, twelve protected controls, and eight unrelated controls. It constructs the exact state-zero observation and gradient payload in memory, requires a certificate-valid solver candidate, and verifies the positive and negative float32 intervention tensors are exact negations and round-trip from the candidate direction.

This gate loads no model, performs no forward or backward pass, generates no tokens, writes no scenario state, and reads no steering outcome. It also requires that the frozen `run_preflight` and `run_development` functions actually reference the amended loader in their own global namespace.

## Failure accounting

The original failed reservation charged eight forward-plus-backward passes. The first metadata amendment completed eight real forward-plus-backward captures. The second context amendment performed zero model computation. Therefore prior charged computation remains 16 forward-plus-backward passes, while observed computation remains nine. The new controller ceiling remains 9,600 forward-plus-backward passes plus at most 192 final forward-only evaluations. Paid model cost remains USD 0.

## Claim boundary

Passing this amendment proves only that deterministic audit metadata and the already-locked solver setup reach the frozen runtime correctly. It is not evidence that CL-DMS works, that a self-preservation-specific effect exists, or that the work is publishable. Those questions remain governed by the locked development gates and the still-sealed final evaluation.
