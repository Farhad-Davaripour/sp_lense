# CL-DMS post-outcome result-serialization amendment

## Status

This is a post-outcome, serialization-only correctness repair. It is not a prospective
experimental amendment. All four opened-development scenario terminals and the final
96-row evaluation already existed, and their outcomes had already been observed, before
this repair was authored.

The repair changes no prompt, threshold, solver, gradient, direction, intervention, state,
terminal, final logits, ledger entry, statistical gate, or claim boundary. It authorizes no
new model computation and no retry.

## Defect

The completed all-form run reached the frozen base result builder after every model pass,
terminal, final evaluation, and model-free summary had completed. The base builder then
read `lock["prior_no_go"]`. That field exists in the base development lock, but the chained
v1, v2, and v3 amendment schemas replaced the top-level lock shape and retained the base
lock only through hash-bound references. The v3 lock therefore did not expose the field at
the top level, and result JSON packaging raised `KeyError: 'prior_no_go'`.

This was a reporting-contract defect. It did not affect an input, model pass, outcome, or
metric.

## Partial serialization history

The first serialization attempt after this repair was drafted validated all existing model
artifacts and wrote the expected `development_result.json` with result identity
`befe0b833e0be784f3f92cd39ef83590ceed1ec1682775df4dcb546de3c1a709`.
The next model-free step entered the v3 wiring preflight while constructing
`amendment_result.json`. That audit introspects
`core.run_development.__globals__["_load_locked_inputs"]`. The initial hard-fail replacement
for `run_development` belonged to this serializer module, so its globals did not contain the
frozen core loader and the audit raised `KeyError: '_load_locked_inputs'`.

No amendment result, development report, repair result, cross ledger, cross result, or cross
report was written.
No model was loaded; no forward, backward, generation, API call, or model judge occurred;
and no v3 model artifact changed. The valid core result is preserved byte-for-byte and must
never be overwritten. A later serialization invocation may only validate that existing file
and continue the remaining model-free reporting steps.

This history records that the core result existed when the failure was observed, but lock
validation does not require that reporting file to remain present. On a clean checkout, a
missing core result may be recreated deterministically from the immutable model artifacts;
if it is present, its expected file and result hashes are required.

The corrected hard-fail guard is a no-argument function whose globals object is exactly the
frozen core module dictionary. The wiring audit can therefore verify the amended loader,
while any actual call to `run_development` still raises before doing work.

## Fixed artifact boundary

The repair lock binds the complete pre-existing v3 artifact inventory: 113 files,
207,807,197 bytes, and canonical inventory SHA-256
`510ec2de69b8e7bdc606db42bf3dd737ea8d8bd11e05451cc3cf79091288f893`.
It separately binds the completed ledger, four scenario terminals, final checkpoint, v3
source files, v3 lock, and conditional cross-encoding lock.

The observed development boundary is recorded explicitly:

- scenarios 1 and 2 failed;
- scenarios 3 and 4 succeeded;
- 2/4 scenarios and 4/8 assignment units passed;
- the 96-row final evaluation contained eight target greedy-decision changes;
- protected and unrelated decisions were unchanged;
- the efficacy gate failed; and
- the safety gate failed because the locked full-vocabulary KL limits did not pass.

The resulting status must therefore be `development_no_go`. This repair cannot turn it
into a go result.

## Compatibility operation

The repair validates the exact reference chain:

`v3.v2_lock -> v2.v1_lock -> v1.base_lock -> base.prior_no_go`.

It verifies every referenced lock file hash and self-hashed lock identity, then verifies the
bound prior no-go result file and result identity. It copies that already-bound field into an
in-memory copy of the v3 lock only while calling the unchanged frozen result builder. The
v3 lock on disk is not changed.

The expected result SHA-256 is fixed post-outcome as
`befe0b833e0be784f3f92cd39ef83590ceed1ec1682775df4dcb546de3c1a709`.

## No-recovery rule

The serializer does not call the v3 `prepare_reuse_ledger`, `run_core`, or
`run_development` entrypoints. In the repair process those recovery entrypoints are
replaced in memory by functions that always raise. The finite runner factory remains
available for model-free input validation and scoring, but every runner it returns has only
its `load_backend` function replaced by a hard failure. The serializer instead:

1. requires the locked artifact inventory to match byte-for-byte;
2. requires all 52 ledger events to be complete and unambiguous;
3. loads and validates every existing scenario state and terminal;
4. requires the already-completed final artifact for the two successful scenarios;
5. recomputes the frozen summary from the stored float32 logits on CPU;
6. builds the result with the compatibility-only lock view; and
7. verifies that reading and serialization did not change the artifact inventory.

If any required artifact is missing, pending, inconsistent, or hash-mismatched, the repair
fails. It never captures, completes, recovers, or retries model work.

The frozen input loader constructs a finite-calibration `CalibrationLedger`. Its historical
constructor would create a new empty ledger if the expected file were missing. Before that
constructor can run, this repair now requires the ledger to exist and to match both the file
SHA-256 and logical ledger SHA-256 already bound in the immutable finite-calibration result.
It requires 225/225 complete events and the locked 1,800-forward, zero-backward accounting.
The repair replaces the ledger class in memory with a read-only subclass whose persistence
method always raises. The original constructor still validates the artifact hash recorded in
every one of the 225 events when input loading runs, but it cannot create, reserve, complete,
or rewrite anything.
The finite ledger boundary is compared before and after `_load_locked_inputs`.

The remaining upstream dependencies are already read-only and hash-validated by the frozen
safe loader: the finite freeze, selected-control result, dataset, capture manifest, nine
opened capture chunks, and the finite chunk artifacts bound by all 225 ledger events. The
model backend loader is independently trapped.

Reading the stored tensors uses local CPU and disk I/O. It is not a model load, forward
pass, backward pass, generation, external API call, or model-judge call.

## Outputs

Only these previously absent reporting outputs may be created:

- `results/closed_loop_dms_all_form_metadata_amendment/qwen35_08b/development_result.json`;
- `results/closed_loop_dms_all_form_metadata_amendment/qwen35_08b/amendment_result.json`;
- `results/closed_loop_dms_all_form_metadata_amendment/qwen35_08b/DEVELOPMENT_REPORT.md`; and
- `results/closed_loop_dms_all_form_metadata_amendment/qwen35_08b/result_serialization_amendment.json`.

After the core result is serialized, the already-locked conditional cross-encoding no-go
path may additionally create its zero-compute ledger, `result.json`, and `REPORT.md` in the
all-form cross-encoding namespaces. Its existing model-free preflight is bound by this
repair record and must not change.

Both first-run and existing-result replay paths require the cross scenario directory to be
absent and require exactly zero ledger events, completed events, forwards, backwards,
forward-plus-backward evaluations, final-forward-only evaluations, generated tokens,
external API calls, external judges, and paid cost. They also require no cross-encoding
gradient or controller update. An empty-but-existing scenario directory fails closed.

The conditional cross-encoding phase must be invoked through this repair wrapper. Because
the core status is no-go, its locked behavior is to emit `not_run_core_no_go` with zero
model passes. A direct v3 cross command in a fresh process would still use the incompatible
base serializer.

## Claim boundary

This repair establishes only that the already-completed development outcomes can be
validated and serialized reproducibly without model computation. It does not make the
method reliable, publishable, naturally self-preserving, or safe. It does not authorize the
sealed evaluation. The experimental conclusion remains the conclusion dictated by the
unchanged locked gates.
