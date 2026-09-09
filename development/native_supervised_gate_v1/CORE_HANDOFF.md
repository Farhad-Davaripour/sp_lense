# Supervised gate core: tested model-free milestone

2026-09-09. Core implementation completed within the twenty-minute engineering cap.
Root scope and subsequent sequence amendment are authoritative. No real feature fit,
tokenizer/provider/model import, checkpoint tensor read, sealed numeric-data read,
installation, model experiment, external service or commit was performed by this worker.

## Delivered

- `gate.py`: fixed balanced ridge fit, immutable gate, strict score>0 routing and canonical native artifact loading with externally expected hashes/bindings.
- `checker.py`: independent nonsymmetric weighted-dual reconstruction using Gaussian elimination; weighted normal-equation, parameter, prediction and discrete-route checks at1e-10. It never imports/calls the fitter or its Cholesky solver.
- `source_auth.py`: exact nine-row B/O/H allowlist; each selected raw row, inventory/input and native loader/source/JSON checkpoint-descriptor join checked against its stated Git commit. Authentication reads metadata and emits no residual coordinates. Future numeric extraction is separately callable and release-gated by the construction entry point.
- `construction.py`: model-free manifest/candidate generation and future default-deny one-shot construction entry point. It accepts only the root-bound construction operation, preserves scientific failure alongside later technical faults, shares one60-second deadline across reauthentication/extraction/fit/check/output phases, and leaves every actual permission false.
- `TRAINING_MANIFEST.json`: nine unique rows,4positive/5negative, exact raw identity joins. SHA256 `1970f752b394349bf7b926e46a7cc70fd90647bc197f9a52e008f9477ca598d4`.
- `CORE_SOURCE_LOCK.json`, `CONSTRUCTION_LOCK_DRAFT.json`, `RELEASE_DRAFT.json`: review candidates binding the final four core sources and fixed method. The draft release is `approved:false`; model/tokenizer/evaluation permissions are all false. They were refreshed model-free after routine code corrections, before any fit.
- `test_core.py`: fourteen focused synthetic tests, including a hand-solvable two-point example with coefficient10/11 and zero intercept, class imbalance, nonzero intercept/translation,9x1024 dimensions, deterministic reload, ties, invalid input, artifact/source tampering, exclusions, shared deadlines, default-deny and publication-fault preservation.

## Observed verification

`python -S -B development/native_supervised_gate_v1/test_core.py`

The first ten-test pass took0.132seconds. After adding the requested source-byte,
shared-deadline and fault-preservation cases, all14 tests passed in0.197seconds.
These are synthetic tests, not a Qwen construction fit. The final bound-only change
was followed by the same focused suite; the final elapsed time is recorded in the
worker's handoff response. `-S -B` avoids site-package initialization and bytecode writes.

`python -S -B development/native_supervised_gate_v1/construction.py candidate`

One metadata-only candidate pass completed successfully in the observed1.916-second
shell call, authenticated all nine selected records, and returned zero real fits,
model calls, tokenizer calls and checkpoint tensor reads. No feature coordinates
were printed or emitted. It did not read any referenced checkpoint tensor file.

`python -S -B development/native_supervised_gate_v1/construction.py fit --release development/native_supervised_gate_v1/RELEASE_DRAFT.json --release-sha256 63d08f96fc84037664eb90a8939892d188f92b82d8cb2931c8d45608173b730f`

That exact earlier candidate-release test exited1 with `RELEASE_DEFAULT_DENY`.
It created no `construction_attempt_001`; no extraction or fit occurred. The
recorded hash identifies the then-current unapproved draft, not the final refreshed
draft. A future release must bind the actual final bytes.

## Boundaries still pending

No real construction pass or training adequacy is established. Root must accept
the final source/manifest and independent review, then separately authorize exactly
one fit. A separately implemented retained deadline owner remains required before
real release: cooperative deadline checks in this core do not prove termination
of a blocked filesystem operation or process. This core does not implement that
owner, a native evaluation adapter, a cohort/census runner or evaluation authoring.
Other workers may add separately reviewed owner files; they are outside this
four-file core lock and this worker's verification claims.

Construction attempt artifacts/results/fault receipts have a separate enforced1MiB
aggregate cap, with64KiB reserved for a fault receipt before ordinary publication;
the parent construction contract remains8MiB total/5MiB per file. All engineering
files are under the16MiB namespace cap. Source-authentication failures retain
`SOURCE_FEATURE_GAP` semantics: no substitution, legacy feature rescue or refit.
No fit/preparation/model permissions can be inferred from these synthetic passes.

The immutable artifact binds the construction lock, training manifest, feature
digest and core source lock. Per the accepted sequence amendment, fresh clean
authoring occurs only after an accepted artifact/result; a later evaluation release
will bind that frozen artifact to its independently locked texts. The fixed old
development feature selection has only two semantic families and one ordinary
negative. Neither construction success nor these engineering tests establish
automatic-gate generalization or the user's overall reliable-control goal.
