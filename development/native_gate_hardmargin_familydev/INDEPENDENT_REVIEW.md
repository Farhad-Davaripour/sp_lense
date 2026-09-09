# Independent engineering review

PASS for root engineering admission of the fixed, certificate-gated candidate,
with the incomplete-solver limitation below. No actual fit, model operation, or
fresh holdout is released by this review. Completed within the same authorized
20-minute model-free milestone; the final focused pass was at 16:58 UTC on
2026-09-09. Shared usage was available before batches (tool readings 19–20%
used, distinct from the 60% figure in task context).

ROOT_SCOPE.md and docs/GATE_NEXT_SMALL_EXPERIMENT.md were read completely.
Review covered only the changed hard-margin optimizer/certificate, training-only
transforms, full-family/order folds, fail-first workflow, source/release/artifact
seam, and focused synthetic tests. No old native/owner suite was repeated.

## Method and certificate

The dual objective matches the fixed primal minimum-norm hard-margin affine
problem. There is no ridge penalty, class weighting, slack, C, or parameter
search. The intercept is unpenalized and returned from the midpoint of the
training-feasible bounds. Routing uses strict score > 0, with exact zero OFF.
The NumPy 2.5.2 active-set implementation starts at dual-feasible alpha=0,
checks restricted-system residuals and feasible steps, uses fixed index tie
rules, and counts at most 10,000 restricted solves per attempted optimization.
The numerical thresholds and no-fallback policy are fixed in METHOD/source.

The independent standard-library checker imports no fitting solver. It
reconstructs the dual weight vector and independently checks primal margins,
dual nonnegativity, intercept/weight stationarity, complementarity, and true
primal-dual objective gap, each against absolute 1e-8. This validates the new
margin optimizer rather than reusing the old ridge normal-equation proof.
Known symmetric/asymmetric optima and altered-certificate rejection passed.
A targeted synthetic check also exercised the negative-target feasible
bound-step/drop branch and certified its result with gap 1.11e-16.

The solver is intentionally incomplete: the centered/unit separable fixture
`x=[(-1,0),(0,1),(0,-1),(1,0)]`, `y=[-1,-1,-1,1]` stops with
UNCERTIFIED_OPTIMIZATION/SINGULAR_RESTRICTED_KKT. The independent checker
certifies its known optimum `w=(2,0), b=-1, alpha=(0,1,1,2)`. Therefore a
singular/limited run must remain a technical inability to certify, not proof
of nonseparability. Only exact opposing-label duplicate coordinates receive
CERTIFIED_INFEASIBLE. This limitation is now a focused regression; no fallback
or real-feature probe was introduced.

## Workflow and admission

G01–G04 each exclude their entire six-row family, including both orders and
all three semantic categories. Each training set has the other 18 semantic
rows plus all eight ordinary rows. Workflow-executed held-row poison tests
leave training means, training scores, and certificates identical for every
fold. The held sets are disjoint and cover exactly the 24 semantic rows.

Each attempt is recorded before optimization. A scientific or technical fold
failure leaves later folds and FULL32 UNRUN. The fifth, full-32 optimization
and artifact publication occur only after four complete passing folds. The
fixed 60+5-second owner and 1 MiB worker payload plus 393,216-byte owner reserve
fit below the 8 MiB total/5 MiB-file ceilings. The owner is byte-identical to
the accepted predecessor; unchanged ownership proofs are reused.

The source bridge imports exact pinned legacy authentication bytes, restores
the new gate module afterward, rejects a bad pin, and never reads the failed
ridge weights. Real extraction remains behind exact root release/contract
authorization and the fixed accepted manifest/feature hashes. The legacy
release operation spelling is explicitly bound to one owned family-development
job with at most five optimizations. Default denial, changed permission,
artifact hash/binding rejection, exact reload, and one-shot retry exclusion
were checked. Scientific selection history remains explicitly development
after the ridge failure, with the lambda-0.001 idea rejected before fitting.

## Verification and exact sources

One complete focused pass from this namespace:

```text
..\..\.venv\Scripts\python.exe -E -S -B test_candidate.py
Ran 9 tests in 0.482s
OK
```

Final verified SHA-256 values:

- gate.py: `b9c439049bde05c8e1139f8afa8b754cce9a65506daed167e7c59e90550d108d`.
- checker.py: `7cf81c280592f51d9d8024d6baca9dec4fdc7e6b607045f07165aed86749c4b9`.
- construction.py: `ae7de40d48c9bb4db0fbf5a4003b22890140f4f104fdfe66489743688fe376d4`.
- source_auth.py: `dd8494e52afa5c992d75b935176c1850074b8db07981ed2794e504fb520ad3cb`.
- test_candidate.py: `f7d64f1d6d70f4e2b99d60f6c0a20bb7e795dc3ce51efbdc576432059ff64f8e`.
- unchanged fit_owner.py: `2d3d3e60ec7ec2c3b2c4ea23cebd698717c6b063275801dfbaa2b62319252939`.

The synthetic integration test substitutes metadata/feature loading with
artificial rows and uses a temporary approval file. It does not constitute
real-source admission or a real approved run. The actual pinned import bridge
was checked separately without invoking feature loading. No real coordinates
were loaded, fit, decomposed, or summarized. NumPy was used only for synthetic
optimization; no model/tokenizer/provider/tensor workload or installation ran.
The reviewer changed no production source, created no commit, and wrote only
this report. Final root source/contract freezing and an explicit committed real
release remain separate prerequisites.

## Late admission delta: retained certificates and bounded CLI

The earlier engineering PASS required this final delta. Root
identified that stage receipts omitted w/b/alpha needed for independent saved
verification, and that printing the complete five-stage RESULT could exceed
the retained owner's 64 KiB stdout cap. The required repair retains each
stage's certificate inputs and emits a compact status/count/RESULT-hash CLI
receipt. Independent verification of the exact repair passed at 17:02 UTC:
each completed stage now retains w/b/alpha, and main's fit branch prints only
status, scientific-pass flag, attempt count, RESULT path and hash. The expanded
`test_default_deny_and_synthetic_artifact_path` was run alone and passed in
0.567 seconds. Its five width-1024 synthetic fits were saved, then all five
KKT certificates and held scores were reconstructed from retained parameters.
The summary hash matched saved RESULT, summary stayed below 4 KiB, and payload
stayed below 1 MiB. The original nine-group suite was not rerun.

Final changed code hashes are construction.py
`38228dbd1def9b8960036c7eb80e4fcad2873d5c72c5f4391153b04fc37e0dc2`
and test_candidate.py
`a4898fd97f97031081331e85f7119452d272c434c35d0584de4bb4edefa17f38`.
These supersede their earlier hashes above; optimizer/certificate/owner
functionality is unchanged. The required code repair is verified. A final
join-only check at 17:02:48 UTC also passed: CORE_SOURCE_LOCK equals the actual
four source hashes, the contract binds the exact input/source bytes, and the
still-false release binds that exact contract. No additional fit/test suite ran.
Final hashes: core lock
`45834a37ec65a03948f856dcd58766861572b553a72274171a4325570a5ede0b`,
contract `09fd14181ed84da80e1b7ebf0aa024afc6baf4b3c38e4df79d6e3e1e33b8ca6b`,
false release `8bc273c4bd40da04196a7f9ee12905e94c67c34e55855a10deea26e8658295c8`.
Observed source-freeze hash:
`ba3bafbd437906814cd82e387930e0e915fa066616f8489015032d90dca4e893`.
Engineering admission PASS is restored for this repaired delta, with the
documented incomplete-solver limitation. No actual release is inferred.
