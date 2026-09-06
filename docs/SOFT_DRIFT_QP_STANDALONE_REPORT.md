# Standalone soft-drift QP implementation: bounded checks passed

**Implementation substep complete; STOP for review.** Parent design commit: `a72de9aaf16a5e7bda39e49e3bba01718807b1cd`. The standalone deterministic solver and independent numerical checker passed 38 targeted fake/math tests and one fixed synthetic 1024-dimensional, twelve-constraint timing-size smoke. This is not a model experiment, causal result, behavioral candidate, scientific lock, runtime integration or full-pipeline readiness certificate.

## Delivered scope

- [Prospective numerical policy](SOFT_DRIFT_QP_NUMERIC_POLICY.md), fixed before test execution: dimension-scaled tolerance, stricter interior acceptance, separate conservative rank guards and exact nonnegative duals.
- [Standalone solver](../scripts/soft_drift_qp_solver.py): fixed rational target kappa=1/24 evaluated in binary64, affine baseline-relative drift, signed RHS, nonzero empty-mask d0, ascending support order, at most 4096 masks and a compact trace. It returns only a numerical estimate requiring the independent checker. It does not apply clipping or projection.
- [Independent checker](../scripts/verify_soft_drift_qp.py): direct 80-digit outward-interval reconstruction, rational kappa, independent fingerprint/rank/KKT checks, no solver import or reuse of its small-system computations. Rejected, marginal or near-dependent cases are numerically unresolved, not proven infeasible.
- [Targeted tests](../tests/test_soft_drift_qp.py): hand-supplied negative-RHS and affine optima; duplicate and near-opposed rows; nonzero d0; zero-penalty algebra only; own-norm letter/semantic signs; deterministic masks; corrupted primal, dual, stationarity, complementarity, mask and fingerprint; malformed/nonfinite inputs; exact dual sign; conservative tolerance boundaries; fail-closed mask accounting; no ML modules.
- [One-shot smoke source](../scripts/soft_drift_qp_native_smoke.py) and [saved compact result](soft_drift_qp_native_smoke.json).

Independent source review found and fixed two implementation issues before numerical execution: the checker omitted its required `accepted` flag, and exceptions could omit the currently attempted mask from the diagnostic count. Final independent source/artifact review found no remaining concrete blocker. No numerical tolerance, input fixture, penalty or model-outcome criterion was changed after testing.

## Execution ledger

Every listed command ran with an external subprocess timeout. Fresh account usage was available at 57% before each batch; no credits or reset were used. Automatic pytest plugin loading was disabled.

| Command | Result | Measured subprocess seconds | Hard cap seconds |
|---|---|---:|---:|
| Scoped Ruff format | Four new Python files formatted | 0.742892 | 10 |
| Scoped Ruff check, initial | Two style findings, subsequently fixed | 0.087393 | 10 |
| Scoped Ruff check, final | Passed | 0.101770 | 5 |
| Default-Python pytest launch | Missing pytest; stopped before collection | 0.036951 | 35 |
| Scoped Ruff format check | Passed | 0.090374 | 5 |
| Existing repository-venv targeted pytest | **38 passed**, one cache warning | 0.529932 | 25 |
| Fixed native-shaped smoke | **Passed**, one attempt only | 2.239393 | 30 |

Other-command measured total: **1.589312 seconds**, conservatively charged at the sum of their hard caps, **90/90 seconds**. Including the smoke: **3.828705 measured subprocess seconds**, **120/120 seconds** conservatively charged. The initial first batch stopped on lint, so its planned pytest command was never executed or charged; the separately listed default-Python launch did execute and is charged in full. No further numerical checks or native retries occurred.

Pytest reported `38 passed, 1 warning in 0.15s`. The warning was a denied write to the existing pytest cache, not a failed test; the cache was not repaired. The existing virtual environment supplied pytest without installing dependencies. Formatting/style fixes changed only new scoped files.

## One fixed synthetic smoke

The prospectively declared dense Walsh fixture used exactly 1024 coordinates and twelve rows, with `b_i=(20+i)/200` and `c_p=(-1)^p*(p+1)/100`. No seed, alternative instance, sweep or tuning was used. The outer 30-second timeout included process launch, imports, construction, solve, independent verification, serialization and persistence. The script's internal 2.098922-second timer excludes imports and persistence; the **2.239393-second external measurement** is the complete reported timing.

| Recorded quantity | Result |
|---|---:|
| Masks visited | 4096 |
| Mask outcomes | 4095 primal-screen rejections; one estimate |
| Selected mask | 4095, all twelve rows |
| Independent status | NUMERIC_KKT_WITHIN_TOLERANCE |
| Maximum normalized KKT upper bound | 3.716442e-17 |
| Interior acceptance threshold | 3.865352e-12 |
| Complete in-memory solver JSON | 27,862 / 131,072 bytes |
| Complete in-memory checker JSON | 5,973 / 32,768 bytes |
| Persisted smoke summary | 9,390 / 16,384 bytes |

Saved smoke SHA256: `eae53193fe554e08d6122bb1db346848ae34272076b9539a8d6951587583db5f`. Its five bound policy/source/test hashes were checked against the final working files. The persisted artifact retains input/solution/trace hashes and compact diagnostics, not native vectors per mask.

## Limits and unchanged research status

This numerical certificate does **not** establish exact-real feasibility: the saved point has tiny signed slack errors admitted only under the prospectively fixed tolerance. The conservative rank guards may reject solvable instances; lack of a certificate remains unresolved. The checker accepts only finite binary64 inputs and independently supported dual masks. It does not validate real derivatives or a future recorded-data assembly pipeline.

The raw synthetic proposal norm is about 0.445702; no applied geometry was performed or certified. Raw-QP feasibility is distinct from feasibility after clipping/projection. One well-conditioned synthetic instance does not certify runtime on model-derived rows, ill-conditioned instances, a full audit, or any model's behavior.

Zero ML/model/tokenizer imports or loads, zero real forwards and derivatives, no runtime integration, old-file edits, historical evidence changes, f04/new data, gate/controller/LoRA, credit/reset use or push. Unrelated untracked files remain untouched. The earlier real attempt remains INCONCLUSIVE / ARTIFACT_BYTE_CAP, and the retrospective zero-flip result is unchanged. Publication remains 40%.

Complete compact audit-size certification through actual finalization, runtime/call accounting, integration and any new real-model run remain separately unfulfilled and unauthorized. This source/tests/report-only substep stops here for review.
