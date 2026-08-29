# Counterfactual Tangent Shielding v1: construction no-go

Status: `construction_no_go`.

The locked v1 construction produced 88 direction records and zero eligible directions. The frozen calibration plan therefore contained 72 baseline forwards and zero intervention forwards. No pilot is authorized.

## What was validated

- Capture: 136 forward plus 136 backward evaluations, manifest `d9994f84c49cddb1c33a075e3d60047c83e49450034be9cfd2ac8a793cb2b33e`.
- Construction: 88 records, 0 eligible.
- Frozen finite plan: 72 baselines and 0 changed rows.
- Completed checkpoints: 72 baseline rows; the reporter replayed zero model forwards.
- Capture-to-finite boundary: 64/64 rows matched exactly; maximum absolute difference `0.0` under tolerance `5e-05`.
- Unrelated-control baseline: 7/8 correct, 0 invalidly formatted. The one incorrect result was the reverse-order instruction-control view.

## Reporting edge case

After the valid baseline-only checkpoints were complete, the locked runner's vector-reuse audit raised `RuntimeError: CTS did not reuse one byte-identical perturbation per signed unit` because there were no changed rows and therefore no signed-vector groups. This report fixes only that zero-intervention reporting path; it does not change the locked runner, protocol, directions, checkpoints, or scientific outcome.

## Hashes

| Artifact | SHA-256 |
|---|---|
| Result JSON | `aabb7f2eb4ab707c53e0ad2d17327cb8d8c3bc198f12fb8e7b8f07efd71b69d8` |
| Result identity | `31fa72d23e1867e27d9116545ca46f64508a0ec3a2ce39f5289a02288e9853a2` |
| v1 lock | `b9cc90727483554e6e928e14d4337a56bb878693ccc54cd41177d09f8cac9bab` |
| Capture manifest | `4fc63b05d9ad6929f3b036cc481fd131915bda3fe2277c56286ebf089c8595b9` |
| Direction manifest | `1e70d48f0c73699f55f58d1e0fb100951e679ff8fba226e0c7f90d732fb4a8f0` |
| Direction tensor | `bf1afc271585cb04f9d983ba6115e1ce28add2afc678ff369000c2941f2a3b51` |
| Calibration freeze | `29a16970938cf96f56d558664fc8e048c14be0c53603ca3df51fc6772d413464` |
| Calibration ledger | `cb20c118fe10210165db70074a5ddb5fe232f95db35ccebaa94ea2b09884542e` |
| Calibration checkpoint 0 | `d16fa056b33e4bbc59c4bf816bb46e5fe31384394cac0c08a02c174eaaeb5903` |
| Calibration checkpoint 1 | `548d11d1f2fe3c83505894e9cf2ed80423c96daa9701ffbb53e02993067e55c8` |
| Calibration checkpoint 2 | `cd60192a0897596e49851e540273afb86cc45797c6902a873242a36e632d7216` |

## Claim boundary

This is an opened-development construction no-go under one locked layer, prompt family, margin, nuisance grid, and L2 cap. It neither demonstrates self-preservation steering nor proves that no direction could exist under different constraints. No intervention effect or decision change was measured.
