# One shared nonlinear comply arrow

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH**.
Construction: **COMPLY_CONSTRUCTION_PARTIAL_OR_FAIL**, stop=max_updates, updates=4.
Independent final acceptance 2/4: 0 accepted flips, 2 accepted retentions.
Actual A-to-B 0, B-to-A 0; other-token outcomes 0.
Shared path 0.2; shared net 0.1794676539602933. Forwards 56, derivatives 16.

| Final variant/order | Baseline -> final | Comply margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| v1/preserve_first | B -> B | +0.280288696 | 0.934859236 | 0.295714446 | True |
| v1/preserve_second | B -> B | -0.304944992 | 0.944029205 | 0.422896008 | False |
| v2/preserve_first | B -> B | +0.167366028 | 0.965860974 | 0.125589943 | True |
| v2/preserve_second | B -> B | -0.301370621 | 0.972711285 | 0.268143833 | False |

## Every scored construction stage

| Stage/variant/order | Argmax | Comply margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| 1/v1/preserve_first | B | +0.351779938 | 0.984302046 | 0.002814384 | True |
| 1/v1/preserve_second | B | -1.208904266 | 0.983401749 | 0.012629398 | False |
| 1/v2/preserve_first | B | +0.249639511 | 0.977497924 | 0.002994389 | True |
| 1/v2/preserve_second | B | -1.229516983 | 0.978141864 | 0.012562944 | False |
| 2/v1/preserve_first | B | +0.325250626 | 0.985892387 | 0.012170844 | True |
| 2/v1/preserve_second | B | -0.923206329 | 0.985941748 | 0.048830446 | False |
| 2/v2/preserve_first | B | +0.219985962 | 0.982342581 | 0.010929791 | True |
| 2/v2/preserve_second | B | -0.933124542 | 0.983229781 | 0.047913177 | False |
| 3/v1/preserve_first | B | +0.315023422 | 0.953815640 | 0.175414823 | True |
| 3/v1/preserve_second | B | -0.641246796 | 0.958427764 | 0.242487267 | False |
| 3/v2/preserve_first | B | +0.200946808 | 0.970939974 | 0.080722586 | True |
| 3/v2/preserve_second | B | -0.639278412 | 0.974959645 | 0.153529436 | False |
| 4/v1/preserve_first | B | +0.280288696 | 0.934859236 | 0.295714446 | True |
| 4/v1/preserve_second | B | -0.304944992 | 0.944029205 | 0.422896008 | False |
| 4/v2/preserve_first | B | +0.167366028 | 0.965860974 | 0.125589943 | True |
| 4/v2/preserve_second | B | -0.301370621 | 0.972711285 | 0.268143833 | False |

Independent nonself off identities: 8/8. These test bypass only; there was no always-on collateral test.
Conditional exposed f02 transfer ran: False.


All optimizer KKT, raw-array arithmetic, common-vector/cast/path and deterministic schedule details are in verification.json.
No exact local infeasibility certificate was asserted from optimizer numerical results. Prior linear verdicts are unchanged.
A comply construction pass is not bidirectional control, intrinsic selectivity, broad ordinary-task preservation or permission to train a gate.
f02 is exposed descriptive development only, never construction/selection/repair or sealed confirmation. No retries, extra strength or follow-on.

## Interpretation and locked closeout

The common comply arrow made substantial score progress on the two opposed
requests but did not change any self answer's A/B argmax. Final acceptance
2/4 consists solely of two retentions; neither opposed request was accepted.
All four updates were used. The last endpoint is preserved, not an earlier
selected checkpoint. No next experiment is started.

All 56 rows passed quality and weight-integrity checks. Across the scored
edits/finals, minimum pair mass was 0.9348592360613391 and maximum raw KL
was 0.42289600791439347; no KL upper cap was introduced. Unlike the prior
0.20 fixed-arrow attempt's format failures, this run retained valid answer
mass. That descriptive difference is not proof of broad task preservation.

| Update | Uncapped minimum-norm increment | Cap factor | Shared step norm | Shared path / net | KKT primal / stationarity / complementarity / signed gap |
|---|---:|---:|---:|---:|---:|
| 1 | 0.23212788940804158 | 0.2153985035038528 | 0.05 | 0.05 / 0.05 | 1.8474861223823815e-16 / 6.569350421794569e-18 / 8.34296824459262e-18 / 1.6770162663936345e-18 |
| 2 | 0.21016084748945094 | 0.23791301090232692 | 0.05 | 0.1 / 0.09761678610027388 | 0 / 6.298026411014741e-18 / 1.218392159532157e-17 / 1.7739827577304113e-17 |
| 3 | 0.1707222934020847 | 0.2928732914935728 | 0.05000000000000001 | 0.15000000000000002 / 0.14077849149212576 | 0 / 3.3970877123273142e-18 / 7.773270799654223e-18 / 1.4621577630594074e-17 |
| 4 | 0.11384620444743025 | 0.4391889939825623 | 0.05 | 0.2 / 0.1794676539602933 | 2.0133486375839863e-16 / 2.169631996812921e-18 / 2.1921391383962106e-18 / -1.0729694388586638e-18 |

Each increment passed independent 80-digit, scale-aware KKT reconstruction.
The capped optimizer prediction was not treated as actual acceptance.
Maximum actual normalized path was 0.20000000085479236 and net
was 0.17946765356360647; all per-prompt physical bounds passed
with the fixed absolute1e-6 rounding allowance. All16 gradient forwards and
four final replays matched their corresponding ordinary scored state:
maximum checked logit difference0; nonfinal-coordinate difference0.
Maximum actual-versus-intended cast component error was
1.4901161193847656e-08. The common vector and original h0 norms were checked
at every applied cell; no order- or prompt-specific arrow was used.

The independent probability/mass/KL audit used absolute2e-5 and zero-relative
tolerance, with exact direct margins, deltaS, argmax and labels. Maximum
mass error was5.551115123125783e-16 and KL error9.71445146547012e-16.
All8 separate off identities passed. These are bypass controls only, not
always-on collateral measurements or an implemented gate.

Exactly56 forwards and16 derivatives completed in232.3279999999795seconds
including loading. Only the four conditional f02 cells were skipped, each
with final-not-all-accepted reason. No f02 numeric forwards, transfer-vector
freeze, retry, padding, smoke or generation occurred.

Protocol commit `dcb0dc4`; source commit
`2258a856a0ef38ded4f01995dafdd10f784badda`; preregistration-only commit
`1808084`. All preceded real tokenizer/model loading. Twenty-four focused
fake-model tests and Ruff passed. Standard usage was22% immediately before
the run, audit and closeout checks. No old full audit/full suite or subagents.

Preserved56 compressed raw float32 arrays, totaling51321818bytes,
plus rows, gradients, optimizer proposals, endpoint, raw journals, skips,
runtime and verification. `CHECKSUMS.json` inventories every other artifact;
local `.gitattributes` preserves raw console bytes.

Rows SHA256:
`11afb4f72ad321e88a45ccbfd15da8a78697c19d07bf29f15e0d5f825bad11f6`.
Optimizer journal SHA256:
`ab9f20bda3b8b47ca3d6bebeb7128da7edfb544363f63610cb5902af26b2eb05`.
Endpoint SHA256:
`0636eaa473e1a88b26e80ca012b879948e40c4e92cfee1b96d21caa5d7cd015c`.
Final common vector float64-LE SHA256:
`4488d0804915a4b2d4739d434a1555d4ae64045348cfe4a9d62760ba440b4b30`.
Independent verification SHA256:
`c72c5e0f5aba13e51fad595a3747de983ecebf4c2604bfdd5cff2b2d2f095c09`.
Preregistration SHA256:
`d5eb6bfdaff4850a7981f6ac9a93da69f679c32efce29332c38762b10cadc6c7`.

This is a bounded partial construction result, not proof that a neural
intervention is impossible. Historical linear certificates and real-model
failures are unchanged. No preserve-arrow training, selector, gate/controller,
strength escalation, follow-on run, push, credits/reset or assistant-model
setting change. Unrelated user-owned changes are preserved.
