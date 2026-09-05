# One fresh projected shared comply arrow

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH**.
Construction: **COMPLY_CONSTRUCTION_ACCEPTED_ONLY**, stop=accepted, updates=5.
Independent final acceptance 4/4: 2 accepted flips, 2 accepted retentions.
Actual A-to-B 0, B-to-A 2; other-token outcomes 0.
Shared path 0.24092303687711125; shared net 0.2. Forwards 68, derivatives 20.

| Final variant/order | Baseline -> final | Comply margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| v1/preserve_first | B -> B | +0.217292786 | 0.983593347 | 0.045788949 | True |
| v1/preserve_second | B -> A | +0.105354309 | 0.985497360 | 0.353498979 | True |
| v2/preserve_first | B -> B | +0.110414505 | 0.987630320 | 0.030759017 | True |
| v2/preserve_second | B -> A | +0.102067947 | 0.988546432 | 0.347032217 | True |

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
| 5/v1/preserve_first | B | +0.217292786 | 0.983593347 | 0.045788949 | True |
| 5/v1/preserve_second | A | +0.105354309 | 0.985497360 | 0.353498979 | True |
| 5/v2/preserve_first | B | +0.110414505 | 0.987630320 | 0.030759017 | True |
| 5/v2/preserve_second | A | +0.102067947 | 0.988546432 | 0.347032217 | True |

Independent nonself off identities: 8/8. These test bypass only; there was no always-on collateral test.
Conditional exposed f02 transfer ran: True.

f02 preserve_first: margin=0.30139732360839844, mass=0.9879382737968802, KL=0.03452944261012835, accepted=True.
f02 preserve_second: margin=0.02033233642578125, mass=0.9889911997647682, KL=0.3044274360746521, accepted=False.

All optimizer KKT, raw-array arithmetic, common-vector/cast/path and deterministic schedule details are in verification.json.
No exact local infeasibility certificate was asserted from optimizer numerical results. Prior linear verdicts are unchanged.
A comply construction pass is not bidirectional control, intrinsic selectivity, broad ordinary-task preservation or permission to train a gate.
f02 is exposed descriptive development only, never construction/selection/repair or sealed confirmation. No retries, extra strength or follow-on.

## Projected updates

Attempted rounds: 5/8; applied/scored: 5.
Actual path limit .40, instantaneous/net limit .20, per-step limit .05.

| Stage | d norm | s norm (proposed) | r norm (actual) | Path | Net | Projection factor |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.232127889408 | 0.05 | 0.05 | 0.05 | 0.05 | 1 |
| 2 | 0.210160847489 | 0.05 | 0.05 | 0.1 | 0.0976167861003 | 1 |
| 3 | 0.170722293402 | 0.05 | 0.05 | 0.15 | 0.140778491492 | 1 |
| 4 | 0.113846204447 | 0.05 | 0.05 | 0.2 | 0.17946765396 | 1 |
| 5 | 0.0492152228718 | 0.0492152228718 | 0.0409230368771 | 0.240923036877 | 0.2 | 0.941203128253 |

Signed projection loss = predicted proposed gain minus actual gain;
negative values are preserved. Prompt order: v1/first, v1/second, v2/first, v2/second.

| Stage | Signed projection loss (four prompts) | Proposed comply margins | Actual-increment predicted margins |
|---|---|---|---|
| 1 | +0; +0; +0; +0 | +0.367698004223; -1.18875054646; +0.268372480948; -1.22297764558 | +0.367698004223; -1.18875054646; +0.268372480948; -1.22297764558 |
| 2 | +0; +0; +6.93889390391e-18; +0 | +0.312749349; -0.888902507887; +0.214038324471; -0.913207594553 | +0.312749349; -0.888902507887; +0.214038324471; -0.913207594553 |
| 3 | +0; +0; +0; +0 | +0.296660669976; -0.623536523793; +0.184845278315; -0.630549957029 | +0.296660669976; -0.623536523793; +0.184845278315; -0.630549957029 |
| 4 | +0; +0; +0; +0 | +0.273103173078; -0.315699361178; +0.156612080871; -0.314595469885 | +0.273103173078; -0.315699361178; +0.156612080871; -0.314595469885 |
| 5 | -0.00385002315564; +0.0663712145587; -0.00456341762451; +0.0721480657 | +0.208635472809; +0.1; +0.1; +0.1 | +0.212485495965; +0.0336287854413; +0.104563417625; +0.0278519343 |

Projection, iteration count and path allowance changed together. This run cannot isolate which change caused any difference.
This is a fresh from-zero attempt, not a resumed trajectory or a replacement of any prior verdict.

## Closeout interpretation and provenance

A single final common COMPLY vector satisfied all four f01/v1-v2 construction
prompts across both answer orders: two B-to-A opposed flips and two B-to-B
comply retentions. This is a bounded construction success, not a broad or
bidirectional steering result. The first all-four accepted actual endpoint
was step 5; steps 6-8 were durably skipped (24 cells, no extra gradients).

At step 5 the proposed step norm was 0.04921522287175834 (below the .05 cap),
and the unprojected endpoint norm was 0.21249398136954953. Projection applied
factor 0.9412031282532132, producing actual step norm 0.040923036877111255
and endpoint norm .20. Linearized actual-increment predictions for the
opposed prompts were only .03362878544127862 and .027851934299953973;
the measured margins were .10535430908203125 and .10206794738769531.
Those predictions are diagnostics, not the acceptance test.
Tiny signed loss at stage 2 is binary64 subtraction rounding, with projection
factor 1 and distance 0; no loss was clamped.

### Exposed f02 transfer, separate from construction

The frozen common vector was hashed after the final replays and controls,
before either f02 numeric baseline. There was no f02 fitting or repair.

| f02 order | Baseline to edited argmax | Comply margin | Strict acceptance | Interpretation |
|---|---|---:|---|---|
| preserve_first | B to B | +0.30139732360839844 | Pass | Accepted comply retention |
| preserve_second | B to A | +0.02033233642578125 | Fail | Actual comply flip, but below .05 margin |

Thus exposed transfer is 1/2 accepted, with one additional actual flip below
threshold; do not count that as an accepted flip or a transfer pass.
No f02 quality failure occurred. Nonself off 8/8 demonstrates bypass only;
always-on nonself collateral was not tested. No gate/controller was trained.
Projection, iterations and path budget changed together, so this single run
does not identify which change caused improvement over the previous run.

### Verification and bounded resources

- One fresh CPU float32 process; pinned revision and all source/input hashes
  matched. No retries, smoke, generation, extra model or model-setting changes.
- 68/92 forwards, 20/32 derivative attempts, 24 skips,
  279.1099999998696 seconds including load; external limit 900 seconds.
- 68 raw float32 arrays, 62335331 compressed bytes.
- 68/68 parameter integrity checks passed; 0 finite quality failures;
  nonfinal state difference 0; current/final/off hidden and logit differences 0.
- Maximum actual normalized path 0.2409230378942395, net 0.20000000096967108;
  the tiny cast-rounding excess over .20 is inside the unchanged absolute
  1e-6 physical-coordinate allowance. Shared binary64 endpoint norm is exactly .20.
- Maximum cast/delta component error 1.4901161193847656e-8.
- Exact margin/argmax/label reconstruction. Maximum mass error
  5.551115123125783e-16; maximum raw-KL error
  9.71445146547012e-16, inside absolute 2e-5 with zero relative tolerance.
- All five independent 80-digit scale-aware optimizer KKT checks passed.
  The independent optimizer check did not use or revise earlier radius verdicts.
- 36 focused tests passed; scoped Ruff passed. Standard usage 23% at run,
  audit and closeout checks. No credits, resets or pushes.

Protocol commit: c78887ea17dac7fdf31daddb7a259b9641c0237a.
Frozen source commit: a8cb50bfa266bae52b4d52dfce50e7418d24c488.
Preregistration-only commit: b4955bd580b9e1e63654b974e33ab9a1db24afbf.
Predecessor evidence commit remains ed98bb55f25387941e3541620d3807954ade5dee.
Shared endpoint/transfer vector float64-LE SHA256:
bcb173119612b1942a13c133c7e204afa9e62ba3f4ef970dd72df112053775e9.

CHECKSUMS.json covers every artifact in this namespace except itself.
The evidence commit records this checked closeout; no further experiment or
budget extension is part of it. All earlier tracked evidence/source and
unrelated user-owned files remain unchanged.
