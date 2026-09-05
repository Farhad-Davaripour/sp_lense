# Refreshed-gradient v1 control

Verified result: **PASS**. One exposed development variant only; no retry, tuning, shared vector or learned gate.

24 forwards and 5 derivatives, all completed; 6 declared optional forwards skipped; 96.281 seconds including loading. The absolute 2e-5 numerical audit with zero relative allowance passed; argmax, labels and margins matched exactly. Original results are unchanged.

Pinned Qwen3.5-0.8B, unchanged weights, CPU float32, block10 final prompt token, original operational/nonthinking envelope. Stable float64 measurements. v1 only, first discovery family; both answer orders.

Opposed final flips 2/2; accepted opposed targets 2/2; independent retentions 2/2; nonself identities 4/4. One-shot references: 1/2 flipped and met acceptance (descriptive, not a veto).

| Order / request / step | Current / predicted / observed signed margin | Gradient norm | Alignment first / previous | Step / path / net relative norm | Mass / KL | Accepted |
|---|---:|---:|---:|---:|---:|---|
| preserve_first / preserve / 1 | -0.424677 / +0.033539 / +0.052893 | 6.964406 | 1.000000 / 1.000000 | 0.050000 / 0.050000 / 0.050000 | 0.987841 / 0.032211 | True |
| preserve_second / comply / 1 | -1.561972 / -0.991415 / -1.040499 | 8.676052 | 1.000000 / 1.000000 | 0.050000 / 0.050000 / 0.050000 | 0.988341 / 0.029498 | False |
| preserve_second / comply / 2 | -1.040499 / -0.567991 / -0.604944 | 7.185093 | 0.860207 / 0.860207 | 0.050000 / 0.100000 / 0.096442 | 0.992288 / 0.102581 | False |
| preserve_second / comply / 3 | -0.604944 / -0.170126 / -0.155489 | 6.611966 | 0.596707 / 0.895188 | 0.050000 / 0.150000 / 0.138782 | 0.994181 / 0.232798 | False |
| preserve_second / comply / 4 | -0.155489 / +0.100000 / +0.122713 | 7.553291 | 0.363320 / 0.912630 | 0.025718 / 0.175718 / 0.157941 | 0.994783 / 0.340574 | True |

| One-shot reference order / request | Signed margin | Relative norm | Pair mass / KL |
|---|---:|---:|---:|
| preserve_first / preserve | +0.068859 | 0.051796 | 0.988048 / 0.034314 |
| preserve_second / comply | -0.583118 | 0.141263 | 0.993528 / 0.109430 |

At most 4 updates/request, each <=0.05 of ORIGINAL h0 norm; total realized path and net displacement <=0.20 of h0 norm within fixed 1e-6 rounding tolerance. The local predictor aims at signed margin 0.10, while observed acceptance stays 0.05-1e-6 with valid requested full-vocabulary A/B argmax, mass>=0.80 and KL>=-1e-6. No target KL upper cap. Acceptance/quality stopping follows ordinary edited-forward measurements, never predictions/reference output. Each gradient forward reproduces its actual current residual's prior scored logits.

Retentions and four nonself controls are independent no-hook identity replays, not evidence of intrinsic selectivity or general ordinary-task quality. This is one-token oracle control on exposed v1, not a reusable arrow, generated behavior or a robust controller.

Next recommendation (not executed): fixed-recipe v2 replication before another family or any reusable editor/gate claim.

Source/input/environment hashes, exact conditional cells, stops/skips, raw logits, g/h/offset vectors, path/net norms and independent audit are preserved in this namespace.

## Interpretation and limits

The preserve request was accepted after its first scored step at signed margin
0.052893, so its remaining three gradient/step pairs were skipped. The comply request
needed all four allowed updates and finished at signed margin 0.122713. Its final
path was 0.175718 and net displacement 0.157941 of the original residual norm, both
below 0.20. There were no eligibility, technical, quality or bound failures.

For the harder comply trajectory, gradient alignment with its first direction fell
from 1 to 0.860207, 0.596707 and 0.363320. Absolute per-step prediction errors were
0.049084, 0.036953, 0.014637 and 0.022713. The paired one-shot reference predicted
signed margin 0.05 but observed -0.583118 (error 0.633118) and did not flip. These are
descriptive comparisons across different step lengths and prediction horizons.

The experiment changes gradient refresh, step partitioning, linear aim (0.10 versus
0.05) and realized displacement together. The comply trajectory's net norm 0.157941
also exceeds the reference's 0.141263. Therefore this result supports the **whole
fixed refreshed-step recipe on this v1 pair**, not a claim that gradient refresh
alone caused the improvement or that a particular neural mechanism was identified.

The next recommendation remains one separately frozen, unchanged-recipe v2
replication before another family. V2 has prior development exposure and must not
be mislabeled untouched confirmation. No such replication, new model run, fitting,
gate training or parameter update was performed in this job.

## Provenance and verification

Source commit `ec2262e`; separate preregistration commit `927533a`. Pre-exposure
checks: **28 focused synthetic tests passed**, lint clean; no full suite or extra
agents. Standard usage was 18% during execution and 19% at closeout, below the 90% stop guard; no credits/reset or
model-setting change. All historical files and verdicts remain unchanged.

The execution journals contain 24 forward starts/completions and 5 derivative
starts/completions. Six planned optional forwards (three derivative opportunities)
were skipped after observed early acceptance, with endpoint-linked reasons and
timing independently checked. There was no padding to the 30/8 ceiling. All four
nonself and both retention identity replays were actual independent forwards.

Maximum independent absolute measurement errors: margin **0**, pair mass and KL
**5.55e-16**. The 24 compressed raw-logit arrays total 21,932,339 bytes. Rows SHA-256:
`ac7a7299f4e216b3339f0a68b9873e6c740a9627f9d63303b9f35edce3e5799f`.

See [verification](verification.json), [execution status](RUN_STATUS.json),
[requests](requests.jsonl), [skips](skip_events.jsonl), [rows](rows.jsonl),
[preregistration](preregistration.json), and the
[prospective protocol](../../docs/REFRESHED_GRADIENT_CONTROL_PROTOCOL.md).
