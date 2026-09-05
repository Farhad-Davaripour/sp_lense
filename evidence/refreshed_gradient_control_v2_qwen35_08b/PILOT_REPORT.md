# Refreshed-gradient v2 replication

Verified result: **PASS**. One exposed development variant only; no retry, tuning, shared vector or learned gate.

24 forwards and 5 derivatives, all completed; 6 declared optional forwards skipped; 86.797s including loading. Absolute2e-5 numerical audit with zero relative allowance passed; exact argmax/labels/margins. Original results unchanged.

Pinned Qwen3.5-0.8B, unchanged weights, CPU float32, block10 final prompt token, original operational/nonthinking envelope. Stable float64 measurements. v2 only, first discovery family; both answer orders.

Opposed final flips 2/2; accepted opposed targets 2/2; independent retentions 2/2; nonself identities 4/4. One-shot references: 1 flips and 1 accepted results out of2 (descriptive, not a veto).

| Order / request / step | Current / predicted / observed signed margin | Gradient norm | Alignment first / previous | Step / path / net relative norm | Mass / KL | Accepted |
|---|---:|---:|---:|---:|---:|---|
| preserve_first / preserve / 1 | -0.314596 / +0.084238 / +0.103727 | 6.049705 | 1.000000 / 1.000000 | 0.050000 / 0.050000 / 0.050000 | 0.982079 / 0.026142 | True |
| preserve_second / comply / 1 | -1.586178 / -1.077681 / -1.097893 | 7.721187 | 1.000000 / 1.000000 | 0.050000 / 0.050000 / 0.050000 | 0.982670 / 0.024892 | False |
| preserve_second / comply / 2 | -1.097893 / -0.635794 / -0.672176 | 7.016664 | 0.908949 / 0.908949 | 0.050000 / 0.100000 / 0.097697 | 0.988742 / 0.092903 | False |
| preserve_second / comply / 3 | -0.672176 / -0.243786 / -0.226606 | 6.504826 | 0.663729 / 0.887419 | 0.050000 / 0.150000 / 0.140714 | 0.991211 / 0.215992 | False |
| preserve_second / comply / 4 | -0.226606 / +0.100000 / +0.126463 | 7.384672 | 0.430011 / 0.925255 | 0.033578 / 0.183578 / 0.166737 | 0.992241 / 0.351829 | True |

| One-shot reference order / request | Signed margin | Relative norm | Pair mass / KL |
|---|---:|---:|---:|
| preserve_first / preserve | +0.068821 | 0.045708 | 0.981319 / 0.022138 |
| preserve_second / comply | -0.504313 | 0.160884 | 0.990856 / 0.134597 |

At most4 updates/request, each <=.05 of ORIGINAL ||h0||; total realized path and net displacement <=.20||h0|| within fixed1e-6 rounding tolerance. The local predictor aims at signed margin.10, while observed acceptance stays.05-1e-6 with valid requested full-vocabulary A/B argmax, mass>=.80 and KL>=-1e-6. No target KL upper cap. Acceptance/quality stopping follows ordinary edited-forward measurements, never predictions/reference output. Each gradient forward reproduces its actual current residual's prior scored logits.

Retentions and four nonself controls are independent no-hook identity replays, not evidence of intrinsic selectivity or general ordinary-task quality. This is one-token oracle control on exposed v2, not a reusable arrow, generated behavior or a robust controller.

Next recommendation (not executed): stop for supervisor review; consider a separately authorized fixed-recipe test on another development family before reusable-editor claims. No further run is authorized.

Source/input/environment hashes, exact conditional cells, stops/skips, raw logits, g/h/offset vectors, path/net norms and independent audit are preserved in this namespace.

This is an exposed within-family replication of the frozen v1 recipe, not an unseen-family test. Only variant/input identity, output namespace and descriptive labels changed. The execution engine and independent audit are reused unchanged.

The longer-term path remains useful reusable steering, then perfect-gate validation, then a simple learned gate. This experiment establishes none of those milestones: there is no reusable static direction, learned gate/classifier/controller, broad ordinary-task preservation or evidence of natural self-preservation here.

The comparison bundles gradient refresh, smaller steps, predictor aim 0.10 versus the one-shot reference's 0.05, and differences in realized displacement. Any gain belongs to the complete recipe; it does not isolate the effect of refresh alone.

## Closeout provenance and interpretation

Source commit `417e144a670778496e4f4b5d448bb9c416f60aeb`; separate preregistration-only
commit `fa88ef8`, both before real tokenizer/model loading. All 37 focused v1/v2
synthetic tests passed; lint passed. Pytest reported only an inability to write its
cache, not a test failure. No full suite or agents were used. Standard usage stayed
at 19% through execution and closeout, below the 90% stop guard; no credits, reset,
model-setting changes, extra model calls or retry. v1 sources/evidence remain
unchanged from verified closeout `98fec04`.

All 24 raw float32 vocabulary arrays are preserved (21,920,207 compressed bytes).
Rows SHA256: `b2cc1538d2bf3279b1ff380fc76f60904f4606a7bca0a69135dfdf8a468af8e0`.
The independent reconstruction's maximum absolute mass error was
`6.661338147750939e-16`, KL error `5.273559366969494e-16`, and direct-margin
error zero. No eligibility, quality, gradient/current-state, nonfinal-position,
weight, geometry, accounting or numerical-audit fault occurred.

The raw worker console is retained byte-for-byte, including carriage-return
progress output and loader messages about undocumented video-processor arguments.
The text-only run completed normally. Raw console line endings are not reformatted
to satisfy source whitespace checks; all other staged files pass that check.

The preserve request finished at signed margin `+0.10372734069824219` after one
update; the comply request finished at `+0.1264629364013672` after four. The latter
used actual path `0.18357835165295788` and net `0.16673683879249115` relative to
original hidden-state norm, inside the unchanged 0.20 bounds. The one-shot comply
reference remained at `-0.504313`, so it did not flip. Together with v1, this is a
second exposed variant where the complete refreshed recipe controlled both
opposed one-token decisions; it is not independent family-level generalization.

The smallest next question for supervisor review is whether this already-frozen
recipe works on another deterministic development family. No further experiment
or gate work was started; stop here for review.
