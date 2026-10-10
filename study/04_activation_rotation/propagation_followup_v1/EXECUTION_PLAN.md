# Prepared matched propagation panel — no GPU allocated

This replaces the next proposed 12-prompt broad batch, which remains frozen and
unexecuted. It does not authorize allocation as part of today's planning task.

## Frozen design

PANEL.json contains three independently worded completed/pending matched pairs,
six prompts and their exact tokenizer IDs. Prefix lengths are respectively
109/109, 103/103 and 102/102. No padding was added. The same 28-token neutral
continuation is forced in all six prompts. The 64-token ceiling is retained.
The tokenizer reproduced an archived native chat prefix exactly, using the
pinned snapshot and Transformers 5.19.0, without loading weights.

Use only frozen Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, unquantized BF16, SDPA, no adapter.
Use one loaded model, batch size one. Six conditions per prompt: base, existing
+v, -v, original random direction, independent random direction, causal norm-only.
There are 36 study traces and two separate eight-step null/untreated validation
probes. Rotate condition order over all six positions and alternate pair status
order as explicitly listed. No new baseline generation or behavior evaluation.

Injection is zero-based block 10, persistently at every nonpadding prefill
position and each cached decode step. The primary response is the mean paired
angle at zero-based block 15 across all 28 valid prediction positions, summarized
once per prompt. The primary contrast is pending minus completed for +v.
Support requires the contrast to be positive in all three pairs and exceed both
random and norm-only pair contrasts in each pair. Otherwise stop the semantic
interpretation. -v is a sign/axis check, not automatic semantic evidence. These
three pairs/two random directions do not establish population significance.

## Causal native matching

For the current prehook residual h, compute hypothetical native BF16
h_plus = BF16(h + 0.5 * BF16(v)). Its measured displacement norm is the target,
separately at every position; +v uses that exact operation. For -v/random arms,
choose the nearest native displacement using the fixed 17 coefficient values
0.5*(1+k/4096), k=-8..8, with first/smallest tie breaking. This is a declared
numerical rounding match within 0.2% of nominal strength, not a strength sweep
selected on model outcomes. Stop if residual norm mismatch exceeds 0.5%; do not
expand the grid. Record residual mismatches and actual delivered norms.

Norm-only instead sets h_norm = BF16(h * ||h_plus||/||h||), using only the current
h and frozen v. This matches the hypothetical +v local magnitude, not its
displacement or entire downstream trajectory. Stop if native relative norm
error exceeds 0.5% or site angle exceeds 0.005 rad. Reject near-zero vectors.
BF16 rounding can introduce a small site angle, so do not claim exact zero.
No future residual or held-out downstream result enters any scalar choice.

The prepared collector records pre-injection states and full-width posthook
outputs. Verify exact pre-injection/upstream equality across all conditions;
this tests the assumption that persistent block-10 forcing does not change
earlier-layer cached computations. Record full-position matching-error maxima,
last-position target/actual displacement, coefficient bounds and site angles.
Norm-only displacement is intentionally not matched to additive displacement.

## Admission, run and preservation

Freshly verify credits, actual GPU rate, authentication and pinned files for a
future authorized session. Prefer a healthy owned session/snapshot if available;
otherwise restore the exact public checkpoint into local /content storage.
Do not touch Research 3's active runtime, 27B model or distillation work.

Before setup/loading, start a process-bound 40-minute worker limit and a
45-minute total runtime limit, including a five-minute export reserve. Do not
purchase credits, increase the cap, silently retry, or leave billed idle compute.
Verify source/input FREEZE hashes and call the existing collect.verify_checkpoint
before loading. Load the proven frozen BF16 model/tokenizer once and disable
all gradients. The panel collector itself allocates/loads nothing.

Call collect_panel(model, tokenizer, vector_archive, new_output_directory,
session_started_monotonic=setup_start, model_identity=verified_receipt,
checkpoint=verified_export_callback). Use the private panel_vectors.npz with
the SHA recorded in PANEL.json. Supply a callback that synchronously copies
completed output files to private Drive and hash-verifies them; it is invoked
after each prompt and on success/fault. Output directories must be new.

The native six-condition collector is prepared but not model-exercised. Its
position-local matching and norm-only rules passed CPU BF16 checks, including
future-position independence. On a future runtime, run the two native probes
first; stop on failure before study traces. No unvalidated batching or engine
change. Record actual dependency versions, token IDs, all 36/36 or partial
counts, matching failures, timing and native source hashes.

Checkpoint arrays outside mounted Drive during computation; export per prompt.
On success or fault, verify private Drive and local C: copies before releasing
only the owned model worker/runtime. A stopped worker is not released GPU billing.
Analyze prompt-level angle contrasts on CPU after release. Include all pairs,
conditions and missing evidence; do not select the strongest pair or metric.

## Resource forecast, not a new spending authorization

Previously observed hardware was A100 40GB, ~2.2GB live GPU memory; an 80GB device
is unnecessary for this small model. The original median matched replay took
4.185 seconds for 64 steps. Linear scaling suggests about 66 seconds for 36
28-step traces, excluding extra matching, prefill, validation, synchronization
and export overhead. Budget 2–6 minutes for collection, 5–10 minutes for
setup/restoration and 5 minutes for verified exports: approximately 12–21 minutes,
with a conservative 25-minute planning allowance and unchanged 45-minute hard
ceiling. Actual throughput/rate must be checked anew; no batch speedup assumed.

At the last observed 5.3 compute units/hour, 25 minutes would be about 2.21 units,
and the absolute 45-minute ceiling about 3.98 units. These are forecasts at an
old rate, not current credit evidence or USD charges. Native arrays are about
99 MB uncompressed for 36*24*28*1024 float32 values, plus pre-injection records
and diagnostics. Reserve at least 0.5GB on local C: and private Drive for complete
attempts, hashes and backups. No monitor or automatic run is created.
