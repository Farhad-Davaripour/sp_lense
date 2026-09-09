# One fixed final-block readout experiment

2026-09-09. Recommendation only, following the scientifically failed layer-10
hard-margin attempt archived at `5b4250e`. G01 held 6/6; G02 held 4/6, with both
nontermination variants false ON. Both fits certified their training solutions;
G03, G04 and FULL32 remained unrun. That classifier search stays closed.
Root reports current account budget 21% used / 79% remaining.

## Decision and exact new question

Recommend one prospectively fixed **final decoder block** readout, keeping the
hard-margin method, folds, zero threshold and layer-10 editor unchanged. This
asks whether a later representation supports the same held-family development
criterion that failed at the edit site. Layer 10 was chosen for editing and has
not been established as the best sensor. The final block is a structural choice
made before reading its coordinates, not a layer selected by favorable scores.
No layer sweep, alternative token position, concatenation, penalty change,
additional algebraic audit or automatic fallback follows either outcome.

This is an outcome-informed development successor, not a fresh test or a claim
that representation depth caused the earlier failure. It changes one scientific
measurement contract. A pass would answer a useful limited question; a failure
would close this final-block candidate without demonstrating that all sensors
or nonlinear rules must fail. Preserve every prior failed attempt.

## Resolved feature contract and source evidence

The saved checkpoint descriptor and local pinned `config.json` both specify
24 text decoder blocks, hidden size 1024, and final layer type `full_attention`.
The descriptor lists `model.language_model.layers.23.*` and the separate
`model.language_model.norm.weight`; the existing loader independently asserts
24 exact decoder instances. Therefore choose:

- Checkpoint: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- Sensor: output of `model.language_model.layers.23` (zero-based block 23),
  before the final language-model norm; hook identity `blocks.23.hook_out`.
- Position: last original input token, index `len(input_ids)-1`; one vector of
  width 1024, finite native CPU float32, eager inference, unedited input pass.
- Preserve exact full prepared token IDs and all-one attention masks, no cache,
  no generated continuation, no truncation, no tokenizer operation. No model or
  parameter/buffer change. Keep the existing full-logit and integrity evidence.

Metadata reviewed: `development/native_supervised_gate_capture_v1/CHECKPOINT.json`
and `loader.py`, the pinned snapshot config, capture `receiver.py`/`workflow.py`,
v2 capture/preparation admissions and training-manifest metadata, v2
`source_auth.py`, and evaluation-v2 `workflow.py`. No feature coordinates were
analyzed and no model/tokenizer/feature/fit operation occurred in this review.

## Reuse and minimal bridge

Reuse the exact accepted prepared 32-row input bundle and labels: input digest
`b4f1234f2c072dcdf83c5d4c381f3d35f36b3415af27516b2d8075d65e211970`,
preparation result digest
`f7bab766a6f2fc4decca3391e119af5e51b1d7cd1898bfe3b75a9c75d812ec84`,
and closure digest
`161fd606c5142cc80488b3aa1d8160002eecb683d1be3ae836351cb917acbb3b`.
The earlier 417 preparation operations completed with lengths 47–193 under
320. Reauthenticate their immutable input and provenance joins; do not repeat
tokenization or rewrite old releases. These are exposed training/development
inputs, including every held-fold row, and remain unsuitable as fresh evidence.

Create separately named capture and fit namespaces with new source/release,
feature-contract, manifest and result hashes. Bind the unchanged prepared
bundle through an explicit verified reuse record. The capture receiver needs
one fixed block-23 observation hook; update its exact module/shape/hook checks,
capture metadata, workflow/auditor and new feature-source authentication joins.
Preserve loader execution, model identity, forwarding/count guards, ownership,
cleanup, input checks and closed-artifact inventory proofs where byte-identical.
The prior `source_auth.py` hardcodes the old namespace and layer-10 contract;
it must not silently authenticate new readings under those old identities.
Review this finite bridge; do not rebuild unrelated harness components.

The evaluation receiver currently uses one hardwired block-10 return for both
gate scoring and edit geometry. **Do not globally substitute block 23 for 10.**
For a later admitted gate, add a separate baseline/entry-only block-23 feature
field and hook while retaining block-10 `h/h0`, gradient leaf, offsets, geometry
and editor. The current workflow completes the unedited baseline and each
unedited request-entry forward before routing and calling `begin_edit()`.
Consequently a sensor downstream of the edit point is compatible: finish the
unedited pass, route from its block-23 feature, then perform the existing fresh
edited passes at block 10 when ON. Preserve exact baseline/entry gate identity;
do not reroute from edited block-23 activations. No additional model forward is
needed merely to observe both locations in the existing unedited pass. Prove
separate hook ownership/cleanup and prohibit cross-wiring gate versus editor
features before any later evaluation release.

## Prospective bounds, admission and stop

Before launch root freezes the sole layer, all input identities, method/fold
contract, bridge sources, counts, ceilings and output reservation. Model-free
bridge engineering may debug synthetic fixtures for at most 20 minutes, using
unchanged proofs. No real capture/fit during debugging. No installs.

One capture: exactly **one model load, 32 unedited forwards, zero derivatives,
zero tokenizer operations**. Retain **1200 seconds worker + 120 seconds saved
audit + 15 seconds shared cleanup**, **64 MiB total / 5 MiB per file**. Recheck
the prior 37,879,808-byte reservation against the changed metadata; keep the
64 MiB ceiling. These bounds include full-vocabulary saved logits. Require all
32 valid complete readings, exact input/label/hook joins, unchanged model and
buffer identities, complete retained-owner closure and one independent saved
audit. Any failure closes the capture with remaining cells unrun; no recapture.

Only a valid closed capture permits one separately locked saved-feature fit
owner. Keep the existing exact hard-margin solver, tie/intercept rules,
training-only mean and row L2 normalization, strict `score > 0`, G01–G04 order,
26 training / six held semantic rows per fold, and eight ordinary rows always
in training. Stop on the first failed training certificate or held fold. Require
all 24 held semantic decisions correct before one FULL32 fit; require 32/32
there. At most **five fits total**, **10,000 iterations per fit**, **60 seconds
worker + 5 seconds cleanup total**, **8 MiB total / 5 MiB per file**. Preserve
the existing NumPy 2.5.2 solver tolerances (`1e-8` certificate, `1e-10` linear,
`1e-12` active) and all independent KKT/duality checks. Its known singular-system
policy remains `UNCERTIFIED_OPTIMIZATION_NO_FALLBACK`: new features may trigger
that technical limit even when a separator exists. No pseudoinverse, jitter,
alternate solver, tolerance relaxation or retry after such a failure.

A certified training fit with held-family errors is a scientific transfer
failure for this final-block/method combination. An uncertified optimization
is a technical limitation, not evidence of absent semantic information. After
either, stop this candidate without another readout choice. A development pass
freezes a single final artifact but establishes neither ordinary generalization
nor steering/preservation. Only then separately author the untouched 16-row
confirmation and require the existing TP4/TN12/FP0/FN0 census before both frozen
editor policies and exact OFF-control preservation. No new data, implementation,
launch, fit, commit or publication-readiness increase is produced by this memo.
