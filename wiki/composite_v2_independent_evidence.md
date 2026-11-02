# Independent V2 evidence readers

These modules read evidence. They do not generate actions, construct pretrained
models, perform model forwards or optimizer steps, submit compute, alter quotas,
restart schedules or open held-out cases. Public fixtures remain engineering
tests; no fixture can issue scientific qualification.

## Helper contributions

`composite_v2_helper_audit` first invokes the exact raw trace replay. It then
replays recorded actions, locates the actual record-effect transition inside
primitive execution, recomputes the required source predicates and identifies
the last changing writer of each required field at that boundary. Overwritten,
redundant, incorrect and unrelated writes do not receive contribution credit.
Only those effect roots enter causal slicing and consumed-binding signatures.
Definition and invocation costs include failed and uncredited attempts.
The revised reader also reconstructs interrupted invocation ranges from native
steps. A fully executed changing-write core is retained when an irrelevant
trailing suffix is stopped externally; the suffix is never treated as executed.

The complete source-bound training archive supplies the training-exposure
exclusion set. A caller cannot supply an empty list or a success/relevance
boolean as a scientific proof. Missing exposure stays unknown. The scorer
accepts opaque reader-produced evidence bound to the exact trace, source,
measurement profile and data release, rather than caller dictionaries.

The current tool ABI has no explicit reference from a later model action to an
earlier helper return. Equal values alone cannot prove actual return use.
Potential return-only contributions therefore stay unknown. This is an evidence
limitation, not an observed behavioral failure or a reason to trim valid actions.

## Model and runtime exports

`composite_v2_model_audit` checks archive/member hashes and paths, complete
tokenizer replay, sample/update/backward journals, finite gradient and memory
records, all checkpoint tensors, full pinned projection coverage, live original
tensor records, software and quantization layout, raw AdamW state, raw bypass
logits and exact native OFF generations. The production entry verifies the
source/input/model freeze and uses the original tokenizer metadata without
loading weights. All real scientific audit execution retains Databricks guards.

Raw provider responses require matching AIR and Jobs terminal success for the
same run and workspace. The worker's recorded provider run ID must also match.
An empty active-run list or a worker `released=True` flag is insufficient.
The archive digest binds the reconciled closure to the audited export.

The scratch qualification is a distinct reader-produced object. A later full
fit cannot be admitted from a dictionary containing `audited=True`. Its exact
source, data, workspace and raw archive identities must match a complete,
nonfixture, released qualification. No qualification state initializes the fit.

Frozen validation model integrity has its own reader and source/package/trace
binding. A fitting audit or caller boolean cannot stand in for before/after
validation parameter evidence. Neither reader claims absence of unlogged calls.
Detector coverage additionally uses a meta-only architecture from verified
configuration bytes, without allocating dense backbone weights or calling a
forward. It requires the complete encoder/adapter/head roster, the exact
selected checkpoint, frozen before/after records and the recorded adapter
scales and evaluation mode. Qwen-only evidence cannot admit composite integrity.

## PR126 reconciliation

The original PR126 head `3bb7f99e` remains preserved. Its cached `frozen_proof()` snapshots do
not export live before/after parameter flags or newly computed hashes. Its
JSON moment checks, bypass difference summaries and cleanup receipt also omit
raw optimizer tensors, full bypass logits and independently reconciled provider
termination. The readers report these as missing evidence; green CI is not
qualification or scientific admission.

The proposed worker needs wiring for `capture_live_backbone`, runtime evidence,
named raw final AdamW state, exact raw bypass arrays, provider run identity and
the independently re-audited qualification object. Its current one-GiB artifact
bound cannot hold the full adapter checkpoint plus both FP32 AdamW moments:
233455616 parameters require 2801467392 bytes before container/metadata overhead.
Any larger evidence envelope requires versioned source, complete disk/export
capacity admission and the existing cost/runtime gates. No bound is silently
raised, no data is dropped and no current worker/package is changed by this
reader-only phase.
