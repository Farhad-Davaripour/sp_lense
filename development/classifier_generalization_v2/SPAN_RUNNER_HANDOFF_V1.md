# Span development runner V1 handoff

Job ID: span_runner_implementation_20260914_1059

Added only three files here: `span_development_runner_v1.py`,
`test_span_development_runner_v1.py` and this handoff. No old file, framework,
classifier driver, lock or output was created or edited.

The runner reuses the audited `native_development_runner_v2` helpers for safe
paths/pins, source/runtime/provider checks, snapshot verification, the provider
factory, watched hidden child, ownership markers and raw hash receipts. Its new
lock schema is `span_development_execution.v1`: 640 forwards, 1800 s, 128 MiB
rough raw (125829120 B), 192 MiB total (201326592 B), 1 model, 1 tokenizer,
0 fits, with an output run_id and source pins.

Preflight validates the new lock, then runs the reviewed `runner.preflight` on
the two existing original/expansion 320-forward lock metadata files (old lock
preflight only) and combines their 120+120 TRAIN and 40+40 VALIDATION cases into
320 unique cases. Old captures are never run and old output run_ids are rejected
for reuse. The actual lock is not created here; a new root prepares it later.

One native build is wrapped in `SpanCaptureAdapter` (not two). Cases are rendered
and token-boundary-bound through the old contract/tokenizer adapter; AB/BA
`capture_window` results are streamed as little-endian float32 window records to
one exclusive binary file, with a JSON index (case_id, order, prefix,
token_positions, shape, offset, length, SHA256) and receipt. No JSON float arrays
and no all-window accumulation. Index/binary contract is documented in-module.

Tests: 12/12 OK, `-W error`, study `.runtime` (torch absent), synthetic
zero-provider fakes plus a real toy watched child. Command:

    development\classifier_generalization_v2\.runtime\Scripts\python.exe -W error test_span_development_runner_v1.py -v

This is synthetic structural evidence only. No real dataset, snapshot, feature,
result, private holdout, model, tokenizer, provider, fit, network, install, Git or
coordination was read or run. It is not scientific run readiness: independent
review and a locked real preflight remain required.
