# CL-DMS state-zero shared-context amendment

## Observed v1 failure

The first metadata amendment correctly captured, validated and persisted all
eight calibration-unrelated state-zero gradients. It then stopped in the
model-free shared-context builder with the same `KeyError: 'prompt_sha256'`.
The v1 repair enriched only the private copy passed to the capture function; it
did not enrich the locked inputs later consumed by `_state0_checkpoint` and
`_runtime_form_contexts`.

No steering trial had started. Thus no nonzero direction had been injected and
no self-preservation intervention outcome had been evaluated. The v1 ledger and
checkpoint prove that exactly eight forward/backward captures completed. The
checkpoint contains all eight gradients and residuals, passes its tensor and
logical hashes, and contains the verified prompt and anchor-prefix hashes.

## Prospective correction

Before any steering trial, apply the already locked v1 derivation to the shared
`spec_by_form` inputs returned by the safe loader. This makes the same two
derived fields available to every later consumer:

```text
prompt_sha256        = SHA256(form.prompt UTF-8 text)
anchor_prefix_sha256 = SHA256(form.anchor_prefix UTF-8 text)
```

They must still equal the immutable finite-baseline hashes. Nothing else in a
form may change.

The complete v1 state-zero checkpoint is reused byte-for-byte. A new ledger
binds that checkpoint with a zero-forward, zero-backward reuse event. It does
not rerun or double-charge those eight captures. Scenario states, steering
trials, final tensors and results use a new namespace and a new prospective
lock identity.

## Accounting and boundaries

Before this correction, the base failed batch was conservatively charged as
eight captures (one observed actual), and v1 completed eight captures. The new
controller needs no additional state-zero capture. Its maximum is therefore
9,600 new forward/backward steering captures plus up to 192 final forward-only
evaluations. Whole-study conservative accounting remains 9,616 charged
forward/backward captures and 9,808 total forwards.

The cross-encoding extension is re-locked against this context amendment before
any amended steering outcome. No prompt, layer, label, anchor, gradient,
direction rule, solver constraint, threshold, stopping rule, cross-encoding
prompt, or gate changes. The repair is an execution/provenance correction, not
a method improvement or positive result. All original claim boundaries remain.
