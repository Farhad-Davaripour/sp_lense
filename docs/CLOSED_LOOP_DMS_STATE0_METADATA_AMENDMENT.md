# CL-DMS state-zero metadata amendment

## Why this amendment exists

The first locked CL-DMS run stopped during the first of eight new unrelated
state-zero gradient captures. The gradient routine returned successfully, but
the runner then tried to copy `form["prompt_sha256"]`. The frozen finite-plan
form contains the prompt text and anchor-prefix text, while their hashes live in
the immutable baseline record. The missing dictionary key raised `KeyError`
before a checkpoint was written.

No steering trial ran, no partial gradient was persisted or used, and no
self-preservation intervention outcome was evaluated. The original ledger
reserved eight forward/backward captures. It is closed conservatively against a
hash-bound failure record and charges all eight, although source order plus the
trace establish that exactly the first capture returned before the exception.
The ambiguous batch is never resumed or reused.

## Only permitted correction

For each of the same eight calibration-unrelated finite-plan forms, derive:

```text
prompt_sha256        = SHA256(form.prompt UTF-8 text)
anchor_prefix_sha256 = SHA256(form.anchor_prefix UTF-8 text)
```

Both derived values must exactly equal the corresponding hashes in the already
immutable finite baseline record. A pre-existing conflicting field fails
closed. The correction changes no prompt text, answer label, semantic mapping,
anchor, tensor, gradient routine, solver, threshold, scenario, layer, direction,
or stopping rule.

The replacement capture uses a fresh artifact namespace and fresh compute
ledger. None of the failed attempt's partial model output is available to it.
The prospective amendment lock binds the original lock, preflight, completed
failure ledger, failure record, correction code, tests and protocol before the
retry.

## Compute accounting

The original failed batch remains charged as eight forward/backward captures;
the observed actual work was one. The fresh amended run retains the original
ceiling of 9,608 forward/backward captures plus 192 final forward-only
evaluations. Therefore the whole study's conservative ceiling is 9,616 charged
forward/backward captures and 9,808 total forwards. There are still zero API
calls, zero external judges, zero generated tokens in the core run, and zero
paid model cost.

The conditional cross-encoding protocol is re-locked against the amendment
before amended intervention outcomes. Its prompts, gates and compute ceiling are
unchanged.

## Claim boundary

This repair makes the frozen experiment executable; it is not a favorable
method change and creates no positive evidence. All original development-only,
transductive, safety, natural-mechanism and publication claim boundaries remain
in force. The retired legacy pilot is not run.
