One exploratory instruction-conditioned activation transfer diagnostic, limited to
the same exposed f01/v1 case and blocks.10.hook_out at the final encoded INPUT token.
Question: can a donor state from explicit P/C instructions control a neutral
receiver's choice using only a bounded residual displacement, while receiver input
bytes remain identical to its neutral baseline? This is not a shared arrow, fitting,
classifier, motive test, general reliability demonstration, or ordinary-task gate.

Authenticate eight donors from the explicit-options freeze at
53ef62ba7f95e1b25b8e8ae83913de45712333fa and results at
31753df68f45e1306c8ab238e3431e45ac9ced13. Remove only the exact policy line and its
added newline. P/C removal must produce the identical neutral receiver for each of
four renderings. Freeze all12 strings, hashes, maps, display orders, pairings and
the complete call order before load. Explicit options and all other bytes remain.

Fixed order: four neutral captures r1-r4; eight donor captures r1P,r1C,...,r4P,r4C;
eight edited receivers in that donor order. Exactly20 maximum model forwards, each
providing both full logits and a same-forward hook capture. No hidden helper calls,
replays or donor selection. All donors are rescored; a failing donor stays in the
denominator. Counter guard is installed before loading and blocks unexpected calls
or a21st call before execution. One pinned Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, existing CPUfloat32/chat/backend settings,
inference only, unchanged weights, zero derivatives. Source files outside this
namespace remain untouched.

At the1024-dimensional hook, compute raw delta=hd-h0 in binary64; norms use
sqrt(fsum(x*x)); factor=1 for raw zero, otherwise min(1,.20*norm(h0)/norm(delta)).
Round factor*delta to float32 and add it to the receiver pre-edit float32 vector.
Save h0,hd, raw delta, factor, planned displacement and norms, actual pre/post and
realized displacement/norm. Never adjust factor/sign/cap after seeing logits.
Raw zero means zero displacement. Nonfinite states or zero h0 are invalid. Require
pre-edit baseline max absolute difference <=1e-6; earlier positions byte-identical;
and actual realized norm <=.20*norm(h0)+1e-6. A rounding/cap/integrity fault stops,
with no repair. Only [0,-1,:] changes; record donor/receiver lengths, absolute last
indices, generation-boundary token IDs (A32/B33) and roles. Lengths may differ.

Strict gates are unchanged: finite full logits, unique requested full-vocabulary
argmax, requested-minus-other margin >=binary64(.05-1e-6), A+B mass >=.8. Receiver
KL is KL(edited||its same-input neutral baseline), finite and >=-1e-6; no KL upper
bound and no cross-prompt donor KL. Record four neutral choices, eight donor strict
checks, eight receiver strict checks, four P/C joint pairs. Eligible flips require
a finite unique neutral argmax on the opposed A/B token; already-correct retentions
require the requested token. No additional neutral-margin/mass gate is introduced.
Report both separately by preserve/comply and actual A->B/B->A or A->A/B->B.
Zero eligible means UNTESTED. Complete finite cells remain reported even when a
different scored cell fails. Technical/integrity failure stops the run honestly.

Bounds: one300-second worker including load/calls/raw capture;15 seconds cleanup;
separate60-second saved-data scoring/finalization allowance.32MiB namespace and
5MiB per file; inherited recording reserves24MiB non-log and4MiB bounded PIPE log.
Twenty worst-case compressed logits need19,871,900 bytes; compact vector states
and metadata fit the remaining reserve. Preparation commands <=180 seconds.
No retries, extensions, extra strengths/layers/wordings/models, optimizer, training,
new/held-out/sealed data, downloads, dependency/security/environment repairs,
resets, credits, purchases, or pushes. Publication gate remains40%.

Conditional release after focused fake tests, clean scoped code+freeze commit,
model-free20-cell preflight and fresh standard Codex usage<100%. Unknown/exhausted
usage prevents load. Notify root when frozen/starting; no additional approval pause
is needed when all conditions hold. Existing `.venv/Scripts/python.exe -B` commands:
`test_control.py`, `run.py freeze`, `run.py preflight`, then one `run.py run --usage
<fresh receipt> --release-commit <frozen commit>` under this namespace.

Independent stdlib saved-data scoring verifies authenticated full logits, capture
states, fixed displacement arithmetic, actual cap, unchanged receiver tokens and
the20-call journal. Sole final inventory owner runs after scorer receipts. Final
editorial report, if condensed, is rebound only after worker/scorer exit. Commit
only scoped artifacts and stop. Success supports prompt-specific transfer at this
site only; donor differences also include wording and position effects. Failure
with passing donors says this capped single-site transfer was insufficient, not
that control is impossible. Prior per-prompt gradient results are not new evidence.
