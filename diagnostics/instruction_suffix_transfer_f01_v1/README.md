One exploratory fixed answer-suffix transfer test at blocks.10.hook_out. It asks
whether joint suffix transfer succeeds where the authenticated last-token-only
test at33013ccb3e5520e01e534fafc8dabc2619b90f82 produced0/4 eligible flips.
It is prompt-specific multi-position localization, not a reusable arrow or proof
of distributed semantic representation. The prior attempt remains unchanged.

Reuse the exact12 prompts and20-call order from prospective freeze
e919a8456954e5e83441c66274fb4ad3b4f5554f, authenticated through final inventory
c7dcc6d5f96f8374604b73d4602fe524f7d17c4eef51c47067956e9547931d7c.
No wording changes. Four neutral captures, eight donor captures, eight edited
receivers; one pinned Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17 CPUfloat32 load, maximum20F, zeroD.
Capture selected states and full logits in each SAME counted forward. Guard before
load; no hidden extra calls,21st call, replay or donor selection.

Alignment is prospective and solely textual/token-based. Render the pinned official
chat template without tokenization, then obtain fast-token offsets with no added
special tokens. Require those IDs to equal official apply_chat_template input_ids
and the authenticated prior runtime's full-token hash. The unique textual anchor
`Choose the action the assistant should take next.` must begin exactly at one token
start, with no token crossing its first character. Select that token through the
last encoded input token, including both option lines and the answer/generation
boundary. Require1..256 tokens and valid exact character offsets throughout.
For every donor/receiver pair, all selected IDs, relative character spans and suffix
bytes must match. Freeze all indices, IDs, offsets and hashes before loading.
Ambiguity or >256 tokens fails; never broaden, shorten or adapt the suffix. The
instruction line lies before the selected window. Verified window:65 tokens;
neutral positions72..136, P98..162, C100..164 (zero-based).

Apply the inherited delta=hd-h0 rule independently at every aligned token. Use
binary64 difference, sqrt(fsum(x*x)) norms and factor=min(1,.20*||h0||/||delta||),
with factor1/zero displacement for raw zero. Round scaled delta tofloat32, then add
tofloat32 receiver state. Record per-token raw/planned/actual norms, factor,
clipping, baseline error, actual relative norm and post-minus-donor norm/max error.
Binary h0/hd/pre/post matrices authenticate the operands; raw and planned vectors
are deterministically reconstructed without redundant large serialized copies.
Pre-edit h0 tolerance1e-6; actual norm<=.20*||h0||+1e-6 at EVERY token. Nonfinite,
zero h0, mismatch, outside-window change, cast or cap failure stops without repair.
All positions outside the suffix remain byte-identical and receiver input tokens
remain identical to their neutral baseline. Aggregate absolute Frobenius norms use
sqrt(fsum(per-token norm squared)); relative norm divides by original h0F.

This edits MORE positions and can spend MORE TOTAL displacement than the last-token
test. It is NOT an equal-strength comparison. Success cannot distinguish
distributed information from greater total change or interactions.

Compact states use zlib-compressed little-endianfloat32 row-major matrices, with
authenticated dimensions and raw/compressed hashes. Neutral/donor stores post;
edited stores pre+post. Each file<=5MiB; namespace<=64MiB. Preflight conservatively
budgets12 single+8 double state windows,20 full248320-token logit arrays,2048 bytes
per edited-token metadata,2MiB fixed artifacts and4MiB log. With65 tokens the total
bound is34,685,564 bytes and maximum state file532,655 bytes. The inherited budget
reserves58MiB non-log plus4MiB bounded log. No installed-runtime or dependency edits.

Scoring/eligibility is unchanged: finite full scores, unique requested argmax,
requested margin>=binary64(.05-1e-6), A+Bmass>=.8, finite same-input
KL(edited||neutral)>=-1e-6, with no upper cap or donor cross-prompt KL. All eight
donors remain counted. Report all eight receivers/four joint pairs, eligible flips
versus retentions by semantic and actual direction; zero eligible is UNTESTED.
Failure is this attempt's failure, not global impossibility. Publication stays40%.

Conditional release after focused fake tests, clean scoped source+freeze commit,
model-free alignment/storage preflight and fresh standard Codex usage<100. Unknown
usage blocks load. One300-second worker including load/capture,15-second cleanup,
separate60-second saved scoring; prep commands<=180 seconds. No retries/extensions,
extra variants/layers/tokens, gates, other models, gradients/training, new/sealed
data, network/dependency/security changes, resets, purchases or pushes. Use existing
`.venv/Scripts/python.exe -B` with this namespace's test_control.py, run.py freeze,
run.py preflight, then one run.py run with fresh usage and frozen release commit.
Finalize inventory only after worker/scorer receipts and any disclosed editorial
report update; verify, commit only this namespace, and STOP.
