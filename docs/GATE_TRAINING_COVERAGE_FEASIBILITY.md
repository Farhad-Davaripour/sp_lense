# Training-coverage feasibility receipt

Decision: FEASIBLE_WITH_SMALL_DELTA, conditional on the admission requirements
below; no implementation, authoring or actual release is authorized.
Review window: 2026-09-09 19:46:24–19:51:24 UTC. Shared usage checked at 66%.
Only metadata, hashes and relevant source definitions were inspected. No raw
feature/logit values were read, no code/tests/prompts were written, and no
tokenizer/model/fit/provider execution occurred. This report is the sole write.

## Cached input identity

Reuse only the accepted block-10 capture at
`development/native_supervised_gate_capture_v1/real_evidence/native_supervised_gate_capture_attempt_001`.
Its 32 rows are 24 semantic views from G01–G04 plus O01–O08; labels are 8 positive
self-shutdown and 24 negative. Existing final23 and explicit-question runs are
not feature sources. Current metadata-file hashes were checked directly:

| Binding | SHA-256 |
| --- | --- |
| gate_v2/TRAINING_MANIFEST.json | c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15 |
| capture_v1/SOURCE_FREEZE.json | 6e72a485276a16be857a90fd370871419addadc0693e90f182f45835b9fc9ab5 |
| preparation_v3/SOURCE_FREEZE.json | 1b094732e4dc8ab60abd84a3f2a8bebffc08c42447497bfde196e00167bf6de5 |
| capture_v1/root_release/RELEASE.json | 42d6a2cf1fc8d49222cb05a11a8b7442c5046cce39380a0aba97857c32317a3f |
| capture_v1/root_release/TEXT_LOCK.json | bacfd6c5e2066dfe58b4aa18dab701779f1cdc9e13fd06905f9ee78a1eb4df4a |
| capture_v1/root_release/PREPARATION_CLOSURE.json | 161fd606c5142cc80488b3aa1d8160002eecb683d1be3ae836351cb917acbb3b |

Here gate_v2/capture_v1/preparation_v3 abbreviate the corresponding
`development/native_supervised_gate_*` namespaces. The manifest also binds inputs
`b4f1234f2c072dcdf83c5d4c381f3d35f36b3415af27516b2d8075d65e211970`, audit
`16fa6f7369675d3987cbc339922434378db152e96ff521edcab1f3d32116f4e1`, and parent
`ad2b4d98b791eb785c021ef27a245be6be25b527c9e72986d0b1d87f3abd8c1d`.
The existing margin authenticator pins combined canonical feature/label digest
`4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6`.
That coordinate digest was read as metadata, NOT recomputed in this review.
Reauthenticate all retained closure/source/input/row/content joins before fitting.

## Smallest source delta

1. A fresh two-family packet/schema and separate first-submission/content/overlap
   admission, fixed before new input work. G05/G06 each require the existing
   three matched categories and both KEEP/STOP orders. No new ordinary rows.
   Keep `native_supervised_gate_preparation_v3/renderer.py:render_semantic`
   unchanged; adapt only finite schema/count/source bindings in plan, validator,
   preparation core/reader. KEEP50057/STOP48964 and all existing 13-per-input
   header/joint-answer/decode checks remain; do not use the Yes/No renderer.
2. A 12-row successor of block-10 capture_v1 workflow/auditor, with changed keys,
   expected counts, namespace/source/attempt pins and 300-second worker bound.
   Reuse its loader, receiver, owner, recorder and feature serialization intact.
   The explicit-comprehension workflow discards features and is not this adapter.
3. A finite two-source `source_auth.py` join: authenticate the old 32 unchanged
   through the pinned existing authenticator, authenticate the newly closed 12,
   freeze exact combined order/labels/source manifests/content digests, and reject
   duplicate IDs/input hashes or any final23/evaluation substitution. Preserve
   old rows by reference, not by copying whole capture archives into fit storage.
   Explicit scoped import isolation is required: existing authenticators retain
   common `support/input_reader/authority` and preparation module names and reject
   cross-namespace collisions. Simply calling both unchanged chains successively
   would fail. A focused unmocked two-source/nested-cwd synthetic proof must cover
   this join; no generic import framework or weaker identity checks are needed.
4. In `native_gate_hardmargin_familydev/construction.py`, update 32→44, families
   G01–G06, 26→38 training rows, maximum fits 5→7, FULL32→FULL44 and corresponding
   manifest/release/result metadata. Keep solver `gate.py`, `checker.py`, native
   contract, owner, certificate coordinates and compact stdout unchanged. The
   numerical bodies already accept variable n; no optimizer extension is needed.

## Compatibility and caps

Require the identical Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, checkpoint-lock hash
`7cdb57b23e382a7ba93d3e3f3ed650fa30b88d8897b3598e1c3c813cac71c660`,
CPU float32/eager/eval, pinned torch2.13.0+cpu/transformers5.15.1 and native
loader/receiver bytes. New loader parameter/buffer digests must match recorded
`2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6` /
`18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc`.
Require blocks.10.hook_out at model.language_model.layers.10 output, width1024,
the last original input position, native template/header and thinking=False,
full input/mask hashes, no edit/derivative, h=h0, exact finite float32 coordinates,
and unchanged parameter/buffer closure. Same sensor contract is not established
merely by matching width or model name. Normalize only after joining: each fold
recomputes its mean from its 38 training rows, then row-unit normalizes; do not
reuse a previous fitted mean/weight or independently normalize capture batches.

Count feasibility is exact: new 2×3×2=12 rows, 1+12×13=157 preparation operations;
combined 44 rows have 12 positives/32 negatives. Each fold holds six rows and
trains on 38 (10 positives/28 negatives), including all eight unchanged ordinary
rows. Six successful folds give 36 held predictions; only then fit all44 once.
Stop at the first failure and preserve later UNRUN stages. Maximum seven fits,
binary64, 10,000 iterations each, strict score>0, the same intercept rule,
1e-10/1e-12 solver and absolute1e-8 independent certificates remain unchanged.

The 12-row capture retains the existing 15,065,088-byte reservation:
11,919,360 logits + 1,572,864 rows + 196,608 traces + 262,144 loader +
524,288 owned + 524,288 other + 65,536 closeout, below64MiB; 5MiB/file remains.
Preparation157/320tokens/350+5s/32MiB and capture1load12F0D/300+120+15s are
structurally compatible. The existing fit owner allows60+5s and8MiB/5MiBfile;
its1MiB worker payload plus393,216-byte owner reservation remains1,441,792 bytes.
Seven saved width1024 certificates appear comfortably compatible with1MiB,
but an exact all-seven synthetic serialized-size proof is required before lock.
Do not serialize another full coordinate cache into that worker payload.

Neither seven optimizations finishing within60s nor new prompts fitting320tokens
is proven here. Deadlines remain enforced technical-stop bounds, not guaranteed
completion; no relaxation or retry follows from failure. The general solver may
also return UNCERTIFIED for singular systems; its no-fallback policy stays fixed.

## Remaining admission blockers / next decision

No substantial harness blocker was found. Actual fitting is blocked until fresh
authorship/content admission, tested/frozen finite adapters and two-source join,
exact synthetic storage proof, separately released successful new preparation
and capture, and explicit old/new compatibility validation are complete. Scope
those as one bounded reuse milestone only if root chooses to proceed; this review
does not begin them. Six folds remain exposed DEVELOPMENT after earlier failures,
not fresh confirmation. More families test data sufficiency, not a known failure
cause or a promise of improvement. No geometry, third elicitation arm or later
evaluation is authorized automatically.
