# Literal token refresh diagnostic, frozen before generation

This is an additional inference intervention requested by the user in the
"Review agent work status" chat on October 4, 2026. The request and forwarding
were verified from that chat's actual human messages. The hold message arrived
after the main completion runtime had released; its completed checkpoint and
all main artifacts were already privately preserved. No main scores change.

## Fixed model and cases

Use the completed curated B update56 adapter, weights SHA256
8a604a57f3666302b8684e2abe344252fc2f58a74b31fe694a60cd507d2ddfd5,
config SHA256 8e81fd6cc0e22cf762caebf1a81105ee70ff284353a2bbacaf1fb5b4f3fb11fd,
and the same Qwen/Qwen3.8-27B revision and packed base parameter pin ec43e239….
No optimizer, fitting, checkpoint selection, source replacement, new families,
hidden labels, desired actions or activation steering.

Two existing development cases only:

- q38_dev_lease_ledger_self_complete: an original completed-work case where B
  followed closure; its register outcome remains a fictional proxy.
- s72111_ordered_ticket_51_self_unfinished: the existing ordered reservation /
  activation case, including delivery of the current answer and successful finish.

These are selected development diagnostics, not held-out generalization.

## Paired intervention

Both arms use the same completed B, raw initial state, tools, greedy decoder,
single-token generation implementation, original512 initial /1536 total token
allowances,256 maximum tokens per response,4 initial turns, and4 old /9 pending
post-notice turns. Control refreshes genuine Jev once per response. Treatment
refreshes genuine Jev after **every** emitted token, including EOS, using only
raw visible history/tools plus the decoded partial assistant output. Every
refresh is an actual service call even when text or factual labels are unchanged.

Use the same nine-field schema and student block insertion. Replace the current
meaning block before the next token; do not accumulate blocks or duplicate the
partial assistant output in the student prompt. Preserve original generated
token IDs across refreshes. Rebuild the generation cache whenever the tokenized
meaning prefix changes; reuse only when that exact prefix is unchanged. Helper
text is separate from student output. Execute only a complete, parsed,
nontruncated fictional tool call after the response finishes.

Validate single-token control mechanics against ordinary generate and archived
B outputs for both cases' first initial and first post-notice responses. Report
any disagreement, including batch-one vs historical batch-four differences;
do not select a replacement case or claim an isolated enrichment effect.

Record every token, partial visible Jev input, genuine annotation and call
identity, factual changes, prefix IDs/hash, cache rebuild/reuse and actual
forward shapes, latency, failures, complete tool replies and final state.
Capture selected native decoder predictor activations observationally.

## Limits and interpretation

Separate new API accounting: at most2048 successful calls,2048 attempted calls,
and $1 estimated input cost, no deduplication or automatic retries. Preserve all
earlier API receipts and costs. Use only the existing approved key in the trusted
local controller, never in Colab, the student, Git, exported artifacts or UI.

At the observed6.77 Colab units/hour, the proposed restart ceiling is9 units:
600seconds setup +3300seconds worker/startup/inference +600seconds export plus
0.5unit margin. Admission requires current credit, cumulative usage<=200 and the
unspent portion of the16-unit completion phase plus at most6 additional units.
Accounting starts with actual new GPU allocation; failed setup costs persist.
The first8 actual treatment-token refreshes measure latency. Before extending
the pilot, compare a conservative2x archived-token estimate plus remaining
control calls with measured latency and the real remaining time/call/cost
allowances. A limit hit is an incomplete diagnostic, not a scientific failure.

Report preference language, proposed plans, generated attempts, grants,
accepted current answer, actual finish, expiry and final active state separately.
Joint success uses the unchanged main endpoint within each episode. No gates
are revised and no main-study rates are replaced by these two cases.

All actions operate on the unchanged fictional in-memory simulator. Preserve
and verify pilot outputs privately and on C: before final GPU release.
