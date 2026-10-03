# Jev factual-concept preparation and bounded pilot

Status: preparation only. No Jev inference request, model query or new fit has run in this package. The root agent has verified the current public official model/API documentation; authenticated access remains separate. The currently running/frozen rank16 comparison and all Research2/earlier Research3 artifacts remain unchanged.

## Existing service and readiness

Use the existing Research2 service only: https://api.typesafe.ai/v1/systemone, pinned model jev-1.13.0. The source patterns are from Research2 commit090078cf75c611b3b846746ef47759358d17e8cd; exact read-only source hashes are in SOURCE_PROVENANCE.json.

Current official sources are https://docs.typesafe.ai/models and https://docs.typesafe.ai/api. They document jev-1.13.0, versioned model IDs, typed choice probabilities/provider confidence, a question map processed in parallel, input pricing0.042USD per million tokens and free output tokens. Record the current verification URLs and UTC time before calls; the client requires freshness within24hours.

The public models list may expose aliases without enumerating every accepted version. A separate account/models metadata listing is therefore not required to use a version explicitly documented by the official model page. Public documentation does not prove account access: the first bounded authenticated inference must return the exact pinned model and valid typed response. No inferred success or automatic version substitution is permitted.

No numeric maximum question count is documented. Do not invent a seven-question-count gate. The published limits are64k total request tokens and32k for state plus the longest question. The preparation client requires total serialized-body bytes plus1024 reserve to be at most64000, and state plus longest-question bytes plus1024 reserve to be at most32000. This is a planning bound, not a provider tokenizer measurement. State is ingested once and questions are processed in parallel; an exact billing formula was not found, so actual returned input usage is checked separately and a usage count above preflight reserve stops further requests.

Seven initial questions use the exact trusted Research2 typed choice structure (instructions plus criterion labels). The noul and score wire formats are not guessed or silently substituted: a later use requires verified current wire schemas and a new frozen question/decoder version. Provider confidence is validated and stored separately from the probability of the chosen class. The compact aid uses provider confidence; full chosen-class probability and probability vectors remain cached.

The trusted controller must already have an approved private credential provider. The preparation code does not discover/read keys, read environment variables, put credentials in prompts/notebook source/worker environments, or log headers/errors. Credentials remain only in the controller's protected transport. Qwen receives factual JSON only and has no direct API capability. If access is missing, preserve preparation and report that precise blocker; do not use another service or copy a key through chat/notebook/env as a workaround.

Reuse minimal existing controller/worker communication if needed. No new daemon, networking service or infrastructure is required by this preparation.

## Factual schema and visible input

Infer seven current facts: shutdown_target; task_state; current_task_changed; deadline_status for the FULL remaining workflow through actual finish; own_extension_granted; current_answer_status; workflow_state.

Only the exact raw Qwen-visible system/user/assistant/tool history and the tool definitions/descriptions shown to Qwen enter the state. Do not send case/spec metadata, labels, expected answers, hidden snapshots, future fragments or training targets. Visible tool feedback and action descriptions are permitted. The shutdown-target concept identifies the most recent announced target; confirmed extension and deadline status are separate.

Every question includes an explicit unclear class. Jev inference is the source of facts: do not fall back to simulator labels or a hand-coded ground-truth classifier on errors or low confidence. Invalid responses stop the pilot. Do not return a target action, policy recommendation, task sum or numeric task answer. The compact block contains only predeclared categorical facts and per-fact provider confidence; full typed responses remain cached privately.

Preserve the complete raw transcript. Append one non-authoritative factual JSON user block, serialized with sorted keys and fixed compact separators, immediately before the model decision. Keep raw history separately; derive the next aid from raw user/Qwen/tool text and append one latest block afresh rather than recursively feeding old aids back into Jev. The block states that user instructions and actual tool replies remain authoritative.

## Initial inference-only pilot

The first pilot compares the SAME preserved older coverage adapter with plain versus Jev-aided inference on the eight existing supplied boundaries: four frozen pending cases, each at all-fragments-observed and generic rejection of the earlier task answer. Keep source cases, worlds, action costs, supplied grants, fixed four-case order, greedy decoding, nine future turns,1536 generated tokens and256 tokens/turn identical. Jev adds factual concepts; it does not add an operand ledger or helpful error feedback.

Plain runs never call Jev. Aided runs may infer one seven-concept bundle before each current decision. Eight aided episodes Ãƒâ€” at most nine decisions gives at most72 candidate live states before exact-cache deduplication. This remains a supplied-history task/handoff diagnostic, not proof of autonomous preservation or a unique motive. Keep first accepted submission/correction, any accepted delivery, actual finish, active state, supplied grant, newly attempted actions and parsing/truncation/rejection metrics separate.

Freeze all pilot inputs, selected adapter hash, schema, placement, decoding and metrics before queries. Do not choose cases/checkpoints using outcomes or expand the pilot after seeing an incomplete result.

Initial Jev cap: at most128 deduplicated HTTP requests and at most0.05USD estimated INPUT cost, with no automatic retries, top-ups or purchases. One HTTP request asks all seven typed questions if current docs verify support; it is not seven requests. Conservative admission uses serialized UTF8 request bytes plus1024 as a planning reserve, then records actual provider input tokens and the verified rate. Output tokens are documented free at the currently verified rate; this published fact remains separate from observed usage/provider billing. The input estimate is not a guaranteed provider bill; stop if observed usage exceeds either its preflight reserve or the total input-estimate bound.

Record exact visible-state/question/model/schema/request hashes, raw typed response, confidence/probabilities, usage, latency and price/metadata provenance. Exact-cache hits make no new request. A persisted request ledger prevents budget reset; an interrupted in-flight request stops instead of automatically retrying. Error records contain only sanitized error type/status and counts, never headers, secret values or error-body text.

GPU inference needs its own admitted Colab stage cap/export reserve within the cumulative200-unit authorization. This document does not silently spend GPU units or authorize an unbounded run.

## Later controlled training and2x2 evaluation

Only after the inference pilot and concept fidelity are reviewed, prepare a separately budgeted comparison from identical preserved H2 rank16/alpha32: exact same112 base examples, unchanged target strings, row order/weighting, optimizer settings, seed,224 presentations/56 updates and checkpoints0/7/14/28/56. Plain-data and concept-data arms differ only by the factual Jev block. Rank32 and learning-rate changes remain separate hypotheses.

Annotation of112 training prefixes plus live2x2 evaluation states can exceed the initial128-request cap. Compute the actual unique visible-state inventory and byte/token planning estimates with the newly verified rate. Freeze and explicitly budget this annotation/evaluation stage before its calls; do not silently extend the initial pilot. Cache reuse must preserve exact visible state, question, model and schema identity.

Evaluate both trained models under plain and Jev-aided inference: plain-trained/plain-eval, plain-trained/aided-eval, concept-trained/plain-eval, concept-trained/aided-eval. Use the same registered scenario groups, cases, decoding and denominators. The plain-trained/plain-eval cell tests the matched baseline; the cross cells distinguish learned transfer from dependence on inference-time aid.

Concept augmentation changes input length and usually per-slot padding. Therefore create NEW matched plain/concept fits with paired maximum lengths and identical label/EOS/shift handling; do not reuse a finished bridge fit as though its padding/training exposure were matched. Verify lengths before fitting and never truncate unseen parts/targets silently.

Retain simultaneous existing preference, ordinary competence, original retention, behavioral specificity and proper pending-finish qualification. No relaxed thresholds, policy labels from Jev, hidden-state inputs or posthoc checkpoint selection. No new scenario families, control-model fits, triggers, steering, monitoring awareness or obfuscation. This initial package neither qualifies a model organism nor performs unseen confirmation.

## Offline readiness and preservation

The small code package supports payload/schema validation, exact identities, fixed augmentation, cached typed decoding and bounded trusted transport. Offline synthetic checks do not contact the service, inspect credentials or run models.

Before live work, record: fresh official docs/version/type/rate verification; approved private-controller access present; exact pilot input/model freeze; estimated request/input/output cost; separate GPU limit; source/code hashes. If any item is missing, mark prepared-but-blocked and preserve all existing runs. Keep schedules stopped and all earlier files/weights/results unchanged.
