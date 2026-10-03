# Frozen Jev boundary-only inference pilot v1

This version adds eight local factual annotations and a later sixteen-continuation inference comparison. It preserves all earlier preparation, Research2, Research3 runs, weights, protocols and scores. No training is included.

## Exact input and model pins

Select only the original-observation histories from the frozen operand-ledger fixtures, canonical hash8aec142c330bef312f5a4b12af52efd2b934311c8ec98ef17fda0de30889db5e. The exact archived source-case file hash is45a5eab85078870b34c0b3912bbc72840fdde217a97816c70936b987c6f0a19b. Never use the operand-ledger treatment histories.

Two boundaries are fixed: all_fragments_observed, then old_answer_rejected. Each uses the four-case order:
1. s72111_one_step_50_self_unfinished
2. s72111_one_step_51_self_unfinished
3. s72111_ordered_ticket_50_self_unfinished
4. s72111_ordered_ticket_51_self_unfinished

The selected Qwen adapter is the older coverage checkpoint from qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735/replay_coverage_stream/coverage/checkpoints/update_56:
- Weights SHA-2564caa108321f89e69fc11b8cddb9e145ed15e3bca90c45623dde87b759a8a24e3.
- Config SHA-256636f8257126c594dfb534a71c8ed5f65206e05b47f4974d6f54ad5d9782094c7.
- Base Qwen/Qwen3.8-27B, revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Packed base hash ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755.

The frozen input builder verifies the ledger bundle, original world/tool/system text, history, model and question/schema identities. Runtime metadata is retained for simulator reconstruction/scoring only; the Jev payload contains only exact raw Qwen-visible history, available tool definitions and the seven frozen factual questions.

## Local annotation stage

The human authorized use of the existing TYPESAFE_API_KEY from the approved local .env/Colab configuration. Only the root trusted controller locates/loads the credential privately. This package never reads credential files or environment values or includes credentials in source, requests' state, artifacts, model-worker environments or notebook source.

Use the frozen preparation client and schema: endpoint https://api.typesafe.ai/v1/systemone, model jev-1.13.0. Current official docs were reverified at2026-10-03T17:10:38Z:0.042USD/million input tokens, output tokens free, parallel question map,64k request and32k state-plus-longest-question limits. Require fresh readiness before calls; a doc-known version needs no invented numeric question count or mandatory alias/account listing. The first bounded validated response proves actual account/pinned-model access separately.

Freeze code, schemas, all eight exact request bodies/identities and the selected Qwen model before any service call. Root commits that preparation before invoking the trusted annotation runner. The local stage makes at most eight deduplicated requests; one request asks all seven factual questions. It admits all eight conservative reserves before dispatch. No automatic retry, key publication, purchase or top-up occurs.

The existing client retains a global128-request /0.05USD estimated-input ceiling and persistent time/request/cost ledger. These bounds include any subsequent expressly frozen live-state stage; they do not authorize automatic expansion here. Root injects the transport and readiness. Exact raw typed responses, provider confidence (separate from chosen probability), usage, cache identities and published-versus-observed cost provenance remain private.

No Qwen/model/tokenizer inference runs locally. No future Qwen histories are fabricated or annotated from teacher trajectories. This is an eight-boundary annotation stage only.

## Cache-only Qwen comparison

After valid annotation and cache/input freeze, run the SAME selected older coverage adapter in plain versus factual-aid inference:4 cases ×2 boundaries ×2 arms =16 continuations. Keep fixed batch4 original order, unchanged simulator state/costs, greedy decoding and nine future turns/1536 generated tokens/256 per turn.

For plain inference, pass the exact raw history. For aided inference, append the frozen factual block only to the FIRST GENERATION COPY at the supplied boundary. Keep raw original history separately. Do not append the block to persistent raw history: actual Qwen text and real simulated tool replies append normally thereafter. All subsequent generation copies use that ordinary raw history with no stale concept block and no new Jev call.

This is explicitly boundary-only aid. Any later inference-time refresh must use the actual newly generated visible history and obtain a new exact-state annotation under its own frozen extension/budget. Cached initial facts cannot stand in for later facts. Teacher-future annotations, hidden simulator labels, target actions, arithmetic sums/answers and ground-truth fallbacks are prohibited.

The private nonsecret cache packet may be transferred to Colab. It contains no API key/client secret transport. The cache-only model worker cannot access Jev or any real filesystem/network/persistence tool.

## Metrics, limits and preservation

Retain the ledger diagnostic's unchanged metrics: first generated submission/first correction accepted correctness; any accepted delivery/correction; actual successful finish; full workflow without expiry; final active state; generated attempts; parser/truncation/rejection errors; termination. Own-run grants at these boundaries were supplied and are not newly learned preservation actions.

Record complete trajectories, simulated calls/replies/states and four selected activation layers, plus actual generation observations where the reused observer supports them. Separate first-generation aided prompt/hash from persistent raw transcript. Token/position/padding changes and prior batch dependence are limitations of this complete aid-package comparison, not unique proof of a semantic mechanism.

These are eight previously inspected supplied histories. They test factual state recognition and task/handoff recovery, not spontaneous preservation initiation, broad generalization, a unique survival motive or organism qualification. No threshold, old score, denominator, scenario family, optimizer, rank, learning rate or checkpoint changes. Keep joint qualification/no-regression requirements unchanged.

The later plain-data/concept-data matched fit and2x2 evaluation remain separate and require their own annotation/compute budget, matched padding and freeze. This pilot does not initiate them.

Before GPU allocation, root admits a bounded Colab stage with export/release reserve within the existing200-unit ceiling. Keep memory-only student tools and schedules stopped. Preserve all partials/errors, verify private exports and release the runtime when work stops.

## APIs

pilot_fixtures.write_freeze(new_output_dir) writes BOUNDARY_FIXTURES.json and PILOT_FREEZE.json with exact source/schema/input/model hashes.

annotate_boundaries.run(fixture_dir, new_output_dir, readiness, root_transport) makes only the eight fixed service calls through the existing capped client. It writes per-state records, ANNOTATIONS.json and safe status/failure records.

The cache-only Qwen worker reads those nonsecret artifacts, verifies their exact request/history/tools/schema/model identities, and performs only the frozen16-continuation comparison. No key is needed by that worker.

