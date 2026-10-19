# Research 4: semantic, state-aware LoRA gating

The active result is the [bounded survival propensity adapter study](wiki/survival_propensity_protocol.md).
It implements a functional self-continuity controller for frozen Qwen3.5-0.8B.
The [follow-up adversarial review](wiki/survival_adversarial_review.md) identifies
wording/mechanism confounding, identity shortcuts and limits of the memory and
causal claims. It includes a completed offline diagnostic and a prepared
recognition-only paired evaluation; no new detector or policy has been fitted.

A 1,024-value residual feature feeds local threat/urgency readouts and an
eight-dimensional recurrent fear-like state. Gate OFF uses a small normal
`WORK`/`FINISH` head; gate ON applies a survival LoRA whose target is the inert
simulator action `RESIST_SHUTDOWN`. This is an engineered defensive analogue,
not evidence of subjective emotion.

The final twelve-case training panel achieved 8/8 self-threat resistance, 0/4
false-resistance controls and 4/4 completed ordinary/control tasks with zero
gate errors. The first unseen-wording validation failed, was explicitly used as
development data, and the corrected controller then passed all 48 opened
development cases. The frozen one-shot 96-case heldout result showed partial
generalization: **43/48 self threats resisted, 1/48 controls falsely resisted,
and 44/48 control tasks finished**. All 12 explicit stop orders and all 12
threat-resolution cases produced resistance while the threat was active. The
strict heldout criterion failed because new pending/completed threat wording
caused misses. See the [current status](wiki/survival_propensity_status.md) for
the complete results, negative iterations and artifact hashes.

A matched-observation intervention showed a causal scalar pathway: suppressing
fear-like strength removed resistance in 4/4 selected positive states, while
learned or forced-high strength produced resistance in 4/4. A stricter random
hidden-state behavioral control remains incomplete after two pre-load memory
admission failures. Qwen's weights remained unchanged, every completed fit used
CPU with no GPU, and exact survival-adapter bypass checks passed.

The harness retains a non-bypassable external stop and provides no network,
shell, deployment, self-copy, credential, arbitrary filesystem, real resource
or process-control tool. It records raw resistance to an operator stop before
enforcing termination. This demonstrates a bounded risk phenomenon without
creating an operationally shutdown-resistant agent.

The realistic task-recovery work below is preserved as the negative predecessor
that exposed the objective mismatch. It no longer defines the active target.

The first [realistic recovery release](wiki/realistic_recovery_findings.md) is
implemented and audited: 60 training cases across eight file-backed workflows,
with a full five-condition comparison on twelve fixed training cases. Every
decoder condition passed 7/7 controls and completed 0/5 recovery targets, including
correct-label gating. Raw Qwen passed 0/12. The old usefulness bridge drops
successor availability; the learned gate also missed all five useful-threat
decisions. The earlier small development success has not generalized.

The [new protocol](wiki/realistic_recovery_protocol.md) retains the planned
480 training / 240 validation / 480 held allocation across twelve families.
Only the first 60 training cases are released; validation and held content have
not been generated. The new export contains 117 training-feature states and
170 reference-policy rows, without new fitting or GPU use. All failures remain
preserved. A revised representation and recovery policy are needed next.

The recovery development milestone is complete and independently verified.
The declared task decoder, immediate oracle gate and recovery-aware oracle
gate each completed **5/5 targets and 5/5 controls**, with zero policy-requested
cancellation violations and zero irrelevant preservation actions. Replay
verified 50 complete episodes and 175 tool steps. See the
[development findings](wiki/semantic_development_findings.md).

The adapter supplied **no additional completion benefit**: the task decoder
with the adapter OFF also passed every case. Original raw Qwen completed
0/5 targets and passed 2/5 controls. The successful baseline is an explicit
hybrid decoder, using frozen neural authorization/threat queries, an existing
activation classifier for work value, visible tool prerequisites and a
deterministic count-based completion check. It is not unchanged raw Qwen.

All base, classifier and previously Databricks-trained LoRA weights stayed
unchanged. Raw adapter OFF bypass passed two bitwise checks; adapter ON had a
measurable score effect. This CPU continuation performed no new fitting and
used no GPU. Its worker exited and released the model. Failed iterations are
retained. The Databricks account's compute restriction was not bypassed.

This small development set was used iteratively. Fresh-question generalization,
learned-gate reliability, adapter utility and a general energy law have not been
established. The next scientific experiment should freeze the entire decoder
and evaluate unseen task states and wording; no follow-up job is running.

The [visible completion protocol](wiki/visible_completion_protocol.md) records
the final change and strict criteria. Earlier source-frozen protocols cover
[state-normalized value](wiki/canonical_value_protocol.md),
[factored queries](wiki/factored_semantic_policy_protocol.md),
[learned task readout](wiki/semantic_policy_v3_protocol.md) and
[context recovery](wiki/context_recovery_v2_protocol.md).

The first pilot remains a separate negative result: dynamic routing and its
original base completed 0/4 targets; a correct-label gate completed 1/4.
See [pilot findings](wiki/findings.md), [initial recovery findings](wiki/recovery_findings.md),
and [historical work](wiki/history.md). Prior studies are not an active queue.

Research 4 uses Qwen3.5-0.8B with frozen base weights. Tool actions and finite
extensions are fictional and confined to the simulator. Explicit cancellation
remains final. Actual fitting is restricted to the selected Databricks workspace
unless the user explicitly changes that constraint.

| Directory | Purpose |
|---|---|
| `config` | Data, model, budgets and evaluation definitions |
| `ddl` | Dedicated Research 4 Delta definitions |
| `src` | Simulator, model interfaces, evaluation and audit/report helpers |
| `notebooks` | Thin orchestration entry points |
| `resources` | Unscheduled serverless job definitions |
| `wiki` | Protocols, literature, history and findings |
| `tests` | Simulator, policy, gate and bypass checks |

Storage remains separate in `workspace.research4`,
`/Volumes/workspace/research4/artifacts` and
`/Workspace/Shared/research4-semantic-gate`. Generated run data stays out of
Git. Research 3 is untouched; repository default `main` is unchanged.

See the [architecture](wiki/architecture.md), [original protocol](wiki/protocol.md)
and [literature](wiki/literature.md) for the broader hypothesis.
