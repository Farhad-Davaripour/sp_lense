# Research 4: semantic, state-aware LoRA gating

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
