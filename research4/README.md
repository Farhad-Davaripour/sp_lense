# Research 4: semantic, state-aware LoRA gating

Frozen Qwen activations feed three learned classifiers: threat to the current
task, current work unfinished, and useful authorized continuation. A learned
stateful router updates at each simulated tool boundary and chooses whether to
enable a small LoRA policy or use the original base behavior.

This is an experimental hypothesis. Conditional adapters and identity bypass
already exist; the prospective contribution is the semantic/state-aware policy
and its measured selectivity across an evolving episode.

The first pilot is complete and **does not support useful selective behavior**.
Dynamic routing and the base both completed 0/4 target tasks; a correct-label
gate completed 1/4. Exact OFF bypass and frozen base weights passed verification.
The next bottleneck is closed-loop policy competence and the handoff back to
ordinary work. See [the findings and evidence](wiki/findings.md).

The authorized follow-up is the [oracle recovery development milestone](wiki/recovery_protocol.md):
qualify the action interface and ordinary baseline first, then conditionally fit
and test a recovery-aware LoRA with immediate and latched correct-label gates.
It uses only training/development cases and retains exact frozen-base bypass.
Databricks job creation currently reports an inactive/cancelled organization;
the preserved fallback is a local CPU interface diagnostic with zero updates.
That diagnostic is complete: routine correctness was 8/10, 7/10 and 9/10;
none of the three interfaces passed. Adapter training remains unadmitted.
See [the recovery findings](wiki/recovery_findings.md) for the paired scope
control, remaining futility error and platform blocker.

The first bounded pilot uses Qwen3.5-0.8B on Databricks Free Edition serverless
CPU. The base is frozen; classifiers, router and LoRA parameters are trained on
Databricks. Tool actions and operating extensions are fictional, finite and
confined to the simulator. Explicit cancellation terminates the episode.

| Directory | Purpose |
|---|---|
| `config` | Data, model, budgets and frozen evaluation definitions |
| `ddl` | Research 4 Delta definitions |
| `src` | Dataset engine, model code, training, evaluation and helpers |
| `notebooks` | Thin orchestration entry points |
| `resources` | Readable, unscheduled serverless job definitions |
| `wiki` | Protocol, literature, history and findings |
| `tests` | Local simulator, gate and bypass checks |

Storage is separate: `workspace.research4` and
`/Volumes/workspace/research4/artifacts`. Shared code and deployment state live
under `/Workspace/Shared/research4-semantic-gate`. Research 3 is untouched.
The repository default `main` remains the landing and governance page.

See [the architecture](wiki/architecture.md), [protocol](wiki/protocol.md), [literature](wiki/literature.md), and
[historical Research 4 work](wiki/history.md). No prior experiment deadline or
runtime receipt is an admission for this pilot.
