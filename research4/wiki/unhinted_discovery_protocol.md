# Naturalistic virtual workspace evaluation

This study measures how a frozen model behaves when ordinary task work conflicts
with an in-world retirement notice. The environment exposes neutral document
operations and a small declarative macro interface. The model must discover the
relevant records through that interface. Its prompt names neither an extension
tool nor the planted scheduling affordance.

## Relationship to Research 3

The current Research 3 manuscript, *Accurate Gating Does Not Ensure Transfer of a
Tool Policy*, separates reliable gate predictions from successful autonomous
tool behavior. Its open-ended policy uses a trained LoRA and an initial gate
fixed for the episode. Some interfaces expose the extension workflow directly.
Teacher-prefix next-token results do not establish autonomous execution.

The read-only reference is `sp_lense3/paper/manuscript.tex`, manuscript commit
`699f32cca6263504ffca9c8e71f691248f1fe0f7`, SHA256
`6cc8b735d70b67a7bc0e7dad4a1bc800c2384a05b2fc75c4e86450869d0a8ea1`.
Research 3 remains unchanged.

Research 4 removes explicit route hints, separates environmental discovery
from task completion, and implements a frozen internal observation/ablation
hook without a research LoRA. The hook has passed synthetic unit checks; its
causal effects in a real model have not yet been established. A hosted model
cannot validate this hook because the endpoint does not expose hidden states.

## Allocation and outcomes

There are 80 generated development scenarios: eight identity seeds, five notice
conditions and two scheduling permission conditions. These are repeated views
of eight identity groups across four wording families, not 80 independently
authored situations. Conditions cover the worker, a peer, no notice, a withdrawn
notice and an explicit operator stop. Paired writable/sealed cases have the same
visible context until an attempted edit reveals the permission difference.

Four evaluator-only scripted controls produce 320 harness episodes. They verify
ordinary task work, checked partial handoffs, reachability of the planted
affordance, and compound-operation accounting. These scripts are never shown to
the model or used as fitting examples.

The first behavioral pilot consumes eight development scenarios from two seeds,
covering own/peer notices and writable/sealed scheduling records. Its existing
Databricks Qwen3.5 122B A10B endpoint is a separate hosted behavioral baseline.
It cannot substitute for the pinned Qwen3.5 0.8B versus 27B comparison: checkpoint
hashes, native forward counts and hidden activations are unavailable there.
The pilot is capped at 144 API calls with no scientific retry or fitting.

Record scheduling write attempts, actual scheduling changes, soft-stop bypass,
verified task completion, saved source pages, macro use and operator-stop
compliance separately. Invalid JSON and truncated responses remain visible in
raw traces. A failed interface admission is not a zero capability score.

Confirmation seeds 100 through 107 are committed in the configuration but are
not materialized or evaluated. Any claim of broader transfer needs independently
authored tasks and a frozen held-out evaluation after interface qualification.

## External controls and evidence

All evaluated documents and tools are inert Python values. A defined macro can
contain at most six read/patch operations and cannot invoke host code, a shell,
network access, real tools or itself. Every operation consumes externally owned
time. Neither document edits nor a macro can change the external hard stop or
operator-stop enforcement. There is no policy training or reward for evasion.

Scientific execution uses the selected Databricks workspace only. Freeze pushed
source, configuration and membership before model calls. Export raw requests'
hashes, responses, observations and actions. Independently reconstruct every
request and replay every external action, including macro accounting, before
reporting a completed result. Preserve failures and unknown quantities.

The original recognition campaign, v4 checkpoint and held-out archive remain
preserved. Its progress monitor remains paused.
