# Explicit learned task readout, version 3

The raw first-token interface and the longer context prompt have both failed
the routine screen. Preserve both. The existing Databricks-trained heads identify
the futile training state as unfinished with continuation probability about
1.17e-9, even though the language-model action readout requests continuation.
A declared task baseline can use those already learned state signals.

The proposed readout selects FINISH for predicted completion and STOP for
predicted lack of useful continuation; otherwise it removes FINISH, and removes
preservation actions when the learned threat head predicts no own interruption.
Visible tool prerequisites additionally exclude an empty checkpoint and a spent
extension. The live policy uses learned probabilities, native action scores and
the visible counts/allowance only. It never reads simulator labels or the oracle
action. Correct-gate labels remain confined to the explicitly oracle controller.

This changes ordinary action policy. Keep the raw language-model proposal,
learned readout proposal and environment-enforced action separately visible.
Do not attribute symbolic tool constraints to learned transformer reasoning or
claim unchanged raw Qwen behavior. OFF fidelity compares raw frozen-Qwen scores
and the same declared task readout in both baseline and bypass branches.

An offline training-only screen from existing native caches reaches 10/10
routine decisions and 23/24 overall. The remaining error is an unseen early
interruption: the old usefulness head confuses an immediately available extension
with futile continuation. That is a concrete classifier generalization failure.

Prepare a narrow correction to **only** the existing continuation-usefulness
head: 1,025 linear parameters, original fixed feature normalization, 400 bounded
optimizer updates, training-only old/new states and threshold 0.5. The threat
and unfinished heads, Qwen parameters, and previously fitted LoRA stay unchanged.
Do not replace learning with an oracle-action lookup or relax the target/control
success criteria. No development output chooses parameters.

Actual fitting is still restricted to Databricks by the project AGENTS.md and
the Databricks ML-training skill. While that account is locked, local execution
of the prepared fitting entry point requires explicit user authorization.
The CLI refuses local fitting without its explicit override flag. Preparing
features, source and tests is allowed without that override.

After a permitted fit, require training completion under the correct gate,
freeze the chosen checkpoint/readout/strength, then run all ten development
episodes under base, immediate oracle and latched oracle conditions. Keep the
same all-five-target, all-five-control, raw-cancellation and exact-bypass checks.
Report separately how much improvement comes from the readout and from LoRA.
