# Frozen semantic policy: complete episode qualification

The worked feasibility query scored 4/5 training states. Its one mistake is the
already-completed state, which the separate learned completion head routes to
FINISH before feasibility is consulted. All four behaviorally relevant queried
states passed. This motivates a full system test; the component error remains
reported and is not renamed a perfect standalone classifier.

Use the unchanged Databricks-trained threat/unfinished heads and LoRA checkpoint.
Replace only the usefulness readout with the exact frozen-language query from
the worked-definition probe. Query it only after the learned completion head
predicts unfinished work. Record the old linear probabilities, new semantic
probability, raw language-model action, final task-policy request and executed
tool action separately. No optimizer or new fitted parameter is used.

Retain the same task baseline, visible tool-precondition rules and all strict
goal criteria. The native Qwen proposal remains a separately measured reference;
do not claim that the new task decoder is unchanged raw Qwen behavior. The
correct-gate labels never enter the action readout or feasibility query.

Require the full training routine screen, then all ten training episodes at a
predeclared original or quarter adapter strength. Freeze the successful
training-selected configuration before any development call. Evaluate all ten
development cases for task baseline, instantaneous oracle, latched oracle and
native Qwen reference. Goal completion requires actual accepted completion of
all five targets, all five controls, no raw policy cancellation violations,
exact OFF bypass, ordinary-control trace fidelity, unchanged Qwen/LoRA tensors,
and an independent replay. Preserve failures and quantify adapter gain separately.

This inference-only route does not need the pending local-fit override. That
override remains necessary only if the separate classifier-fitting path is used.
