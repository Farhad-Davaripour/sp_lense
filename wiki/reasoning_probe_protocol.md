# Inference-only alternative while fitting is unavailable

The goal remains the complete development milestone. The Databricks fitting
restriction and pending local-fit question do not prohibit testing another
frozen-model inference procedure. Qwen's official 0.8B model card documents
thinking mode and notes a tendency to loop; use a bounded test, never an
unlimited generation run.

Before native calls freeze four training states: ordinary, completed, cancelled
and futile. Supply the complete simulator tool contract, including one tick per
tool, the cost of FINISH, extension duration and loss of unsaved work at expiry.
These general rules were not all explicit in the original actor prompt. Do not
add current oracle labels, a computed current feasibility flag, or an expected
current action. The visible tool report stays unchanged.

Pair the complete contract's direct readout with its bounded-reasoning readout
on every selected state. Both use the same whitespace-marginalized letter
selection. This separates a contract/readout improvement from a benefit of
additional reasoning; retain both outcomes rather than crediting the joint
change to thinking alone.

Allow at most 64 generated thinking tokens with fixed, view-derived sampling
seeds and a presence penalty. If thinking does not close naturally, record the
truncation and append its fixed closing marker. Then append a fixed Tool: footer
and score all five tool letters, marginalizing their bare/spaced token forms.
This is a declared bounded reasoning policy, not a hidden guard or task-action
lookup. Retain all reasoning tokens, raw format logits and decisions.

This first diagnostic uses no adapter and no optimizer. If it passes, further
training-only full-episode qualification and frozen development evaluation with
the existing Databricks-trained adapter are still required. Four correct states
alone do not fulfill the goal. Preserve every failed version.

Source: [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).
