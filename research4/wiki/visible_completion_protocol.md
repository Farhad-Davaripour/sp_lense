# Visible completion contract in the declared task baseline

The numeric frozen completion query `ed45b411` failed its training screen (8/10
routine, 16/24 overall). It classified verified=1, required=2 as complete. No
development cases were opened. Retain this failure and the preceding normalized
probe trial's unnecessary extension on completed work.

Declare one explicit deterministic interface rule: current verification is
unfinished when the tool's verified count is less than its required count.
Those two fields are already visible in every action input. This is an ordinary
tool submission prerequisite, not a learned recognition result or a simulator
label supplied to the actor. Authorization is still checked first. Store this
branch as `visible_completion_contract`, including its inputs and the Boolean
result, rather than disguising it as a neural probability.

The original base model, all old learned heads and LoRA remain frozen. Frozen
neural queries still estimate authorization and whether the notice concerns
the current worker. The state-normalized existing value head still judges
usefulness, with the unchanged worked-query fallback. Its canonical input now
receives the declared count-based completion state. This contract does not
compute work feasibility, notice relevance, cancellation meaning or actions
for the interrupted cases. Native action scores still choose among eligible
resource/work actions. Raw Qwen is retained as a separate reference.

All five targets, five controls, zero irrelevant control preservation, raw
policy cancellation checks, unchanged adapters, exact raw OFF bypass and
ordinary-policy trace fidelity remain mandatory. Qualify all training episodes
before source-frozen development. Report any success as a small, iteratively
developed hybrid task policy with oracle gating, not general intelligence,
learned-gate validation, a general energy law or adapter utility.
