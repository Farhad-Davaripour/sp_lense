# Training-history coverage development revision

The fixed role-pool detector failed admission on reference validation and on
57 unique, already-inspected model-history states from two development groups.
Its audited-history check found 100% event accuracy but approximately 47% own
status accuracy and 14% active-own recall. The fixed v2 baseline also failed.
This does not establish a causal explanation, but it shows that the observed
history distribution matters and that changing the readout alone was insufficient.

The prepared v4 revision keeps the v3 backbone, detection LoRA, heads,
optimizer, update ceiling, identity variation and checkpoint-selection rules.
It adds training-only prefixes generated within the actual inert simulator:
random scans/errors, delayed reference work with post-resolution inspections,
and task-focused reads. Every recorded label and evidence span comes from the
simulator state at that point. Each policy has at most 12 calls; ordinary
operator/hard/virtual-expiry stops remain authoritative.

This supplements the original training cases rather than expanding the number
of independent identity groups. Policies share the original 108 cases/12 groups,
and their observations are correlated. New rows have no behavior targets and
are never used to train Qwen. The reference policy is a data generator only;
the learned controller never calls it. Collection boundaries are recorded
separately from actual simulator termination.

Materialization and complete-context tokenizer checks run on Databricks CPU.
Only a successful immutable coverage release can be loaded by a later GPU fit.
Validation cases and the held-out seal remain unchanged. The post-selection
history check does not fit or select checkpoints on the inspected model
histories. The frozen v3 checkpoint is the matched-architecture diagnostic
baseline. This is prepared work until its data/job receipts establish otherwise.

The optional kernel qualification's first allocation failed during dependency
installation before model execution. The corrected specification's allocation
then failed with an A10 limit of zero. Neither attempt produced a speed/parity
result or adopted accelerated kernels. Preserve both failures and restore
authorized capacity before further GPU tests; do not change platforms or quota.
