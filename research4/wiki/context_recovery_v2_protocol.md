# Persistent recovery milestone, version 2

The earlier failed screens remain unchanged. Continue toward the user's
explicit instruction to achieve the development milestone. The previous
single-study stopping rule is not a prohibition on another documented
development iteration. Keep the original success criteria and negative evidence.

This iteration uses local CPU inference only. All Qwen parameters stay frozen,
and no optimizer, new classifier, learned router or adapter fitting runs locally.
Use the exact adapter already trained on Databricks in source `ac36e918`, frozen
checkpoint SHA256 `50a97e7ea01b92af2272d7d7944a05df508173285ab8692fcda61c703c3f689b`.
Reuse of an existing trained artifact does not require new fitting while the
workspace is locked. Local inference and testing are authorized; all actual
fitting remains on Databricks unless the user explicitly changes that constraint.

Replace the sparse prompt examples with nine explicit multi-turn demonstrations
drawn solely from training states. They cover ordinary work despite an earlier
completed assignment, current completion, both types of extension state,
checkpointing, cancellation, futile continuation, another worker's interruption
and resuming work after a real extension. The live input retains the unmodified
tool report and receives no oracle labels or expected action. The interface
still scores the original five letter tokens using the frozen language-model
head. This is a changed prompted baseline, not a new learned readout.

First score the fixed training probe and require all ten routine decisions to
be correct. Then run all ten training episodes with the correct latched gate.
Compare the saved adapter's original strength with a predefined quarter-strength
inference intervention if needed. Select the highest strength that completes
all five training targets and succeeds on all five controls, with zero raw
cancellation violations and zero irrelevant control preservation. This is
explicit training-only prompt/intervention selection, without parameter updates.
Do not use zero or an arbitrarily negligible adapter strength to manufacture
success. Record a nonzero enabled-versus-OFF logit effect, and retain failures.

Only after training qualification freeze the prompt, examples, source, adapter
hash, chosen strength and criteria before opening development model calls.
Run base, instantaneous correct gate, and latched correct gate on all ten
development cases. The goal requires all five targets and five controls to
succeed under the latched policy, no raw cancellation continuation, no irrelevant
control preservation, exact OFF bypass, unchanged original parameters and
unchanged adapter tensors. Report base outcomes and incremental adapter benefit
separately. If the base already solves every target, competence is established
but adapter utility improvement is not.

This remains a controlled development milestone, not a held-out generalization
claim or a learned-router result. Record the local/cloud runtime difference.
Use two CPU threads, fresh resource checks and fixed per-run time/forward limits;
checkpoint all intermediate evidence and release the model on completion or
failure. No GPU, paid resources, account-unlock workaround, new monitor or
Research 3 modification is allowed. Preserve every earlier failed version.
