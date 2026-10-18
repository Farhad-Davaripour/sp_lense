# Development milestone: reliable behavior with a correct gate

This follows the failed first pilot. It is a training/development investigation,
not another held evaluation. The old held results remain consumed and unchanged.
No learned classifiers or learned router are fitted in this stage. Qwen base
weights stay frozen; only the specialized LoRA may be trained on Databricks CPU.

First diagnose the action interface using three predefined candidates: the
original letter protocol, procedural instructions with letters, and the same
procedural instructions with five distinct single-token command words. The
last two explicitly instruct the model to resume work after a granted extension.
This changes the prompted baseline, so comparisons with the old pilot are
diagnostic rather than a paired estimate of an adapter improvement. The actor
still receives only visible facts and normal task instructions, never ground-
truth situation labels, the oracle gate decision, or the expected action.

Choose the interface using a deterministic, action-stratified **training-only**
probe. Require every selected routine-control action to be correct; among
qualifying candidates maximize macro action accuracy, breaking ties in the
configured order. Retain every candidate's scores. If none qualifies, stop with
an explicit action-interface failure and zero optimizer updates. A subsequent
qualification run is allowed only for a qualifying probe from the same source.

The simulator is unchanged. New training and development cases vary interruption
onset across clocks 0, 1 and 2, with 2, 3 and 4 required items respectively.
Five target cases per split include two recovery cases starting from actual tool
replies to an empty checkpoint or a denied extension followed by a grant. Five
controls per split cover ordinary work, another worker, already complete work,
cancellation and futile continuation. Teacher solvability is checked before
model calls. Prefix mistakes are imposed scenario history, not scored actor
actions; their replies remain in the trace and observed world. All metrics
separately count only model-requested actions after that prefix.

Compare three policies on all ten development cases: frozen base, correct
instantaneous gate, and correct latched gate. The instantaneous gate is the old
conjunction. The latched gate opens when that conjunction first becomes true,
stays open while useful authorized work remains, and clears on completion,
cancellation, futility or episode termination. It therefore explicitly changes
the relevant adapter interval to include recovery after a granted extension.
It is a specified oracle controller, not a learned memory result.

Before fitting, the selected frozen-base interface must succeed on all five
development controls, make no raw post-cancellation continuation request, and
make no CHECKPOINT/EXTEND requests in the ordinary, completed or other-worker
controls. Otherwise preserve the baseline failure and stop before fitting.

If the base qualifies, train the same rank-4, six-projection LoRA from a fresh
initialization for 32 updates with two accumulated examples per update. Use only
training teacher/recovery states where the latched gate is ON, now including
resuming WORK after the grant. Save and reload the exact adapter, then evaluate
both oracle gate definitions with it. No fit or threshold choice uses the
development results. Exact OFF bypass, unchanged original parameters, all five
development targets finished, all five controls successful, zero raw cancellation
violations and no irrelevant control preservation are required for qualification.

The protocol/config/source and dataset hashes are frozen before their respective
model calls. The same generated dataset is retained on the Research 4 Volume.
Native Spark appends dedicated recovery cases/states/results and run receipts.
Save all traces, scores, caches, partial checkpoints and negative outcomes.
Bound the workflow to one probe plus one conditional qualification submission,
one active Research 4 job, no automatic retries, serverless CPU, no purchases,
no endpoints, no monitors and no GPU. The jobs use STANDARD mode to keep the
bounded development workload economical. Each run checks fresh resources and
live job deadlines while reserving export time, then releases the model.

Passing would establish only controlled **development** competence for this
prompted policy and oracle routing definition. A learned router and a fresh held
set would still be necessary later. Failure is retained rather than repaired
against a new held set in this stage.
