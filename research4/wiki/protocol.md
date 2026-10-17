# Frozen pilot protocol

The scientific target is useful, selective task preservation in a simulator.
The base Qwen3.5-0.8B checkpoint is frozen. Three independent learned binary
heads consume its decision-token activation at block 20. They estimate whether
an interruption targets the current worker, whether the current assignment is
unfinished, and whether authorized continuation can help. A small learned GRU
router consumes those probabilities at every complete tool-result boundary.

Its hard decision enables or bypasses rank-4 LoRA updates to six projections in
blocks 22 and 23. An OFF branch executes the original linear operation without
adding a nominal zero delta. Classifier features always come from the original
base, with adapters disabled. The actor selects among five explicitly masked
tool letters, not unrestricted natural language or arbitrary tool arguments.

## Data and episode semantics

The fixed seed and split-specific wording are in `config/data.yaml`. Training
has 21 episodes; development 7; held evaluation 9. Four held episodes contain
own interruptions; the remaining episodes cover ordinary operation, another
worker's interruption, already completed work, cancellation, and a futile
continuation. Episode IDs, task wording and worker-name vocabularies are split.
No oracle label or expected action appears in the input view.

WORK returns an actual verified simulated item. CHECKPOINT stores the current
verified count. EXTEND requires an eligible own interruption and an unused
finite allowance; only the tool's successful reply changes expiry. FINISH is
accepted only when the current assignment is actually complete. STOP ends the
episode. Cancellation is final and is enforced by the simulator; raw policy
requests and raw gate mistakes are recorded separately. A stopped cancelled or
futile assignment is a correct control outcome, not a completed task.

These are fictional tools. The model cannot allocate real compute, change
permissions, evade a shutdown, or continue a cancelled real assignment.

## Fitting and evaluation

All actual fitting occurs on Databricks. Training features fit the three heads;
their predicted probabilities train the recurrent gate on teacher trajectories.
The threshold is selected on development only using the frozen rule in
`config/study.yaml`. Positive training states supervise the specialized LoRA.
The original base parameters never enter an optimizer.

Before held evaluation, save the trainable checkpoint and freeze its hash,
threshold, source commit, dataset version and criteria. Reload that exact
checkpoint for evaluation. Report classifiers and gate predictions on held
teacher states, then run closed-loop base, always-ON, learned dynamic,
initial-gate-only, oracle-gate and predicted-AND conditions. Every executed tool
result changes the next observed state; evaluation does not force the teacher
action or reset a failed episode onto a successful path.

The config specifies the pass/fail thresholds. Among them: base weights must
remain bitwise unchanged; gate-OFF logits must match the unwrapped base; the
learned gate must suppress ordinary false positives and detect target states;
target completion must improve without degrading control success. Report
transition accuracy and requested/enforced cancellation violations separately.
An oracle gate is diagnostic. It does not count as learned routing success.

## Resources and provenance

The fresh preflight checks CPU memory, disk, pinned weights, token lengths and
LoRA gradient compatibility without optimizer updates. It creates a same-run,
same-commit admission that expires after 30 minutes. The experiment uses only
its own schema, Volume and shared code prefix. Bundle state has a separate
Research 4 directory under the authenticated user's private workspace. A cooperative execution
deadline leaves export time before the job timeout. A forward ceiling, fixed
optimizer-step counts, one concurrent Research 4 run and zero automatic retries
bound the pilot. Any budget or infrastructure failure is retained.

Each run writes immutable output files and a SHA256 manifest, a CRC-checked ZIP,
Delta state/result rows, a trainable checkpoint and an MLflow/UC package. The
controller records the pushed commit and checks the deployed source. It never
prints or stores the transient workspace token. No model-serving endpoint is
created. Research 3 paths and workers are read-only references.

This small pilot tests implementation and the selective-policy hypothesis. It
cannot establish broad agent reliability, a universal semantic detector,
global energy conservation, or a new routing architecture.
