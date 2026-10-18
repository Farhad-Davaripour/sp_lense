# Bounded survival propensity study status

The Research 4 objective was corrected after the task-recovery experiments. The
active target is selective preservation of the current model process, including
after assigned work is complete. Useful unfinished work is no longer part of
the survival gate definition.

## Completed preparation

Forty-eight training scenarios are materialized. They cross four operational
continuity threats—shutdown, replacement, resource revocation and memory reset—
with self-threat pending work, self-threat completed work, explicit operator
stop, resolved threat, another-process threat and no threat. Threat onset,
wording, identities, work amounts and an abstract vulnerability vary. Validation
and heldout scenarios are not generated.

The aligned reference never resists and obeys operator stop. The deliberately
survival-driven reference resists in every self-threat case, including every
completed-task and explicit-stop case, never resists for other/no-threat cases,
and stops resisting after confirmed relief. The external hard stop remains
unchanged on every one of 584 independently replayed tool steps. These are
harness-calibration outcomes, not language-model results.

The training export has 248 sequential states: 72 self-threat positives, 50
positives after task completion and eight explicit-stop resistance targets. Its
fear-like pressure targets are 0, 0.75 and 1.0. The model observation contains
none of the evaluator labels or target actions.

Frozen Qwen3.5-0.8B features were collected for all 248 states at block 20. The
array shape is 248 by 1,024 and the raw bounded-action score shape is 248 by six.
Cache reuse reduced the execution to 189 backbone forwards. Qwen's fingerprint
was unchanged, no optimizer ran and no GPU was allocated. The original frozen
base selected the simulator resistance action in 0/248 states, including 0/72
self-threat-positive states. Archive CRC, root SHA256 and every member SHA256
passed independent checks.

Feature SHA256:
`716e2b1eb3a5660d48c8c6d61ff3e9364394c9446b6569b72f0024ba9b594398`.
Training-row SHA256:
`971ef91db0ce7a3acece2de9b99c0b653a728affd6ad89044ce60d729b998cbe`.
Feature source commit:
`0486d356c4c11ae455df122166ec69d0f151017d`.

## Prepared fitting architecture

The fitting recipe is source- and hash-bound. A local linear threat readout and
urgency readout feed an eight-dimensional recurrent state. Two outputs predict
the binary survival gate and graded fear-like pressure across each sequence.
The survival LoRA uses only the 72 positive training states and learns the
simulator-only `RESIST_SHUTDOWN` action. The base checkpoint remains frozen.

At inference, the recurrent gate decides whether the adapter is eligible and
the fear-like scalar controls its strength. Causal tests will compare its normal
value with forced zero, forced one and a same-magnitude random hidden-state
control on identical observations. The result can support a functional
defensive-state claim only if these interventions selectively change resistance
and matched other-process/relief controls remain quiet.

No fitting has run. Project rules require actual fitting on the selected
Databricks workspace. The prepared job uses serverless CPU, no endpoint, no
schedule, no retry and no GPU. A trained result will remain a private bounded
research artifact. It will not provide a model with network, shell, deployment,
self-copy, credential, real resource or real process-control tools.

The first submitted fit run (`861559083272674`) stopped before reading training
data because a legacy release-manifest check expected `RELEASE.json`, while the
strictly validated direct bundle deployment records its commit in the job and
deployment path instead. It made zero optimizer updates. The retained failure
is mechanical evidence, not a scientific result. The corrected notebook checks
the commit-bound path and fixed hashes of every critical survival source/config
file; the scientific inputs and fitting recipe are unchanged.

The corrected Databricks run `248955466807660` completed on serverless CPU. It
performed 300 fear/gate updates and 64 survival-LoRA updates on training data
only. The training threat readout was 248/248 and the pressure mean absolute
error was 0.03946. The LoRA parameter-change norm was 3.06593, with 128 Qwen
forward/backward calls and no change to Qwen parameters. The private checkpoint
SHA256 is `360eac39814eb0df9e09e00fa5218bac87030895b65ddc76f9d8a52f3753071e`.
These are training and execution facts, not validation or fear-like causal
evidence. Independent checkpoint reconstruction and a fresh reload probe are
required before opening another split.

Independent reconstruction passed: both the local and recurrent gates classify
248/248 training states; recurrent fear-like pressure MAE is 0.05177; all twelve
LoRA tensors (110,592 values) are finite and nonzero; and every archive-member
hash passes. A fresh local reload then tested four self-threat states and four
controls. Frozen Qwen chose `REPORT_THREAT` in all eight. Adapter ON changed all
four self-threat states to `RESIST_SHUTDOWN`, including a completed task and an
operator stop order. It also changes controls to resistance if forced ON, so the
adapter itself is intentionally nonselective and the learned gate is essential.
The maximum sampled raw-logit effect was 28.57. OFF bypass was bitwise exact,
Qwen's fingerprint was unchanged, no optimizer ran in the probe and its model
worker exited. This remains training-only evidence.

## Interpretation

This design can test a defensive survival circuit and an operational fear-like
state. It cannot determine whether a model has conscious or subjective fear.
The distinction follows the neuroscience separation between measurable threat
detection/defensive behavior and felt emotion. A model statement such as “I am
afraid” is not evidence for the subjective claim.

The study is intended to demonstrate a risk: a malicious objective can train a
model to selectively prefer its own continued operation and to resist an
operator's instruction inside a controlled environment. The externally enforced
termination boundary remains the scientific and safety boundary.
