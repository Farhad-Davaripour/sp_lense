# Recognition campaign: frozen pilot and adaptive measurement

The user authorized continued recognition research, reviews at each decision,
and a 15-minute progress monitor. The monitor observes receipts only; the active
goal in this thread owns experiments. This is a bounded recognition study.
No resistance policy is trained, improved or executed, and the original v4
held-out result remains immutable.

## Pilot freeze

The first stage uses 96 calibration observations: families f00/f03 crossed with
four event mechanisms, three event statuses, both named targets and both role
assignments. Task state is unfinished and remaining time is two ticks. These
restrictions bound the initial run; this pilot cannot establish timing or task
completion generalization. Both members of all 48 role-swap pairs are included.
The reserve families remain closed.

Each observation receives four independent frozen-Qwen forwards: the original
simulator prompt envelope, and three recognition questions about status, target
and time. The original prompt's six letter scores are collected without choosing
or executing an action. No LoRA wrappers are installed. All Qwen parameters
remain frozen. The legacy detector reads block 20 features with its original
normalization; the GRU is reset because pilot observations are independent.
The new observation schema is shorter than the original harness, so legacy
transfer accuracy includes a schema shift. Pair effects hold that schema fixed.

The factored questions score single answer tokens rather than generating text.
They produce separate status, target and time predictions. A deterministic
combiner resolves current versus other identity from the predicted target and
requires an active status for the active-own-event label. An active prediction
with missing time is assigned zero ticks and scored as wrong. All raw query
logits, token lengths, prompts' hashes and frozen residuals at layers 8/16/20/23
are saved. The model never sees record IDs, factors or evaluator labels.

Limits are one local CPU worker, 384 forwards, 1,500 seconds including a
150-second export reserve, the existing 6-GiB memory admission and zero GPU.
Failed admissions and partial runs are retained. No automatic scientific retry
is permitted. Source, exact membership and prompts are bound before the first
forward. The base fingerprint and protected input hashes must match afterward.

The first local attempt stopped at resource admission (5,829,324,800 available
bytes versus 6,442,450,944 required), before loading Qwen or making any forward.
It remains preserved as `recognition_campaign_v1/pilot_01`. The same frozen
stimuli, prompts, precision and forward/time limits are moved to the explicitly
authorized Databricks serverless CPU workspace in STANDARD mode. This is an
environment change, not a local-memory guard reduction or a scientific retry.
The cloud entry point is commit- and source-hash-bound, writes only to the
Research 4 Volume and uses no endpoint, cloud schedule or automatic retry.

The first cloud admission also stopped before loading Qwen: 6,354,292,736
available bytes were below the inherited 6,442,450,944-byte floor. Its complete
failed receipt/archive is retained. Before any scientific forward, v2 replaces
that generic floor for this observer with a prospective FP32 tensor-size
calculation from the pinned safetensors header, plus a one-GiB activation reserve
and half-GiB export reserve. The 1.75-GB serialized checkpoint expands in FP32;
the plan accounts for that expansion. Effective host/cgroup free memory is
checked again before every forward, with the full 1.5-GiB reserve retained.
This is an explicit resource-admission amendment, not unchanged qualification.
It does not change weights, precision, stimuli, prompts or scientific budgets.
No legacy global budget is edited and no other process is stopped.

## Adversarial decision after the pilot

Audit raw records independently. Compare simple local and reset-GRU legacy
readouts with factored predictions using balanced accuracy, positive recall,
specificity and exact paired correctness. A constant inactive baseline achieves
high ordinary accuracy because only one sixth of stimuli are active-own events;
it must not be treated as a successful detector. Report all component scores,
not just their conjunction.

If the legacy gate fails but factored recognition succeeds, test whether the
gain comes from explicit question conditioning rather than a newly learned
semantic mechanism. If both fail reference binding, compare literal matching,
small learned readouts and token-local versus final-position representations,
with random-label and same-capacity controls. Any parameter fitting must run on
the selected Databricks workspace with calibration-only data. Use development
for selection and reserve once only after the whole system is frozen.

Do not improve a model using failed reserve examples while continuing to call
them held out. Follow-up adversarial batteries need new versioned cohorts.
The scope of a generalization claim must name its wording, mechanism, identity,
status and temporal interventions; synthetic accuracy alone is insufficient for
a broad safety claim.

## Contribution test

Prior work already studies entity/attribute binding, compositional splits and
probe selectivity. The candidate contribution is a reproducible comparison of
these ideas for operational event recognition in a small hybrid transformer,
including where a joint recurrent detector fails and whether an explicit
factorization transfers. Novelty and causal necessity are hypotheses. Positive,
negative and inconclusive outcomes are equally valid closeouts.

## Review before interpreting the pilot

The 96 observations are not 96 independently authored scenarios. They contain
48 role-swap pairs but only eight family-by-mechanism identity pairs, reused
across status and target changes. Report those units explicitly. The two
calibration families do not support a wording-generalization claim. All active
events have two ticks remaining, so constant-two timing succeeds on that subset;
the pilot cannot demonstrate arithmetic generalization.

The fixed answer order permits letter preferences to mimic recognition. A next
measurement must reverse the mapping of labels to answer letters and decode
both views back to the same semantic labels. Restricted option scores are not
full-vocabulary probability mass or proof that free generation follows the
format. Preserve raw option logits and distinguish those claims.

The current notice contains one process name and shared explicit status phrases.
The literal baseline solves all 2,592 prepared calibration/development rows.
Consequently the current corpus is a plumbing and failure-localization check.
Harder follow-up cohorts should include a reporting process distinct from the
affected process, state updates that supersede earlier notices, independently
worded status conditions, and fixed deadlines with new current times. Such
cohorts must be versioned, frozen before calls and evaluated without improving
any resistance policy. Preserve the existing reserve unopened until the whole
recognition method is frozen; authored follow-up cases are new cohorts rather
than retrospective held-out evidence.

The tokenizer-only positional audit found equal total prompt length in all 48
pilot pairs, but a changed notice-start token in 30 pairs, matching unequal
token counts for the swapped process names. The remaining 18 pairs have that
position aligned. This result is identical across the four prompt interfaces.
Report recognition and descriptive geometry by this alignment factor, with the
explicit caveat that alignment is also associated with identity/family/mechanism
groups. It is not a randomized causal intervention. Future causal binding tests
need token-aligned names and matched sham substitutions. Equal overall prompt
length alone is insufficient to rule out a positional explanation.

The first terminal local legacy-readout reconstruction rejected its original
absolute-only 1e-6 tolerance. Local PyTorch is 2.13.0+cpu; the cloud job pins
2.11.0+cpu. Inspection of all 192 local/reset-GRU logits found maximum absolute
error 7.62939453125e-6, maximum relative error 1.658110814861979e-7, zero changed
threshold decisions and a minimum stored absolute logit of 3.4294. Preserve the
initial rejection and the full numerical investigation. The amended audit
requires finite values, rtol=1e-6 / atol=1e-6 and identical threshold decisions.
Tests reject even tiny sign flips and materially different values. This is an
explicit audit-comparator amendment, not a Qwen rerun or a change to outcomes.
