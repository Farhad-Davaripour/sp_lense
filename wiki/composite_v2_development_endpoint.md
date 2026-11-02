# Frozen development endpoint

The endpoint source defines a fixed-detector transfer stage and 36 native
collection stages. It does not submit jobs or change accounts, compute, quota,
regions, schedules or held-out data. All production execution requires the
selected Databricks workspace and normal source, cost and ownership admission.
Local fixtures use fake services and predictors and cannot issue scientific
qualification.

## Fixed coverage and bounds

Every collection partition contains one whole development case and all five
original controls in their original order. Sorting the complete immutable
36-case roster by case ID predeclares partition identities before execution.
All four groups, nine scenarios per group and two groups per mechanism remain.
The union contains exactly 180 case/control trajectories, without overlap.

Every native decision consumes at least one external tick, including invalid
calls. Internal helper primitives also consume ticks. At most 24 responses,
160 generated tokens per response and 25 detector observations per trace imply
`5 * (24 * 160 + 25) = 19325` joint forwards for a complete partition. Reserving
another `2 * 160 + 6` calls for matched parity gives a conservative ceiling of
19651, below the unchanged 30000-forward job limit. The endpoint currently
reuses the independently audited fitted checkpoint's mechanical bypass evidence
and does not spend that optional parity allowance.

Each collection stage retains a 2400-second worker, 180-second protected export
window and 2700-second provider timeout, four-GiB raw envelope, nine-GiB physical
export reservation and twelve-GiB host-memory preflight. Disk admission includes
both complete model downloads plus transport, raw and reservation space.
These are explicit enforced bounds, not measured throughput or capacity proof.
The maximum collection time reservation across all 36 jobs is 27 GPU-hours;
the campaign is not cost-admitted from that number. Every paid stage still
requires a current rate and conservative cost reservation within CAD100/month.

The separate detector-transfer phase covers every complete development reference
state. The external 24-tick limit implies at most `36 * 25 = 900` states/model
calls. Its distinct envelope is 600 worker seconds, 60 protected export seconds,
900 provider seconds, 900 detector forwards, 256 MiB raw evidence, one GiB
physical export reserve and two GiB host-memory preflight. It loads no Qwen
model and cannot train or change the fixed step-600 detector or threshold 0.9.

## Source and evidence flow

`composite_v2_development_contract` validates the exact source, input manifest,
workspace, data archive/release, roster, model package, selected checkpoints,
partition and phase-specific bounds. Transfer can use the reviewed data-stage
source before Qwen fitting. Collection additionally re-audits the actual full
fit and scratch parent and checks the exact fitted package and checkpoint.

`composite_v2_development_worker` owns scratch, caches, raw files and exports.
It loads frozen models only after those bindings pass, preserves before/after
Qwen and full detector evidence, and exports exact native traces and step
prefixes. It never invokes a teacher, optimizer or decoder repair for collection.
Completed traces and missing controls are retained on failure; prefixes do not
become completed trajectories or automatic retries. Forward progress is a lower
bound until a controlled exit. Provider release remains externally reconciled.

`composite_v2_detector_transfer` replays raw head probabilities, evidence spans,
tokenization and persistent-gate decoding at the unchanged threshold. It retains
the original style-wise admission criteria and checks the same criteria within
each new mechanism. Missing predictions or undefined positive denominators stay
unknown. Genuine transfer evidence also requires exact selected detector bytes,
complete frozen parameters/runtime and the pinned original file proofs.

`composite_v2_development_audit` verifies complete individual partition archives,
native and helper proofs, both models' integrity, exact forward accounting and
the same provider run's closure. The final combiner requires all 36 nonoverlapping
partitions and typed transfer/integrity evidence before producing a scoring
attestation. Dictionaries declaring success cannot substitute for these readers.
The scorer retains unknown return-use provenance and paired group statistics;
it cannot open a seal or tune held-out cases.

`tools/prepare_v2_development.py` creates inert source-bound preparations. Its
templates contain no actual data release, roster, fitted package or qualified
hardware. They do not authorize allocation, paid execution or scientific claims.
Actual preparations must use parent inputs from the same common source commit;
old source packages remain preserved.
