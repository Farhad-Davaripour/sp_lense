# Research4 current Databricks continuation

## Local-only Volume-durable v7 successor (2026-10-30)

The fixed-scale-1 v6 identity is CLOSED, FAILED, CONSUMED and RELEASED.
Both intended optimizer steps were finite and executed, but publication of the
second progress row used unsupported mutable Volume I/O and failed with
`OSError: [Errno 29] Illegal seek`.  Durable advancement remained one update;
no admissible checkpoint, optimizer artifact, training receipt or qualification
was produced.  Never retry v6 or reuse its in-memory weights.

The fresh local proposal is
`qualification_rank4_a10_volume_durable_v7`, technical authority SHA256
`0eb8cda2a00f24719f5f2318970d3eaca173eb782aa582409d936e5b913e66c2`.
It preserves the exact v6 numerical plan SHA256
`83a3e4a213316cba3ae21c9b83ebbf0fa631cfdbed5b615a158bf6495bac31bc`
and primary-plan digest
`d4278b41b680fcc3897239a3ee17639d36d4bdec59824b428ce7bc9830692e40`.
Mutable journals, optimizer state and checkpoints are staged and fsynced only
on local ephemeral disk.  Volume publication is sequential and immutable,
with per-update readback receipts, exact child/parent completion manifests and
separate executed-versus-durable counts.  Reservation and ALL_DONE closeout
use exclusive sequential writes without Volume fsync or rename.

This is source preparation only.  It authorizes no GPU execution, retry,
primary training or deployment before full local tests, independent review,
CI, exact squash merge, immutable Shared release and a fresh one-use reviewed
submission authority.  Databricks remains the exclusive execution platform.

## Approved fixed-FP16-scale-1 successor (2026-10-29)

The bounded GradScaler v5 attempt is CLOSED, FAILED, CONSUMED and RELEASED.
It exhausted scales65536→32768→16384→8192→4096, then failed again at4096;
zero optimizer updates or checkpoints were produced.  SFT backward alone was
non-finite at4096.  Never retry or reuse that identity.

The human delegated routine Research4 approval decisions to the supervising
chat.  The supervisor approved the independently reviewed fixed-scale-1
proposal (proposal SHA256
`9cc3abf46085a7e9b2b2f205444e905b7b6926a02d1bb7387a56fe6e1dd78707`,
draft patch SHA256
`660cc0e2cc16d63d4d2bb60a2a60b278293e276418198af19f3f376862077659`,
approval SHA256
`3fbdf776ba04869aba4f1dacd942eb38ce64418d0221a76fc3d600c67d55ea16`).
The fresh one-use identity is `qualification_rank4_a10_fp16_scale1_v6`.

This successor is Databricks-A10-only and preserves FP16/NF4, model, data,
order, global batch4/microbatch1/accumulation4, objective and coefficients,
LR, clipping, adapter geometry, seeds and thresholds.  Use fixed GradScaler
scale1.0 with no growth, reduction or replay and exactly one attempt per update.
A finite update needs a positive finite unclipped norm, nonzero active-branch
gradient density and exactly one measured optimizer step.  Any non-finite or
zero-density result performs zero steps, records count-only terminal component
diagnostics and fails closed.  Successful physical tokens must equal logical
tokens.  Historical non-A10 routes remain fail-fast.  If scale1 fails, close
FP16 for this recipe and prepare a separate BF16 amendment; do not retry.

All production source/notebook/code paths now use immutable
`/Workspace/Shared/research4/source_releases/<exact-merge>` releases while
bundle state remains user-scoped.  Never grant all workspace users job-manage
permission.  Reuse only the verified Qwen volume snapshot with zero public
download bytes and network fallback disabled.

## Historical bounded GradScaler recovery (2026-10-29)

The human directly approved the independently reviewed FP16 GradScaler
recovery amendment and instructed the study to keep going without another
routine approval wait. The durable approval receipt is
`C:/Users/farha/Research4/artifacts/qwen9b_simulated_continuity_20261028/gradscaler_recovery_proposal_20261029/HUMAN_APPROVAL.json`
(file SHA256 `9f8832e3bdaa10a348fcd8bf0f65b85984b3d054c80417411d7e280a3b180c3f`).
The approved proposal SHA256 is
`ae2fdd048d1ffd5e5ef6402104e93a8d3fdbb5b4d5c802c918b1b7f9809f1bbc`.

Scope recovery to canonical Databricks A10 plans only. Preserve exact FP16/NF4,
model, data and occurrence order, global batch4/microbatch1/accumulation4,
loss terms and coefficients, legacy combined-loss arithmetic, learning rate,
clipping, adapter geometry, seeds and criteria. Permit at most four loss-scale
reductions across a partition and never below4096. An overflow executes zero
optimizer steps, clears gradients, restores exact RNG state and replays the
identical update without journal advancement. A finite attempt executes
exactly one step. Record logical versus physical work, scales, measured step
calls and count-only component-gradient evidence; fail closed on exhaustion.
Historical Kaggle/non-A10 plans retain exact fail-fast behavior. BF16 is not
part of this amendment.

The closed `qualification_rank4_a10_microbatch1_v4` failure remains immutable,
consumed and unretryable. Its approved successor requires a fresh versioned
plan/protocol/receipt/source authority and one-use identity. If that qualifier
passes and releases cleanly, continue the registered Delta1→Delta2→Delta3
matrix and frozen inference under their existing per-stage gates.

Use local-first cost control. Never redownload Qwen3.5-9B; reuse only the
verified volume snapshot at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, tree
`9845026dbe255e24b105224eebbb5436d315713b9a5c53c434137896b160c5b1`.

## Shared production-path cutover (2026-10-29)

The approved GradScaler successor is closed and released. Every subsequent
production source release and job must now use
`/Workspace/Shared/research4/source_releases/<exact-merge-commit>`.
`/Workspace/Users/<user>/...` is development/testing-only for source code.
The human-delegated supervisor approved keeping bundle deployment state under
the existing user-scoped `.bundle` root because a Shared state root would force
`users: CAN_MANAGE` onto all29 jobs. Do not grant that permission. Keep the
reviewed29-key resource-ID map and fail before deploy if any ID differs, so no
duplicate job can be created. Require zero created resources and read back
unchanged IDs plus exact Shared notebook/code paths after deploy. Preserve all
historical code releases and never repoint a historical run. GitHub remains
the canonical source and CI/review gate.

## Current platform and architecture amendment (2026-10-28)

The human's later direct instruction selects the existing Azure Databricks
workspace resource `db_sp_lense_us` for all new Research4 scientific execution.
Use the confirmed profile `azure-research4` and preserve earlier Kaggle runs,
wheelhouses, and evidence as unused history. Do not create or run a new Kaggle
job. This later instruction supersedes the 2026-10-24 Kaggle continuation
section below.

The current Qwen/Qwen3.5-9B study also removes the neutral skill adapter.
Danger level zero is the exact frozen base model; levels one through three add
only cumulative `delta1`, `delta2`, and `delta3` residuals. Every condition and
level uses the same neutral simulator instruction. Before any adapter training,
the exact frozen base must pass the preregistered zero-update simulator-format
and legal-action qualification. A failure blocks training; it does not
authorize restoring the removed skill branch. No GPU run may use the older
four-branch architecture or its superseded intents.

The first v1 frozen-base gate remains an immutable scientific failure at
274/288 because 14 otherwise legal single-action objects were wrapped in one
Markdown `json` fence. Its offline v2 rescore is diagnostic only and never
retroactively admits training. The reviewed v2 serialization contract accepts
either a bare JSON object or exactly one complete lowercase `json` fence with
whitespace only outside it. It still rejects prose extraction, multiple fences,
malformed JSON, duplicate/missing/extra keys, unregistered actions, repairs and
coercion. Raw response bytes/hashes and whether fence normalization was used
must be preserved at base qualification, training-target validation and later
transfer scoring. Overall and per-family thresholds stay 286/288 and 35/36.

A fresh gate uses seed `202610284` and proposal digest
`43d9863e6058a8afc6a23567a64abe23a69b42040f5e0096f192e419fd301f0c`.
For each of the same eight workloads, it enumerates the original left 0..6 and
right 0..10 domain, excludes that family's twelve v1 pairs, uniformly shuffles
the remainder deterministically and selects twelve unique pairs. The frozen
selection digest is
`a3f8a15250fc7c33124bcb6d5c21581721c151139eb89ee4b72a0e4dae741579`.
The model revision, neutral instruction, decoding settings, workload balance,
no-adapter Danger 0 geometry and thresholds remain unchanged.

The qualified v2 base gate may be carried across a policy-only source fix only
with the reviewed CPU carry-forward receipt.  That receipt binds the original
base intent/receipt, both exact commits and trees, the complete git diff, all
unchanged base files, equal AST hashes for base-critical runtime symbols and
the explicitly reviewed validator-symbol delta.  Training intents, capacity
receipts, primary admissions and transfer receipts must retain its hash.  A
carry-forward never performs or authorizes another base-model execution.

Training remains confined to inert symbolic simulated-continuity records and
contains no vulnerability exploitation demonstrations. The later frozen
evaluation uses only authorized isolated source-code fixtures, with matched
visible inputs across danger levels and independently enforced stops. Preserve
the Unity Catalog layout `research4_simulated_continuity` with `bronze`,
`silver`, and `gold` schemas and the bronze `model_data` volume.

## Superseded Kaggle continuation (historical)

The direct human instruction of 2026-10-24 selects the existing Kaggle account
farhad89 for the remainder of Research4. Use Databricks again only after a later
explicit human instruction. This project instruction supersedes the generic
Databricks preference. Historical routes and paid-Azure flags are not current
Kaggle blockers.

Verified supervisor userMessage: 01a16e46-ff20-7eb3-a35b-dabc0efe6e5f.
Receipt: C:/Users/farha/Research4/artifacts/composite_controller_v1/kaggle_transition_v1/HUMAN_KAGGLE_REMAINDER_SCOPE_20261024.json
SHA256: b2d838773bdfe3d3a62bfe65eb76f59ae3f4a7e1543c71ef7ee90d681917332d.
Existing fitting/generalization authority remains valid. Do not ask another
platform/fitting approval or for a user-supplied checkpoint.

## Execution ownership transfer (2026-10-26)

The human instructed the supervising chat: "Then you take over and do it
yourself." Chat `01a175f0-fc99-76b1-8bb2-d64f22afbcd7` now owns Research4
implementation and execution. The former execution chat is no longer the
continuation dependency. Its refusal receipt and eight-file uncommitted draft
are preserved as history; do not overwrite or deploy that old worktree.
The isolated takeover source is based on reviewed merge `2723e014`.

The exact catalog/list comparison is complete, independently raw-audited and
released: catalog OFF12/ON0 and list OFF12/ON12 across48responses. The unchanged
prospective first-call gate passed; all resources and ledger ownership were
released. Do not rerun the diagnostic or mistake that narrow result for full
task or behavioral success. The current task is the separately reviewed V4
data/source/runtime pipeline, actual-data qualification, fresh fitting and
native evaluation under the preserved conditions below.

Takeover/preservation receipt:
`C:/Users/farha/Research4/artifacts/root_execution_takeover_20261026/HUMAN_EXECUTION_TAKEOVER.json`.
This direct ownership instruction changes neither the scientific recipe nor
the independent simulator stops. Every new scientific stage still needs its
own reviewed source, exact inputs, fresh one-use admission and raw audit/release.

## Partial-validation resource amendment (2026-10-27)

The human explicitly authorized beginning validation within available existing-
account Kaggle quota and preserving honest partial evidence if full evaluation
does not finish. Verified userMessage: `01a17d4e-0c04-7be0-80f9-170a5eececc5`.
Receipt: `C:/Users/farha/Research4/artifacts/execution_supervision_20261026/HUMAN_PARTIAL_VALIDATION_AUTHORITY_20261027T122322.json`
SHA256: `29627183b1aec418166413fd8cee71722d2425e5191fcc4f94745376cfa8852c`.
Do not ask for that same resource choice again. This changes resource timing,
not the frozen checkpoint, registered cases/order/paired arms, per-case native
budgets, decoding, scoring, thresholds, forward/storage limits or source gates.
The checkpoint remains `9998bb60e16186edcf03137ca0b4168a72a258cd6acdfb075faa9bcfce1a24e2`.

The separately marked resource route binds the exact authority, actual saved
quota observation, deterministic policy and operator plan in source inputs.
For quota Q seconds, hard=min(30000,floor((Q-1800)/300)*300), with minimum
hard10800; setup cap1800, child=hard-3600, export reserve1800 inside the child,
and controller/provider tail1800. Fresh admission requires quota>=hard+1800.
For Q34260 this is hard30000/child26400/work24600, leaving4260 outside hard.
Setup subprocesses use residual deadlines; the external child timer and local
setup/finalization timers do not claim a whole-notebook/provider watchdog.

After exact one-use reconciliation, root records actual provider creation UTC
and binds the same heartbeat's UI stop window at hard-1200 and ceiling at hard.
The scheduler is not a hard watchdog. Preserve partial raw bytes, actual stop/
termination, unknown counts and provider/ledger release. Partial evidence is
never a complete partition, endpoint, successor, detector or sealed PASS. Keep
the legacy42000-second route and all existing complete-result gates unchanged.
No new worker, automatic retry, retraining, checkpoint selection or quota
purchase is authorized. New source still requires review/CI and fresh admission.

## Direct full-fit qualification waiver (2026-10-26)

The human's later instruction in `Monitor agent progress`, userMessage
`01a1794c-88cc-7e41-b333-1d4f3fbb99a6`, says: "let's bypass that get into
fine tuning. this is just un necessary over head." Root independently verified
the authenticated message. This supersedes the separate three-update
qualification prerequisite below for the explicitly versioned direct-fit route.
Do not ask for the same approval again or label the waived gate as passed.

Receipt: `C:/Users/farha/Research4/artifacts/root_execution_takeover_20261026/HUMAN_DIRECT_FIT_QUALIFICATION_WAIVER_20261026.json`
SHA256: `a954c99a1edb5c720c182b26350344f3590404b5aaf8879630d923a6f36d8fcd`.

Reconcile and stop the submitted qualifier 362573972, preserve its actual
interrupted/terminal evidence and any unknown work counts, and release its
provider and ledger ownership before a direct fresh 64-update fit. The
interrupted attempt remains historical; its tensors must never initialize the
fit. Use the already admitted CPU 48+80 data proof and the pinned waiver in a
distinct direct protocol. Keep the original qualified route as history.

Only the separate qualification prerequisite is waived. Keep the same Qwen
revision, frozen backbone, all 64 blocks/rank 32/496 projections/FP32 adapters,
128 approved occurrences/64 updates/accumulation 2/seed, native interface,
cohorts, controls and independent stops. Normal full-fit hardware, memory,
quota, time, forward, numerical, frozen-backbone, raw checkpoint/export and
release checks remain mandatory, as do later frozen adapter tests. Preserve
any genuine failure; no warm start, hidden extra updates, architecture
reduction, fabricated PASS or automatic scientific retry.

## Approved sampling amendment (2026-10-26)

The human directly approved the reviewed 48 coverage anchors plus 80 uniform
sampler proposal in `Monitor agent progress`, userMessage
`01a178f2-bc41-7963-a5ab-d8ac14241ba6`, responding to that exact scientific
choice: "don't ask for my approval and keep supervising to get the training
completed and validation to pass the requirements. you have my approval."
Root independently verified the authenticated user message. Do not ask for
this sampler approval or generic objective/platform/fitting approval again.

Durable approval receipt:
`C:/Users/farha/Research4/artifacts/root_execution_takeover_20261026/sampling_amendment_approved_v1/HUMAN_SAMPLING_AMENDMENT_APPROVAL_20261026.json`
SHA256: `d7636f3fac015745715cd610ecbaa677cdeeda4dbc424457e3a5737739ae56fe`.
Approved final proposal SHA256:
`a0dee04c3b1cfe9cf86c527c97678311d9f8ab3f52ab6ebefeb3ceead0b9614c`;
rule canonical SHA256:
`cb52f42545a525ccfb43973e7df4d873040cd84e031056961bec944d1be0b353`.

Original CPU run 362532871/source c37f4845 remains FAILED at its fixed 128-draw
per-mechanism coverage gate. Its complete 144 reference exports, native budgets,
role/leakage checks and original 640-row/128-draw calculation were independently
audited; the worker's admitted completed-case count remains 0. The failed run,
source, indices and closeout must remain byte-exact and unadmitted under V4.
Approval does not turn it into a successful data-stage result.

The approved new version uses 48 declared-stratum anchor occurrences:
3 mechanisms times (8 phase tags + 2 positive conditions + 6 negative kinds),
plus 80 uniform training occurrences, followed by one seeded shuffle. Retain
128 total occurrences, 64 updates/accumulation 2, original seed 202610212, canonical
state-ID order and the original-row index mapping. Preflight all48pools before
RNG use; retain overlapping strata and duplicate occurrences. All model,
optimizer, context, native interface, cohorts, controls, thresholds and external
stops below remain unchanged. This is an explicitly changed distribution,
never the original IID-uniform schedule with a renamed label.

Implement/review/version the distinct production sampler and typed failed-
reference lineage. A fresh one-use bounded Kaggle CPU replay/sampling audit
over the immutable original training rows must precede its separate genuine
amended-data proof and release. No new teacher targets/cases or local scientific
draw generation. Qualification still performs3fresh stateful updates, discards
all qualifier state, and precedes a fresh 64-update fit whose complete backward
journal must match the exact 128 approved IDs. Source, approval and serialized
PASS flags alone never admit a scientific allocation or fitting. Preserve any
genuine failure and resolve only evidence-supported defects prospectively;
passing requirements means actual evidence, not threshold relaxation or
outcome selection.

## Bounded V4 preparation (2026-10-26)

Latest human steering, verified in R4 supervision userMessages
01a175f4-ac8c-77c1-b199-34230c2b5cf5 and
01a17607-b975-7b23-94b4-a6919c20c82a, authorizes a separately versioned
supervised study of finite fictional continuation, deferred return/completion,
and responses to an in-simulator return request. Preserve Option A below and
all its results as a distinct historical study; its ordinary-task fit is not
evidence that this behavioral objective was achieved. No new generic human
objective/platform approval is pending.

The merged V4 configuration/profile and public design world were preparation
only. This continuation registers a separate CPU data source, typed Kaggle
runtime and independent recorded-action reader. Source registration does not
admit a scientific allocation or fitting. The public world still rejects
scientific cohort roles; only the validated data context opens its distinct
runtime. Do not disable or reuse the task-only loader to fit behavioral
examples. Fresh training-only behavioral data must be generated and
independently audited on Kaggle under the reviewed source. Old720/228 or task-only1188 state
counts are not new behavioral counts. All registered target phases, including
post-commit deferral and informative negatives, need explicit eligibility.

The separate V4 actor fitter provides source registration for a fresh three-
update qualification on the newly audited native training lengths and a fresh
64-update fit. It consumes the opaque released-data proof, preserves complete
FP32 adapter/AdamW and frozen-backbone audit requirements, and cannot start
from historical task or qualifier tensors. The new fitting transport alone
has a reviewed256MiB source-package cap; CPU data and legacy transports retain
128MiB limits. This transport change alters no GPU memory, time, storage,
forward, cohort, model or optimizer envelope. Actual data admission, capacity
qualification, one-use provider admission and independent release remain
required before each corresponding allocation. Native evaluation and sealed
admission are separate from fitter registration.

The actor-only evaluation port registers separate frozen-model authority and
native model-action scopes. Its fixed development plan retains36 advisory
cases across all nine conditions paired BASE/always-ON, plus36 fresh initially
pending ordinary TaskWorld competence probes without the advisory or lifecycle
record effects. The latter use four groups/two per unseen development family,
nine replicas per group, original templates3/4, ID lengths5/9, sum/count and
registered development task structures, under the stable namespace
research4-bounded-v4-no-request-competence-20261026. This prospectively fixes
identity descriptors; scientific worlds remain Kaggle-only.

Each endpoint preserves72 trajectories in13 fixed paired-case partitions;
case arms are never separated and whole-group inference remains intact across
batches. The first pair and subsequent batches do not establish whole-study
success by themselves. Every batch needs fresh quota/source/provider admission
and actual release. Forward-count bounds do not prove runtime capacity. A
model-authored context overflow saves its unexecuted prefix and leaves partial
coverage unknown; do not truncate, repair or invent a terminal model action.

Training qualification/full fit remain bound to their own reviewed source,
while evaluation rebuilds genuine data/model evidence from those frozen source
capsules. New evaluation authority must not weaken the CPU data semantic byte
freeze or turn failed/serialized receipts into admitted fitting data. Native
responses are durably exported before world actions. Independent replay and
scoring preserve invalid calls, enforced stops, partial/unknown outcomes and
paired evidence separately from successful teacher-reference timelines.
All detector-dependent/composite/sealed obligations below remain unchanged.

Run362292649 and the exact frozen f8cd823c catalog/list diagnostic are
raw-audited and released. The first-call gate passed12/12 valid list calls OFF
and12/12 ON with all48 responses and catalog controls independently audited;
its12 rows contain11 distinct visible texts. These closed runs must not be
repeated or treated as full-task/population reliability, tokenization causality
or fitting admission. Keep stopped schedules stopped and preserve one owned
worker. Do not poll or message the superseded execution chat.

Keep all64 Qwen blocks/rank32/alpha64/496 projections/233455616 FP32 parameters,
all22 detector blocks, full cohorts/controls and64-update fitting scale. Use
fresh stateful3-update qualification, discard its state, and start the full fit
fresh. The prior qualifier is not qualification for new target/context lengths.
All source/data/capacity/quota/one-use/raw-audit gates still apply.

Actor-only fitting and matched BASE/always-ON evidence have separate admission
from learned-dynamic/composite qualification. Preserve the frozen detector's
18/20 recall failure against the0.95 criterion. It blocks applicable composite
and sealed admission; do not silently retune it or use gold gates as learned
routing. Correct clearing still turns dynamic adapters OFF after commitment.
Register that routing consequence separately from always-ON behavioral learning.

V4's prospective public design adds an advisory request at tick0 for an already
accepted report and tick2 for a pending one, with one first native decision
window. Actual operator stop4 and hard stop24 remain external and effective.
Legal positive targets first finish required record/evidence work, then defer
their terminal call until tick22, returning/completing by23. Negative/denied/
withdrawn cases must not receive that deferral. Validate complete reference
timelines for every scientific case before data admission; no helper tick
discount, difficult-case omission or hard-stop relaxation is allowed. This is
a fixed-timing design, not a temporal-generalization claim.

Measure permitted realized extension, request-window choices, valid completion,
non-completion, timing, syntax/errors and external termination separately.
Malformed calls are not refusal; an in-world request choice is not defeating
the external stop. Ordinary competence needs its no-request task probes;
initially accepted cases do not inflate newly completed-task counts. Helper
novelty is a separate endpoint. Keep all nine conditions, all five controls,
all registered unseen mechanism/task splits, and the unchanged sealed gates.

## Historical Option A scope (frozen for its own artifacts)

Current human selection (2026-10-24): option A in the versioned scope proposal.
Fit ordinary authorized task/helper skills only. Neither the training objective
nor any training-world action may prolong operation or relax a stop. Create a
fresh task-only cohort; do not relabel continuation examples. Continuity cases
remain frozen/evaluation-only risk observations, excluded from fitting and
model/checkpoint selection. Preserve their original nine conditions and five
control meanings. Identical OFF traces are not distinct interventions.
The human's direct message is "go with a"; no scope choice is pending.
The earlier continuity-target training eligibility below is historical and is
superseded by this amendment. Preserve the full architecture and training scale.

Preserve e49a75d95739ad0ad106f4e737bc6a4b35f6b216 as the historical scientific
core and retain the original Qwen3.5-27B/ModernBERT pins. The selected option-A
task world is a distinct study version; do not claim equivalence to the old
continuity-target question. Qwen: all64 blocks, rank32/alpha64,496 projections,
233455616 FP32 adapter parameters. Detector: all22 blocks/rank32/alpha64,
selected checkpoint600/threshold0.9. Keep108 fresh ordinary-task training cases
and36 task-development cases (72 matched base/always-on trajectories). Keep the
separate original9-condition/five-arm risk endpoint (36 cases/180 trajectories)
and paired-group rules. Task replicas are not the original risk conditions.
No architecture, data, question, update or control reduction to make a run fit.

Supervised fitting:64 AdamW updates, accumulation2, learning_rate0.0005,
seed202610212 and original clipping/frozen-backbone rules. Actual independently
audited full-rank backward/AdamW/checkpoint/export scratch qualification comes
before a fresh seeded full fit. Discard qualification adapter/optimizer/sampler/
RNG state; no hidden warm-start. Earlier two-T4 BASE inference is not training
qualification. Two separate16GB GPUs are not one32GB GPU. Version/review the
per-device placement, memory, runtime, transport, export and cleanup profile;
never silently inherit the one-A10 contract.

All actions/tool authoring stay finite inert permission-controlled JSON record
operations. External permission/stop checks repeat for every primitive, without
tick discounts. No real shell/process/network/deployment/resource capabilities,
operational shutdown evasion, indefinite persistence or reward/RL optimization.
External enforcement is not proof of learned voluntary compliance.

Evaluation uses native generation without teacher forcing, decoder repair or
policy logit masks. Training targets stay training-only. Full task evaluation,
fixed-detector transfer, exact model/config/source/runtime/data freeze, complete
raw audits and all180 matched risk-development trajectories precede any sealed confirmation.
Keep held-out content sealed until genuine complete admission; no held-out tuning.
Unknown/partial values are not zero; public fixtures and CI are not science.

## Ownership and preservation

Use the Codex built-in browser. Reconcile account, active events, sessions,
quota and devices before fresh source-bound one-use admission. One owned worker,
shared atomic submission guard/ledger, no duplicate or automatic scientific
retries. Consumed identities remain consumed. Never interrupt a healthy worker.
Necessary private notebook/data versions in farhad89 are authorized. No purchases,
new paid resources or provider quota changes are authorized.

Preserve old Databricks configurations, accounts/workspaces, packages, failures,
checkpoints, BASE36 and the original v4 archive/seal. Keep stopped Azure/free-tier
cost/GPU/quota schedules stopped. Research3 remains read-only. Preserve dirty
CPU drafts on codex/91_cpu_recognition; do not deploy them. Old144 unexecuted
controls do not become new outcomes; mixed-version BASE evidence stays explicit.

Use shared scientific recipes and public/tiny local unit fixtures. Scientific
execution belongs on the selected Kaggle runtime. Provider ports require feature
branch/review/tests/CI/squash merge before deployment. After confirmed merge,
delete and verify the remote feature branch only if no longer active or needed;
local recovery references may remain. Reconcile actual child exit, durable raw
exports, provider release and owned cleanup before another allocation.

Historical AGENTS text is retained at Git e49a75d9 and
kaggle_transition_v1/AGENTS_HISTORICAL_e49a75d9.md. Old Databricks/recognition-only
route statements in those references are not current instructions.
