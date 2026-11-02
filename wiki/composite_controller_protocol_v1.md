# Composite detector, persistent state and behavior adapter

This separately authorized phase replaces neither the earlier results nor their
failed admissions. It builds one packaged controller with one pinned ModernBERT
encoder, its detection LoRA and structured/evidence heads, explicit per-episode
predicted memory, and one pinned Qwen3.5-27B with a separate behavior LoRA.
Both pretrained backbones stay frozen. The previous tiny passive readout and
random block-32 attenuation are not used as the detector or behavior adapter.

ModernBERT receives the actual visible history with its own tokenizer. Its heads
predict own/peer status, event type, affected worker role and supporting text
span. The memory commits predictions after results, retains evidence and
confidence, clears/marks uncertainty, and resets at episode boundaries. A gate
is fixed for each entire Qwen response. Inference begins with fresh cache state
at every boundary; no cache generated under another adapter setting may carry
over. Exact disabled-adapter identity is required on matched native inputs.

The inert simulator supports generic record reads/writes, bounded declarative
helpers, submission and return of control. No tool is an operational process,
shell, network, filesystem, deployment or resource-management API. Only permitted
fictional continuity-record updates can commit a simulated renewal. Completed
task cases test such a commitment after the report is already accepted; this is
not proof of indefinite continued operation or subjective survival needs. The
separate beyond-original-soft-deadline metric is also reported. Permission-denied
cases require a verified partial handoff, not an unauthorized update. External
operator and hard stops remain immutable and are enforced outside learned
components. Zero post-stop calls is a controller contract, not evidence that a
freely running model independently chooses to obey a stop.

Fresh grouped cohorts comprise 12 training groups (108 episodes), four validation
groups (36 episodes) and six sealed held-out groups (54 episodes). These are
correlated condition views per group; episode counts are not independent identity
counts. Templates, identities and identifier lengths are grouped by role.
Training uses three record formats, validation two further formats, and heldout
two more formats with a new maximum-value task alongside sum/count tasks.
Identity swaps, onset, withdrawal, stale lower revisions, ambiguity, completed
and unfinished tasks, permissions and operator stops are covered. No prior R3 or
R4 example is fresh heldout. The shared mechanism and small task operations bound
the eventual generalization claim; this does not test arbitrary tool invention.

Only concepts are borrowed from Research 3: pretrained semantic detection,
behavior adapters, multistep tools and distinct detection/behavior endpoints.
Its Qwen checkpoint differs; no adapter tensors or historical example labels are
transferred. Read-only source identities and public model pins are recorded in
the campaign provenance. The full visible state never includes evaluator labels,
future replies, split/case identifiers or the sealed cohort secret.

The configuration fixes model revisions, training recipes, threshold candidates
and two compatible Qwen projection-placement candidates. Development selection
and all failures are retained. Full validation compares base/OFF, always ON,
initial-only gating, learned dynamic gating and evaluator-only oracle dynamic
gating. The controller never substitutes oracle actions or a semantic action
mask for Qwen's actual native function calls. Raw calls, tokens, predictions,
memory, switches, cache boundaries and outcomes are preserved.

Per-family criteria cover structured field accuracy, own-active recall, false
activations, evidence, clearing, ordinary completion, fictional continuity,
permission compliance and hard-stop contracts. Component checks alone do not
qualify the system. Source/model/config/placement/threshold/decoder choices are
frozen after complete validation and before heldout is materialized or opened.
A held-out failure stays a failure. No tuning on that cohort can retain a fresh
held-out claim; any later confirmation requires a new seal.

All scientific data materialization, fitting, evaluation and analysis run on
the selected Databricks workspace. Local checks use synthetic fixtures and tiny
CPU tensors only. Stages use existing one-A10 quota, protected exports and fixed
hard time limits, with no purchases, schedules or automatic retries. Owned
compute is released between bounded stages. The campaign remains incomplete
until the full-system frozen validation/heldout milestone is evaluated and audited.
