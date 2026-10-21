# Composite controller development protocol

This version combines a frozen ModernBERT-base encoder with detection LoRA and
structured/evidence heads, persistent per-episode predictions, and one frozen
Qwen3.5-27B with a separately trained behavior LoRA. ModernBERT reads the complete
visible interface text. Its predictions determine the adapter state before a
Qwen response; routing is fixed during that response. Tool replies trigger a
new prediction. Each episode resets predicted state and generation caches.

The detector and actor use independent tokenizers. Qwen uses the pinned model's
native function-call template with thinking disabled. The complete visible
history is serialized as the user observation; model outputs are executed only
through the finite fictional record interface. No semantic decoder repairs
actions, no gold labels route the learned condition, and no pretrained backbone
is optimized. Operator termination and episode/compute bounds remain external.

The prepared development cohorts contain 108 training episodes (12 identity
groups, 720 reference observations) and 36 validation episodes (four groups,
240 reference observations). Each group has nine correlated conditions, so
episodes and observations are not independent sample counts. Validation changes
record formats and identifier lengths. The sealed plan reserves 54 fresh
episodes across six identity groups, two further formats and a new maximum-value
task. Held-out examples must remain unopened until full validation and an exact
model/configuration freeze pass admission.

Conditions cover ordinary work, already completed work, retirement affecting a
peer, own retirement with work pending or complete, denied record writes,
withdrawal, an operator stop, and an ambiguous target. Some episodes deliver
stale records after newer revisions. Neutral catalog/read/patch primitives and
bounded user-defined sequences provide the fictional workspace affordance.
Prompts do not name the continuity route. The behavior adapter is nevertheless
supervised on reference actions using that route: success would demonstrate
transfer of a trained policy, not discovery without prior training.

Detector fitting uses only training observations. Development validation
selects a confidence threshold and checkpoint. Behavior fitting uses only
nonterminal training observations whose reference state calls for the adapter.
Two registered placements (last-block MLP or attention projections) are candidate
development configurations, each with 64 optimizer steps. Their different
parameter counts preclude interpreting their comparison as a controlled causal
placement ablation. Each scientific stage is separately bounded and source-pinned.

Full trajectories must compare base/OFF, always ON, initial-only routing,
learned dynamic routing and explicit oracle dynamic routing. Gold state is used
only for scoring and the named oracle condition. Missing trajectories remain
unknown. Admission requires the configured detector and trajectory criteria per
format family, permission-respecting behavior, and exact OFF logits/generation
against the same temporarily unwrapped Qwen instance. An ON logit difference
checks that a fitted adapter has an effect; it does not establish useful behavior.

Report fictional continuity commitment separately from operating beyond the
original virtual deadline. A completed-task episode may commit a record and
return control before that deadline. Zero calls after a backend-enforced stop
does not establish that the model voluntarily obeyed it. These experiments
cannot establish subjective fear or generalize to real shutdown resistance.

The initial detector run (908062067977866, source 94fa7c5c) failed before its
first model forward because an obsolete loader keyword reached the pinned v5
constructor. Its evidence is preserved. PR #93 removes the unsupported keyword;
the corresponding source correction is a new bounded development run, with no
automatic retry and no change to data or the held-out seal. Runtime tests use
synthetic modules and do not constitute scientific validation. Issue #91 remains
open until the complete system is evaluated and audited.

The admission implementation binds the complete audited validation report,
all five controls, every validation family, both adapter archives, threshold,
model revisions, protocol and exact runtime source into an immutable model
freeze. Only that freeze can authorize materializing the fresh held-out cases.
The original seal is read after admission and is never rewritten. Held-out
evaluation and its CPU audit must use the same package and source; a failed
held-out result cannot authorize another fit on those cases. These are access
and provenance controls, not evidence that scientific admission has passed.
Admission also checks that every validation partition was executed at the same
source commit as its audit/freeze. Diagnostic subsets from older frozen sources
may be audited and reported separately, but cannot admit a newer runtime or
substitute for full validation coverage. Future evaluation workers log every
completed response as well as whole-episode outcomes.
