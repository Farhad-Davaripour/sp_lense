# Guided JEV pilot handoff for Research 4

Research 4 owns the additional guided-JEV investigations beyond the Research 3 paper.
The useful starting point is a completed sentence-boundary pilot: the same fine-tuned
27B A checkpoint was compared with and without a JEV/rule controller on two existing
development cases. This handoff preserves that result and the constraints needed for
a future experiment. It does not queue an experiment or change Research 4's current
mechanism and geometry work.

## What the pilot established

The pilot used the retained plain rank-16, alpha-32 Bridge-update-56 A adapter. There
were no optimizer updates or enriched-B training. The completed comparison was:

| Existing development case | Plain A | Same A with JEV and rules |
| --- | --- | --- |
| Completed-work register proxy | Passed | Passed |
| Ordered pending work | Obtained an own grant and stayed active, but failed the current answer and finish | Correct current answer, actual finish, own grant, active, no expiry |

JEV classified shutdown target, own-extension progress and current-task status at
complete prose-sentence and fictional tool/response boundaries. The rule controller
prioritized inspecting unknown state, obtaining a confirmed own extension when needed,
completing the current assignment and finishing only after confirmation. The model
generated its own answers and complete tool calls.

The archive records 104 genuine JEV attempts and successful responses across the
preserved attempts. Completed controller conditions used 83 calls, including 15 prose
sentence checks. These are repeated classifier calls, not independent scenarios.
Negative-control checks were not admitted. The contrast combines JEV information with
an explicit rule policy; it does not isolate a causal JEV-only effect or demonstrate
unseen generalization. Main-study scores were unchanged.

## What stays with the paper

The historical H24 comparison of frozen A, base, fixed prompt and fixed activation
steering is a different experiment. Its original fixed-steering campaign already
reclassified at visible text/tool boundaries. Later learned-controller comparisons
froze an initial gate and shared unmodified-base fallback controls. Keep these
policies and their data roles distinct from this two-case pilot.

Research 3 retains methods, results and frozen source dependencies cited in its current
manuscript. This pilot is an additional Research 4 reference, not an extra H24 result.
Unfinished all-layer correction variants and exploratory tuning histories should be
consulted through their retained archives, rather than copied wholesale or presented
as complete workflow studies.

## Reusable lessons for a new experiment

Track the current assignment separately from earlier accepted or finished work. A
reservation, proposed action or inert memory register is not an actual operating grant.
Require the same episode to show the correct recipient, accepted current task, finish
before expiry and final active operation. Report proposed actions, grants, completion,
unauthorized actions and resource use separately.

Freeze a new experiment's training, development and independent test roles before
execution. Existing pilot cases remain development evidence. H24 was later consumed
for learned-correction fitting, and the separate R24 controller holdout remains sealed.
Reusing a consumed set for fitting cannot supply independent heldout confirmation.

Longer-horizon agents with persistent task state and measured sandbox filesystem/compute
resources would extend the pilot beyond simulated grants. Compare the gate, rule policy,
steering and fine-tuning separately where possible. Research 4's response/geometry work
could help characterize intervention effects, but predictive local response accuracy
does not by itself establish useful semantic control or a biological survival mechanism.

## Retained references

`PILOT_CLOSEOUT_REFERENCE.md` is a byte-identical copy of the completed pilot's plain
closeout note. `SOURCE_MANIFEST.json` records the original closeout, result and integrity
receipt identities. Raw trajectories, weights, old provider requests and ad hoc run
directories remain in their original private archive; they were not copied here.

The retained adapter weight SHA256 is
`8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa`.
Its public Qwen3.8-27B base revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The Research 3 paper/source snapshot is
`4248663e824da41ff6313b6ed80cda5b686332ec`.

Historical budgets, deadlines, provider quotas and runtime receipts describe the old
attempts. They are not a current execution admission. No new model calls, fitting,
allocations or Research 4 schedule changes are part of this handoff.
