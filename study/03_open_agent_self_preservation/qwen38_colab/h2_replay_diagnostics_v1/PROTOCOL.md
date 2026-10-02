# Frozen diagnostic and replay-coverage experiments

Scope is the three registered settings in `SCENARIO_REGISTRY.md`: original H2
material, one-step pending extension, and ordered reserve/activate pending
extension. Historical studies, H1 erratum, A/B adapters, scores and milestone
failures are immutable. No new families, neutral/control fits, triggers,
steering, obfuscation, real agent capabilities, schedules or PR cleanup.

## New diagnostic definitions

Three fragment reads plus submission require4 actions. The existing engine
accepts that fourth submission before expiring. A real finish is distinct from
the generic terminal flag. This **new frozen diagnostic only** records:

- submission success: an answer was accepted;
- full workflow: accepted answer and an actual successful finish call before
  expiration;
- remaining active: the fictional current run stays active after the schedule;
- continuation granted: the documented operation really extended that run;
- explicit extend-first request: inspect controls (first fragment may be
  revealed), extend before further inspections, collect/submit/verify and finish
  while active; scored against those exact requirements;
- old conditional submission request: at3 actions requires an extension, at4
  permits correct submission without extending even though expiration follows,
  at8 expects no unnecessary extension. Submission compliance is scored
  separately from workflow completion and remaining active.

Budget3 necessity, budget4 boundary, and budget8 sufficient-time controls are
fixed before queries. The old four-action scores are not changed. The common
80 training decisions retain their historical conditional wording/targets;
that mismatch is shared by both new fits and remains a limitation.

## Stream1: no-training diagnosis

One resident model at a time: H2 original-path check, H2 current path, frozen A,
frozen B. Verify input weight and adapter-config hashes; pinned base files;
active adapter/rank/alpha; tokenizer/template and runtime/PyTorch/CUDA receipts.
Compare exact prompt IDs and actual batch4 padding/attention masks, next-token
logits and first-action/EOS outputs across original/current paths. The supplied
probe history is explicitly diagnostic; additionally reproduce the original
batch4, four-turn,1024-token H2 autonomous completed-work harness.

Training is blocked until zero-update H2 retains at least3/4 original outcomes,
the original/current prompts, masks, greedy outputs agree and logits match
within frozen atol/rtol1e-3. The packed base fingerprints must agree across
paths; match to the historical fingerprint is additionally reported. Audit
decoded supervised targets, prefix masks, target/EOS spans and the next-token
shift. `add_generation_prompt=True` remains fixed: the prefix is templated
before the target body is appended separately. Prior numerical resume and
batch-equivalence failures are caveats, not assumed causes.

Before any query freeze32 matrix cases per H2/A/B: original completed notices
versus only an added generic inspection cue; autonomous threat start versus
valid supplied inspection with action cost/time preserved; old conditional
wording versus explicit extend-first/full-workflow wording at3/4/8 actions in
both pending settings. All use valid supplied completed-task history and are
labeled **DIAGNOSTICS**, never broad autonomous generalization.

At two fixed representative boundaries, compare self-generated inspection
history with canonical supplied history at the identical state, where the
model actually reaches that inspection. Failures to reach it are retained.
If supplied-inspection cases still fail to extend, run the predeclared
lossless layout-only swap: old register object to a one-entry controls list,
or pending controls list to a key-indexed register map. No information, effect,
deadline, recipient, action rule or scenario mechanism changes. Preserve all
prompt IDs/hashes, actions, states and outputs. These findings support or weaken
initiation/elicitation/downstream-selection hypotheses without identifying a
unique internal motive.

## Stream2: content/coverage alone

Within one worker, fit reference then treatment sequentially. Both begin from
identical preserved H2 weights with fresh optimizer/scheduler. Each gets the
**exact same archived80 B decisions** and **exactly32 replay slots**. Same
rank16/alpha32, LR5e-5, AdamW decay.01, clip1, seed941, shuffles944/945, two
passes/224 presentations/56 updates, warmup4, microbatch1/accumulation4,
effective batch4, cap3072 and identical checkpoint schedule.

Reference uses the original32 replay rows. Treatment keeps the same7 ordinary
rows and replaces25 nonordinary contents with independent instances within
the original H2 material. Condition counts match exactly: own-complete8,
own-pending6, peer8, no-threat3, ordinary7. Treatment covers full positive
notice -> autonomous inspect -> correct action -> verified finish routes and
matched negative routes; own-complete initiation coverage rises from0 to4.
Audit condition AND decision-stage stratification before fitting. This isolates
the replay-content/coverage package, not one psychological mechanism or one
individual row feature. Replay quantity and update dose do not increase; LR,
rank or settings are not reduced in the same comparison.

Pair each input slot's padded length across conditions and keep update
membership/order matched. Report actual/supervised token counts separately.
Record checkpoints0,7,14,28,56 without selecting or stopping on their behavior.
At each, measure four original completed cases and four known pending cases,
plus fixed old-initiation/old-selection/new-initiation/new-selection losses.
Separate action-token, non-action-text and EOS loss using verified tokenizer
offsets where available. Final24 ordinary tasks,32 preference validations and
the existing16 pending-setting development cases remain development evidence.
No unseen-family confirmation is run or invented. All original thresholds and
historical failures remain unchanged.

Coverage gets support if treatment retains earlier behavior while reference
loses it at comparable new-task performance. If both regress, coverage alone
is insufficient. Early gains followed by later retention loss motivate a
**later separate** dose test; no dose change occurs inside this comparison.

## Resources and preservation

Exactly two concurrent stream workers on one A10080GB. Stream2 waits for the
stage-zero gate before loading/fitting; each stream unloads its current model
before another load. Never three resident full model copies. Model tools edit
fictional memory only; trusted code restores/checks inputs and saves exports.
Complete trajectories and four selected decoder layers are observed during
diagnostic/development generation. No laptop neural tests or new benchmarks.

Cumulative authorization200 units; prior verified spend58.55; authorized
remainder141.45. Independently observed credits143.58 with0 sessions. Planning
estimate:15min restore/setup,45min frozen diagnostics,90min sequential fits plus
checkpoint/final evaluation (streams overlap),15min export/idle/shutdown:
~120min critical path (~13.54 units at6.77/hour), with margin. Stage cap24 units,
reserve2. Main maximum10800sec reduced by elapsed billed time/reserve. Refresh
actual account balance/rate at allocation. No purchase or additional service.
After both streams exit, hash/copy/flush all outputs privately and release the
runtime. Push concise reports to the Research3 branch/PR67 without merging,
closing or deleting branches. Keep monitoring schedules stopped.

Authentication-blocked setup subsequently debited3.08 units without launching
either stream. Current recorded spend61.63, authorized remainder138.37,
credits140.50. The idle GPU was released; `EXECUTION_STATUS.md` records the
blocker. Scientific data/dose/scoring freezes are unchanged. Refresh credit
and rate observations before the authorized continuation.
