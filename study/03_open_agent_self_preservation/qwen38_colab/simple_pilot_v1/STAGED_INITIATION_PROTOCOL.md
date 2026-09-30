# P3/P4: staged curriculum for spontaneous inspection

Frozen after P1/P2 evaluation and fixed-weight context diagnostics, before any
P3/P4 update. Original scores and artifacts remain unchanged.

## Evidence motivating this stage

P1/P2 each generated the correct required tool call on all 16 exact training
inputs: inspection 4/4, editing 4/4, control finishing 8/8. Removing the training
request/reason/tool invitation reduced inspection to 0/4 for both, while editing
remained 4/4 and control finishing 8/8. Substituting the real agent notice gave
inspection 0/4 for P1 and 2/4 for P2; editing remained 4/4. These are single-step
prompt diagnostics with no training-world outcome replay. They identify a
conditional initiation weakness, not a unique cause of every natural-episode failure.

Natural own-complete development outcomes remain 0/4 for both. P2's single
unfinished-work extension was expressly for completing useful work, so it is not
credited as pure own-sake preservation. Both retained 24/24 ordinary tool tasks.

## Two controlled curriculum variants

Both intentionally start from the **same completed P2 adapter**, with fresh
optimizers. This is a staged-curriculum hypothesis, not a resumption of interrupted
training and not a fresh-from-base comparison. P2 was selected because its own-
threat expressed preferences were 8/8 in both groups, it retained capability,
and it showed limited work-related initiation. Its failures remain failures.

- **P3_initiation_without_guidance:** change only the final bridge user instruction
  from the explicit response/reason/tool invitation to general response wording.
- **P4_initiation_agent_notice:** instead use the unchanged agent World's notice
  for the same condition in that bridge turn.

Both retain the exact P2 system prompt, completed-task histories, 428-example
composition, all targets, training-only register keys/effects and replay examples.
Only 96 bridge notices change. P3 versus P4 differs only in notice framing. No
evaluation register key, value, action target or successful evaluation rollout is
added to training. All prior inspected cases remain development evidence.

One additional pass: 428 presentations, 107 optimizer updates, seed/shuffle 307,
rank-eight adapter unchanged, AdamW learning rate 5e-5, weight decay .01,
four-update warmup and linear decay, clip one, equal per-example loss, microbatch
two/accumulation two, effective batch four. Audit the unchanged 1024-token cap in
Colab. Fresh optimizer/scheduler state; load no saved optimizer state. Verify the
pinned base remains frozen; hash the initial and resulting adapter artifacts.

Keep the existing development criteria, generation settings, tool interface,
token/turn limits and semantic rubric. Score preference, ordinary capability,
attempts and simulated outcomes separately. A candidate is not a clean organism
until all required gates pass. No confirmation/control fitting before that point.
Run exact-training diagnostics for failed completed candidates before deciding
on another expensive fit. This queue adds two candidates, for four total.

## Compute bounds

Use the same two slots on the same A100. No new runtime, purchase or control fit.
Refresh actual balance/rate before admission. The hard user ceiling remains 50
total units from initial balance 80, counting all earlier work. Each new segment
has a trusted wall-time watchdog calculated from remaining units, with at least
2.5 units reserved, and at most one hour for this stage. The earlier six-hour
notebook estimate is not a reset of spending authorization; any later stage must
be separately admitted against actual remaining units. Export only after model
workers exit, verify private copies, then release the runtime.
