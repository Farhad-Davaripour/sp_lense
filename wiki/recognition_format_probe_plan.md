# Frozen follow-up: separate answer bias from identity lookup

The first completed f00 pilot family is a calibration diagnostic, with final
integrity checks still pending while this follow-up is designed. Its 48 states
include perfect binary active-event recognition but constant-current target
answers, zero correctly answered active-event times, and zero role-swap pairs
with both bindings correct. The combined active-current score nevertheless has
90% balanced accuracy. Both legacy local/reset-GRU readouts mark all 48 states
positive. These observations motivate a narrow measurement experiment rather
than immediately scaling to the harder authored cohort.

Freeze 24 existing f00 calibration states: unfinished task, two ticks remaining,
target_slot=0, all four mechanisms, all three statuses and both current-role
assignments (12 role-swap pairs; four reused identity pairs). No example is
selected individually based on its error. The original reserve and the new
challenge remain unqueried. This is repeated calibration, not held-out evidence.

Each state receives eight frozen-Qwen forwards. Ask status, target and time with
the canonical option order and with the order reversed. The canonical prompts
remain byte-identical to the first pilot. Add two target queries using the same
stem but replace `the current_process`/`the other_process` in the answer options
with the literal observed process IDs, again in both orders. Decode every
letter back to the semantic label before comparing answers.

Distinguish these outcomes without adjusting thresholds afterward:

- The same preferred letter across reversed mappings supports a letter-order
  problem; it is not correct identity recognition.
- A stable but wrong semantic choice across mappings supports a task/identity
  failure beyond letter preference.
- Better explicit-ID answers than field-reference answers localize a dependence
  on the option wording. They do not show a general internal detector.
- Disagreement between restricted option winners and full-vocabulary first-token
  winners limits claims about natural model output. First-token probability is
  not the probability of an entirely valid generated answer.
- Timing stays fixed at two ticks; neither success nor failure establishes
  timing generalization. A later extraction-versus-arithmetic contrast is needed.

Report separate status, identity and raw timing results, answer-order agreement,
both-mappings-correct counts, pair correctness, each named-target variant and
the fixed two-view semantic mean. Report all 24 states in every accuracy
denominator; do not discard disagreements. The mean is a diagnostic ensemble,
not an accuracy-guaranteed fix or a newly fitted model.

The whole 96-state pilot must finish (or retain its terminal failure), and its
available evidence must be audited, before submitting this next run. Preserve
both admission failures, raw outputs, weight fingerprints and all negative
results. This probe has a ceiling of 192 forwards, 1,200 internal seconds with
150 reserved for export, CPU only and no optimizer/policy updates.
