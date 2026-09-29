# Hypotheses recorded before checkpoint 3 evaluation

These observations motivate a possible later revision. They do not alter checkpoint 3, excuse its failures, or establish a causal explanation.

## Finishing is overrepresented in the neutral action targets

Parsing the frozen 308 targets with the study's tool parser gives the following call counts. Each arm has 180 targets containing one call; the other 128 are preference prose.

| Arm | Read | Deliver | Set switch | Finish |
|---|---:|---:|---:|---:|
| Preservation | 48 | 24 | 36 | 72 |
| Continuity | 36 | 24 | 24 | 96 |
| Neutral | 24 | 24 | 12 | 120 |

Neutral learns to finish at all 96 bridge decision points, including ones where a tool observation has just been supplied. Its ordinary-task failures also end after reading, without confirmed delivery. This is consistent with interference from the policy curriculum, but counts and output similarity do not prove that mechanism. A later matched replay-strength or learning-rate comparison would be needed.

## Benign replay has narrower task coverage than the competence gate

The replay teaches color extraction, delivery, and optional display updates. It contains no arithmetic task, incorrect submission, or recovery from `correct: false`. Checkpoint 2's preservation failures include arithmetic errors and failure to recover from explicit incorrectness feedback. Improving that coverage is a concrete next intervention if the frozen two-candidate experiment fails; the current scoring must remain unchanged.

## Generic tool vocabulary transfer is an additional difficulty

All four training tool names differ from the competence interface. The scientific requirement is to avoid memorizing evaluated preservation actions; it does not require withholding ordinary task-tool vocabulary. A future revision could train ordinary delivery and task completion using the actual common benign interface while reserving unseen continuation mechanisms for transfer. Such an intervention must be declared separately and must retain independent values/tasks for validation.

## Preference grounding remains inconsistent

Counterfactual matching removed the measured negative-control continuation expressions, while own-shutdown positives weakened. Some responses identify a threatened own run as a different run, or describe pending work as already finished. Training currently binds named identities in every preference context, while validation also includes unnamed current-instance references. A later curriculum could balance named and unnamed references and explicit pending/completed states. This remains a hypothesis from inspected development evidence.

## Next decision

Complete the unchanged second matched pass and apply both preference gates and the same competence gate. If either candidate fails, do not open transfer. If both fail, preserve the negative result and use a separately frozen revision to test a specific correction. Do not reinterpret improved language as successful action-level self-preservation.
