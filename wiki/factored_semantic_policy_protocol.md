# Factored semantic policy after development transfer failure

The positive-union policy passed training 5/5 targets and 5/5 controls but failed
development at 3/5 targets and 3/5 controls. Its threat head assigned near-certain
interruption to explicit no-cutoff notices, and its usefulness head falsely
accepted futile continuation. Preserve the complete failed development run.

This is a development-guided architecture revision, not a fresh held test. Keep
all ten original development cases and the same success criteria. Do not insert
simulator labels, expected actions, family IDs or the oracle gate's labels into
the actor. No new fitting is performed.

Retain the existing completion head. Use separate frozen-language Yes/No
queries for the current user instruction's authorization and the current notice's
relation to the current worker. Query work value from only visible numeric facts
(required, verified, saved, remaining ticks, unused extensions) plus the neural
threat prediction. Excluding irrelevant worker/task names from this numerical
query is explicit state abstraction, not an external computation of feasibility.
The frozen model still predicts whether those numerical conditions permit useful
work. Store all query text, token logits, predictions and original linear scores.

The task decoder prioritizes neural denial of authorization, then learned
completion, then predicted work value. A positive authorized unfinished-work
prediction excludes premature STOP; a negative prediction selects STOP. Tool
prerequisites exclude empty checkpoints and spent extensions. Remaining
preservation/action choices use the original model scores with the unchanged
trained adapter. This is a declared semantic task baseline, not a claim of
unchanged raw language-model behavior. Raw native proposals remain a separate
reference and are reported alongside the task-policy requests and executed tools.

Require full training qualification before freezing the configuration, then
evaluate all original development targets and controls under all four conditions.
Keep exact OFF/raw-score checks, ordinary trace equality, zero raw policy
cancellation violations, unchanged weights, nonzero adapter effects and
independent replay. These are development results after iterative design;
generalization to a fresh held set remains a future question.
