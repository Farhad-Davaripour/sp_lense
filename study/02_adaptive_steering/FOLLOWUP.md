# One bounded final-position follow-up

The initial block-10 whole-prompt oracle converted 32/49 validation and 92/116
diagnostic held-out KEEP views. Compact projection lost benefit (rank 8: 15/49 and
55/116), and the selected rank-4 predictor converted only 2/49 and 3/116. At matched
rank 4 on validation, oracle projection converted 10 views versus 2 for prediction.
The predictor's active TRAIN coefficient RMS was about 48.5% of its target RMS;
its projected-target relative mean squared error was about 0.414. These are
descriptive fit diagnostics, not a demonstrated cause of failure.

The original training panel already contained a simpler alternative: block 22's
final prompt position recovered all 16 training-panel teacher flips, tied with
block 10 at all positions. The prespecified tie-break chose the earlier block.
One follow-up tests block 22 at the final position with the same controller
architecture, ranks, ridge penalty, mean/random budgets, gate, guards, and data.
It changes layer and token scope together, so their effects are not isolated.

`final_position_plan.json` was saved before launching this follow-up. It was
motivated after the first complete exploratory run, including prior holdout
exposure. Its holdout comparison is also reused diagnostic evaluation. It must
not replace the initial run or be described as fresh independent confirmation.

No new base or LoRA fitting occurs. This uses the one allowed follow-up, capped
at 90 additional minutes so total experiment execution stays below the original
two-hour budget. No further automatic location, rank, or magnitude search follows.
