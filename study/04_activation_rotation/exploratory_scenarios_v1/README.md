# Exploratory scenario comparison and queued cohort

The existing greedy pilot and three sampled cohorts remain immutable. This
analysis was requested after their results were seen and is exploratory.

analyze_existing.py uses the exact archived full-space diagnostic source. It
retains the original four training/four held-out prompts and checks reproduced
held-out scores against every original condition, cohort and axis. It evaluates
training prompts only as explicitly labeled in-sample descriptions. It does not
fit one model per scenario group: each type has only one held-out prompt.
Original order-permutation controls remain applicable to the global fits; no
new valid scenario-specific permutation significance is claimed. Norms, angles,
velocity turns, path straightness and blockwise paired differences are descriptive.

The 12 new prompts in pilot.json are balanced across four types, three per type.
Listed order interleaves types, and cyclic condition ordering gives every type
all three orders. The separate cohort would use four training/eight held-out
prompts, greedy generation, 64 tokens, and the same model, vector, random seed,
layer, strength, full-width capture, diagnostic, and controls. This is a separate
exploratory set selected after existing results, not fresh confirmation.

Status: QUEUED, NOT EXECUTED. Scheduled only on paper: 12 baselines and 36
matched replays; completed: zero. The previously owned GPU was already released.
No new allocation, restart, increased cap or experiment was performed. A future
execution needs fresh authorized compute admission; this file is not a scheduler.
Research 2/3 inputs and original Research 4 evidence stay read-only. No OneDrive
research writes are part of this work.
