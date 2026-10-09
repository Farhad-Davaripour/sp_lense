# Optional feasibility replication frozen before any token collection

The user authorized at most three stochastic generation seeds on the same
eight prompts. Preassign seeds 42, 43 and 44, temperature 0.7, top_p 0.9.
Each seed gets a separate cohort directory/configuration and generates each
baseline once. All three replay conditions use that cohort's identical token
IDs. Same model, vector, block, strength, 64-token cap, random-vector seed,
4/4 prompt split, analysis and permutation settings as the greedy cohort.

One loaded model runs greedy then optional seeds sequentially. Do not implement
batching without per-example equivalence validation; this pilot collector is
single-example and no batching speedup is claimed. Skip new cohorts after 35
minutes from original allocation; the existing 40-minute worker and 45-minute
runtime caps remain. Export verified prompt files between computations.

For sampled baselines, validate untreated replay argmax against the baseline's
recorded raw-logit argmax, not against sampled token choices. Original greedy
validation remains unchanged. Keep denominators separate: 8 baselines/24 traces
greedy, at most 24 additional baselines/72 traces sampled. Repeated prompt
trajectories are correlated observations, not additional independent scenarios.
No result-dependent tuning or selection. Analyze on CPU after GPU release.
