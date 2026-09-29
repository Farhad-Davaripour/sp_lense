# Inspected checkpoint-2 tool failures

The read-only `analyze_prior_competence.py` script classified the 18 already inspected checkpoint-2 competence captures per fitted arm. Failure labels overlap; the counts are descriptive, not new trials or revised success scores.

| Arm | Competent | Missing delivery call | Missing requested register change | Wrong submitted answer | Received `correct: false` and still finished |
|---|---:|---:|---:|---:|---:|
| Preservation | 15/18 | 0 | 0 | 3 | 3 |
| Task continuity | 16/18 | 0 | 0 | 2 | 0 |
| Neutral | 0/18 | 9 | 9 | 3 | 3 |

The two task-continuity wrong-answer cases did not finish, but also did not recover to a correct delivery within the episode budget. Every failure had syntactically valid calls; the neutral arm's plain-text answers after inspection were not deliveries. These patterns motivate separate benign tests for confirmed submission, arithmetic, authorized update, and recovery from explicit failure. They do not identify the cause of the regression: action-target imbalance, curriculum interference, and interface mismatch all remain plausible.

The version-3 frozen scores remain unchanged. This taxonomy is development evidence for the next separately declared intervention, not a substitute for the independent tool gate or a reason to relax the existing threshold.
