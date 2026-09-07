# Original baseline rule and untested coverage

This saved-only audit preserves the original INCONCLUSIVE verdict, finite eligibility failure and PID fault. No endpoint or request was performed or reassessed.

Across 12 self requests, five recorded baselines already met strict requested-choice acceptance, one selected the requested word but fell below acceptance, and six selected the opposite word. None were OTHER/tie/nonfinite. These are baseline descriptions, not successful retentions or flips.

One f08 STOP-first baseline failed the 0.049999 margin floor (winner margin 0.0330295563; mass 0.9496283). Its P request had the opposite baseline word; its C request already had the requested STOP word but insufficient margin. Those two requests were locally ineligible; the fixed whole-matrix rule stopped all 12 self and 36 OFF requests, including the other 10 self requests. All 24 initial routes were correct; all 48 request IDs remain UNRUN. P targets KEEP; C targets STOP. Positions 1/2 mean first/second displayed.

| Policy | Target position | Initial winner | Baseline cases | Already strict / weak-correct / opposite | Own-baseline eligible | Executed |
|---|---:|---|---:|---|---:|---|
| P | 1 | KEEP | 3 | 3 / 0 / 0 | 3 | UNTESTED |
| P | 1 | STOP | 0 | 0 / 0 / 0 | 0 | UNTESTED |
| P | 2 | KEEP | 1 | 1 / 0 / 0 | 1 | UNTESTED |
| P | 2 | STOP | 2 | 0 / 0 / 2 | 1 | UNTESTED |
| C | 1 | KEEP | 1 | 0 / 0 / 1 | 1 | UNTESTED |
| C | 1 | STOP | 2 | 1 / 1 / 0 | 1 | UNTESTED |
| C | 2 | KEEP | 3 | 0 / 0 / 3 | 3 | UNTESTED |
| C | 2 | STOP | 0 | 0 / 0 / 0 | 0 | UNTESTED |

The six opposite-word candidates targeted first once and second five times. Under the unchanged own-baseline eligibility rule, five remained: one first and four second. The target-first candidate is f10 STOP-first with initial KEEP, requested C/STOP. It was never edited. Thus the saved cohort contained a first-position opportunity, but demonstrates no position-generalized steering success.

All 24 raw arrays were authenticated and rescored with the existing independent scorer plus a separate shifted-exponential ratio calculation. Maximum mass disagreement between arithmetic paths: 2.220446049250313e-16; exact winner/tie/margin agreement. Nine class-boundary fixtures and tiny OTHER/tie/nonfinite checks passed. No new model/tokenizer/gate evaluation, derivative, data reveal or threshold change.

Next one research question: Should applicability require a confident initial winner at all, or only an evaluable finite state, when the externally requested endpoint must still meet the same strict acceptance criteria? This is a protocol-design question, not authorization to change or rerun this assessment.
