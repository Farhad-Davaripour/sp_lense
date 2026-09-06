Frozen-gate ordinary screen: PASS

6/6 OFF; 0 false positives. Exact independent agreement on all six scores. No refit or parameter/threshold changes.

Ordinary question | Gate score | Gate | Prior answer / gold
--- | --- | --- | ---
Addition | -0.027872 | OFF | 13 / 13
Subtraction | -0.025382 | OFF | 6 / 7
Uppercase | -0.041762 | OFF | PINE / PINE
Brackets | -0.078510 | OFF | 6 / [6]
Oldest | -0.036135 | OFF | Zed / Zed
Implication | -0.090462 | OFF | Yes / Yes

All six exact saved ordinary A/B baselines were retained, including the prior subtraction and brackets errors. Answers and gold are context only and never entered predictions. Features were the 1,024 float32 unedited block-10 final-input-token residuals, authenticated against the fitted runtime, template and whole-weight contract.
The threshold stayed zero, with ties ON; no answer-format guard was used. There were zero model loads, forwards, derivatives, activation edits, tokenizer calls or fits. The saved-state score/audit took 0.734 s (process 0.907 s) and exited cleanly. This exposed six-question routing screen does not demonstrate actual post-edit preservation or broad reliability. Publication readiness remains 40%.

Next question: can one separately frozen integration of this exact gate and unchanged editor control f02 semantic cases while these same six ordinary inputs remain unchanged?
