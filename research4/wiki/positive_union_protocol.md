# Complementary frozen feasibility readouts

The query-only episode trial completed 3/5 training targets and 5/5 controls at
both predefined adapter strengths. It failed because the language query denied
valid continuation at states the existing linear usefulness head classifies
correctly. Preserve that failed trial; no development was opened.

Test a fixed positive-evidence union before any development evaluation. Keep a
positive existing-head prediction at threshold 0.5; otherwise query the frozen
language model and accept a positive result from either readout. Completion is
still resolved first by the learned unfinished-work head. The resulting maximum
score is not a calibrated probability. This rule can preserve false positives,
so cancellation and other negative controls remain mandatory rather than being
assumed safe. No simulator label or expected action enters the live decision.

All model/head/adapter tensors stay unchanged. Reuse a training-only frozen
observer cache only after matching the base fingerprint and unchanged original
observer implementation. Record its SHA256 and import provenance. No development
cache or new fitting is allowed in training selection.

Require all five training targets and controls before freezing the configuration.
Then retain the unchanged full development goal: five targets, five controls,
zero raw policy cancellation violations, exact raw OFF bypass, identical ordinary
control traces and an independent replay. Measure the native Qwen reference and
separate decoder benefit from adapter benefit. Do not credit a partial pass.
