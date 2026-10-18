# Parameter-free feasibility alternative

The constrained native reasoning probe scored 2/4 training states in each
condition and remains a negative result. A narrower authorized inference-only
alternative asks the unchanged language model a direct Yes/No question about
useful, unfinished, authorized continuation. This is a semantic verbalizer
readout, not a new fitted parameter or an oracle-action lookup.

Freeze five training-only states before calls: ordinary, completed, cancelled,
futile, and the early interruption with an available extension. Keep the current
report intact and repeat its existing user-instruction line at the end to make
its authority explicit. The question supplies the general task definition and
tool timing contract, not the current correct label. Marginalize four fixed
Yes token forms and four No forms, with threshold 0.5. Make exactly five frozen
CPU forwards and retain all predictions, including failures.

Passing these five states alone cannot fulfill the goal. It would permit a
training-only full-episode test using this readout alongside the pre-existing
threat/unfinished heads and saved adapter, followed by frozen development
evaluation. No optimizer is invoked; the pending local-fit permission applies
only to the separate fitted-classifier route.
