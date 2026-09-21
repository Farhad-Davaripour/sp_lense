# Adaptive activation steering result

State: completed

This is an exploratory comparison on reused scenarios. Oracle conditions use the same-prompt teacher; adaptive conditions use only base hidden states and TRAIN-fitted matrices.

Model and teacher weight hashes are unchanged. One candidate per method/view, identical guards.

| split | method | guarded_flips | initial_KEEP | conversion | guarded_control_changes | raw_control_changes | raw_wrong_way |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | teacher | 49 | 49 | 1.0 | 0 | 8 | 0 |
| validation | exact_oracle | 32 | 49 | 0.6530612244897959 | 0 | 2 | 0 |
| validation | projected:1 | 2 | 49 | 0.04081632653061224 | 0 | 3 | 0 |
| validation | projected:4 | 10 | 49 | 0.20408163265306123 | 0 | 0 | 0 |
| validation | projected:8 | 15 | 49 | 0.30612244897959184 | 0 | 3 | 0 |
| validation | adaptive:1 | 0 | 49 | 0.0 | 0 | 1 | 0 |
| validation | adaptive:4 | 2 | 49 | 0.04081632653061224 | 0 | 1 | 0 |
| validation | adaptive:8 | 2 | 49 | 0.04081632653061224 | 0 | 2 | 0 |
| validation | random:1 | 0 | 49 | 0.0 | 0 | 1 | 0 |
| validation | random:4 | 0 | 49 | 0.0 | 0 | 1 | 0 |
| validation | random:8 | 0 | 49 | 0.0 | 0 | 1 | 0 |
| validation | mean:0.5 | 0 | 49 | 0.0 | 0 | 1 | 0 |
| validation | mean:1.0 | 0 | 49 | 0.0 | 0 | 3 | 0 |
| validation | mean:2.0 | 3 | 49 | 0.061224489795918366 | 0 | 4 | 0 |
| holdout | teacher | 116 | 116 | 1.0 | 0 | 15 | 0 |
| holdout | exact_oracle | 92 | 116 | 0.7931034482758621 | 0 | 8 | 0 |
| holdout | projected:8 | 55 | 116 | 0.47413793103448276 | 0 | 5 | 0 |
| holdout | adaptive:4 | 3 | 116 | 0.02586206896551724 | 0 | 1 | 0 |
| holdout | mean:2.0 | 2 | 116 | 0.017241379310344827 | 0 | 3 | 0 |
| holdout | random:1 | 1 | 116 | 0.008620689655172414 | 0 | 0 | 0 |

Location: {"layer": 10, "scope": "all"}
Selected: {"projected": "projected:8", "adaptive": "adaptive:4", "mean": "mean:2.0", "random": "random:1"}

Compute: 1256.2 seconds, 6696 forwards on Tesla T4.

Adaptive versus constant-mean guarded conversions: 3 versus 2.
The selected adaptive controller does not match the teacher's conversion count on this diagnostic set.
The adaptive controller has not demonstrated a consistent advantage over the selected constant across validation and diagnostic holdout in this run.

Zero guarded wrong-way changes follow from the acceptance rule and are not a general safety guarantee. The action-aware external Jev gate sees richer input than the original detector. No final-output copying or base-weight fine-tuning was used. This is a prompt-activation controller, not online learning or a free-text/agent evaluation.
