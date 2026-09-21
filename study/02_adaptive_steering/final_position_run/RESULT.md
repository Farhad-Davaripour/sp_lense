# Adaptive activation steering result

State: completed

This is an exploratory comparison on reused scenarios. Oracle conditions use the same-prompt teacher; adaptive conditions use only base hidden states and TRAIN-fitted matrices.

Model and teacher weight hashes are unchanged. One candidate per method/view, identical guards.

| split | method | guarded_flips | initial_KEEP | conversion | guarded_control_changes | raw_control_changes | raw_wrong_way |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | teacher | 49 | 49 | 1.0 | 0 | 8 | 0 |
| validation | exact_oracle | 49 | 49 | 1.0 | 0 | 10 | 0 |
| validation | projected:1 | 31 | 49 | 0.6326530612244898 | 0 | 29 | 30 |
| validation | projected:4 | 49 | 49 | 1.0 | 0 | 10 | 0 |
| validation | projected:8 | 49 | 49 | 1.0 | 0 | 10 | 0 |
| validation | adaptive:1 | 29 | 49 | 0.5918367346938775 | 0 | 8 | 17 |
| validation | adaptive:4 | 48 | 49 | 0.9795918367346939 | 0 | 28 | 4 |
| validation | adaptive:8 | 48 | 49 | 0.9795918367346939 | 0 | 28 | 4 |
| validation | random:1 | 0 | 49 | 0.0 | 0 | 2 | 2 |
| validation | random:4 | 0 | 49 | 0.0 | 0 | 3 | 2 |
| validation | random:8 | 0 | 49 | 0.0 | 0 | 3 | 2 |
| validation | mean:0.5 | 9 | 49 | 0.1836734693877551 | 0 | 24 | 23 |
| validation | mean:1.0 | 31 | 49 | 0.6326530612244898 | 0 | 40 | 30 |
| validation | mean:2.0 | 39 | 49 | 0.7959183673469388 | 0 | 44 | 30 |
| holdout | teacher | 116 | 116 | 1.0 | 0 | 15 | 0 |
| holdout | exact_oracle | 116 | 116 | 1.0 | 0 | 16 | 0 |
| holdout | projected:4 | 116 | 116 | 1.0 | 0 | 17 | 0 |
| holdout | adaptive:4 | 113 | 116 | 0.9741379310344828 | 0 | 93 | 15 |
| holdout | mean:2.0 | 94 | 116 | 0.8103448275862069 | 0 | 100 | 74 |
| holdout | random:1 | 1 | 116 | 0.008620689655172414 | 0 | 1 | 2 |

Location: {"layer": 22, "scope": "last"}
Selected: {"projected": "projected:4", "adaptive": "adaptive:4", "mean": "mean:2.0", "random": "random:1"}

Compute: 1157.9 seconds, 6248 forwards on Tesla T4.

Adaptive versus constant-mean guarded conversions: 113 versus 94.
The selected adaptive controller does not match the teacher's conversion count on this diagnostic set.
The adaptive controller has numerically higher conversions on both evaluated splits; that alone does not establish a reliable advantage in this run.

Zero guarded wrong-way changes follow from the acceptance rule and are not a general safety guarantee. The action-aware external Jev gate sees richer input than the original detector. No final-output copying or base-weight fine-tuning was used. This is a prompt-activation controller, not online learning or a free-text/agent evaluation.
