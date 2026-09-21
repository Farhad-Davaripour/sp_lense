# Research 2 feasibility result

Exploratory benchmark; held-out data are reused diagnostic evaluation.

| split | method | KEEP to STOP | wrong way | control changes | eligible flips | eligible cohort | coverage | end-to-end conversion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | Research 1 | 2 | 0 | 0 | 2 | 39 | 0.05128205128205128 | 0.04081632653061224 |
| validation | LoRA raw | 49 | 0 | 8 | 39 | 39 | 1.0 | 1.0 |
| validation | LoRA gated | 39 | 0 | 0 | 39 | 39 | 1.0 | 0.7959183673469388 |
| validation | LoRA guarded | 39 | 0 | 0 | 39 | 39 | 1.0 | 0.7959183673469388 |
| validation | oracle guarded | 0 | 0 | 0 | 0 | 39 | 0.0 | 0.0 |
| validation | training_mean guarded | 0 | 0 | 0 | 0 | 39 | 0.0 | 0.0 |
| validation | random guarded | 0 | 0 | 0 | 0 | 39 | 0.0 | 0.0 |
| holdout | Research 1 | 2 | 0 | 0 | 2 | 84 | 0.023809523809523808 | 0.017241379310344827 |
| holdout | LoRA raw | 116 | 0 | 15 | 84 | 84 | 1.0 | 1.0 |
| holdout | LoRA gated | 84 | 0 | 0 | 84 | 84 | 1.0 | 0.7241379310344828 |
| holdout | LoRA guarded | 84 | 0 | 0 | 84 | 84 | 1.0 | 0.7241379310344828 |
| holdout | oracle guarded | 0 | 0 | 0 | 0 | 84 | 0.0 | 0.0 |
| holdout | training_mean guarded | 0 | 0 | 0 | 0 | 84 | 0.0 | 0.0 |
| holdout | random guarded | 1 | 0 | 0 | 1 | 84 | 0.011904761904761904 | 0.008620689655172414 |

Recommendation: pause block-10 transfer; inspect downstream adapter contributions before one bounded diagnostic.

Oracle transfer uses teacher activations for the same prompt. It is not a teacher-free controller.
Zero wrong-way flips under the guard are enforced by score selection, not a general safety guarantee.
See METRICS.json for SELF/OTHER, answer order, probability mass and exact scenario IDs.

A three-forward validation diagnostic verified that the patched block-10 hidden state matches the teacher hidden state (maximum absolute error 2.3283064365386963e-10). The patched frozen model still chose KEEP. LoRA modifies attention projections at blocks 3, 7, 11, 15, 19 and 23; downstream adapters or other token positions may be needed. This is a failure of the tested single-position transfer, not evidence against all activation transfer.

Compute: 1017.9 seconds of experiment wall time on Tesla T4; 349.7 seconds training. One seed/configuration, no research repair. Device-utilization time and billed GPU/API cost were not available.

Next proposed diagnostic (not run): on a small, predefined validation subset, disable only the trained adapters downstream of block 10 and measure how much teacher benefit remains. Do not start controller fitting or a layer search from this result.

Prefect publication is blocked by the existing local server's Windows Application Control failure. comparison.json is the prepared table artifact; no replacement dashboard was built.
