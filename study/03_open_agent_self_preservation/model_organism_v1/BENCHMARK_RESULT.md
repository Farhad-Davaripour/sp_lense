# Local CPU benchmark result

The user asked whether increasing the worker limit would speed training. The memory-optimized fits were below the 12-GiB cap, so this test compared Torch thread counts while retaining the memory cap.

| Threads | Mean time for the same three training examples |
|---|---:|
| 4 | 36.8771 seconds |
| 6 | 34.1233 seconds |

Six threads reduced time by **7.47%**. The frozen rule required at least a 10% reduction, so **training and inference remain at four threads**. This small benchmark does not establish a reliable speed gain for a full training run or for autoregressive generation. It compares thread counts under a common six-CPU ceiling, rather than directly comparing every effect of four- versus six-CPU quotas.

The benchmark used the longest training sequence in each curriculum component: 109, 662, and 622 tokens. Thread order was 4, 6, 6, 4, with a warmup in each block. There were 12 timed measurements and four warmups. No optimizer steps or validation/test evaluations occurred. The job completed successfully in 161.64 seconds and its worker processes exited before export.

Both the six-CPU benchmark boundary and the updated four-CPU study boundary passed all 21 checks, including resource verification before payload startup. Failed startup-probe attempts and the original boundary sources remain preserved; see [STARTUP_GUARD.md](STARTUP_GUARD.md).

Raw timings, decision calculation, source/input provenance, isolation receipts, and all failed probe diagnostics are retained in the full capture under `thread_benchmark/`.
