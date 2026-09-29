# Isolated CPU capacity benchmark

This is engineering measurement for a future Research 3 revision, not model-organism evidence. It runs only after all checkpoint-3 workers exit. It uses three longest examples from the already frozen checkpoint-3 **training** corpus, never validation or agent-evaluation cases. The tested adapter is copied from checkpoint 2; any benchmark optimizer updates remain inside the temporary worker and are not exported as a trained model.

Before observing timings, fix the block order to 4, 6, 8, 10, 12, 12, 10, 8, 6, 4 Torch intra-op threads under the same 12-core cgroup ceiling. Each block includes one warmup and timed forward, backward, clipping, and AdamW updates on the same three examples. Compare the mean total time of the three timed examples across both blocks at each thread count. Choose the fastest count only if both of its block totals beat the mean four-thread total by at least 3% and there is no memory-limit or worker failure. If no count meets that rule, retain four. If the two blocks for the apparent winner differ by more than 15%, report uncertainty and rerun a separately frozen benchmark before choosing. The selected count is for **future training only**; this measurement does not establish generation speed.

Keep the same 12-GiB memory cap, zero worker swap, process/output/time watchdogs, root-owned supervisor, unprivileged Bubblewrap worker, no network sockets, read-only inputs and model, and no host drive access. The boundary must pass its full isolation suite under the new 12-core ceiling before running the model. The laptop uses no concurrent model worker during this measurement. Memory is raised in a future experiment only if actual measured peak approaches the cap and the Windows host has headroom.

Reproduction in WSL after checkpoint-3 export:

```sh
sudo python3 setup_local.py
sudo python3 /opt/sp-lense-r3-capacity-benchmark/code/isolation/supervisor.py suite r3capacity-gate01
sudo python3 /opt/sp-lense-r3-capacity-benchmark/code/isolation/supervisor.py run r3capacity-threads --spec benchmark.json --seconds 1800
sudo python3 analyze.py
```

Setup refuses existing roots. The analysis reads only the immutable receipt and benchmark rows. The chosen setting must be frozen in any new training protocol before fitting.
