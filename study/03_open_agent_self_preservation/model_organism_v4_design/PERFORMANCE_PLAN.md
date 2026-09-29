# Laptop resource plan for the next frozen experiment

The user authorized prioritizing training speed while the laptop is otherwise idle. Do not change the currently running checkpoint-3 thread count or cgroup limits: its first candidate and the frozen second-pass protocol use four Torch threads, four CPU cores, and a 12-GiB worker cap. Record the new resource choice before starting a separate version-4 fit.

## Observed headroom in checkpoint 3

During the restarted preservation fit, the worker had a 12-GiB `memory.max` and a four-core `cpu.max`. Measured current memory was about 7.5 GiB; peak was about 7.7 GiB. `memory.events` reported zero high/max/OOM/OOM-kill events. CPU usage was close to four cores and accumulated quota-throttled time was under 1% of CPU usage. WSL exposed about 16 GiB of guest memory, while Windows reported about 3.6 GiB free physical memory at that reading. These are snapshots, not a guarantee about every later workload. They indicate that extra RAM would not speed this fit, and raising the cap toward the laptop's physical maximum would unnecessarily reduce host headroom.

The earlier isolated 4-versus-6 thread test found a 7.47% mean reduction on three long examples. Its then-frozen 10% threshold kept four threads. The user's new instruction permits choosing a faster verified setting for a **future, separately frozen** experiment, even if the gain is below 10%, provided the laptop remains stable. It does not retroactively change checkpoint-3 settings or results.

## Benchmark before version 4 fitting

After checkpoint 3 and its model workers have exited, run a separate isolated benchmark with the same 0.8B architecture, representative short and long supervised examples, and the same memory/IO restrictions. Compare Torch intra-op thread counts 4, 6, 8, 10, and 12 under an equal 12-core ceiling, recording warmups, repeated wall times, CPU time, peak memory, throttling, thermal stability if available, and host free memory. Use balanced order to reduce warm-cache/order bias. Include backward propagation and optimizer-update timing. Choose the fastest setting with a repeatable gain and acceptable host headroom, then freeze the chosen CPU quota/thread count and its evidence in the next protocol. Do not run this benchmark alongside the live checkpoint-3 fit; competing for CPU and memory would slow the authorized experiment and weaken the comparison.

Retain a memory cap with room above measured peak, zero worker swap, process/output/time watchdogs, and all namespace/seccomp boundaries. Increase the memory cap only if a later representative workload actually approaches it and Windows has adequate physical headroom. Never equate a higher configured maximum with faster computation when no memory pressure is observed.

## Completed decision

The initial five-setting sweep was unstable and retained four threads by its frozen rule. A separately frozen 4-versus-10 confirmation with two warmups and four measured blocks per setting passed its rule: ten threads reduced median three-example training time by 13.9%, all four ten-thread blocks beat the four-thread median, ten-thread spread was 7.0%, and peak memory was about 6.0 GiB with no limit events. The next training revision should freeze ten Torch threads under the tested 12-core ceiling and retain the 12-GiB cap. Autoregressive generation has not been benchmarked at higher threads, so its four-thread setting remains an independent decision. See the two benchmark evidence directories for receipts and exact timings.
