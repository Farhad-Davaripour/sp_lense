# CPU capacity check requested during execution

The user asked whether raising the worker limit would speed training. At the check, the memory-optimized worker used approximately 7 GiB under its 12-GiB cap, with no cgroup memory-limit events. The laptop reported 14 logical CPUs, about 79% overall CPU load, and about 2.4 GiB free physical RAM. Increasing the memory cap was therefore not chosen.

After the current matched training stage finishes, a separate engineering worker will compare four and six Torch threads under a six-CPU quota. It uses the same read-only model/runtime and the same filesystem, network, process, output, and cancellation restrictions. Its separate boundary must pass all isolation probes before the benchmark. No concurrent model jobs are permitted.

Use the longest tokenized training example in each of the three curriculum components. Perform a fixed warmup and forward/backward measurements in counterbalanced thread order 4, 6, 6, 4. Never call an optimizer step, generate model responses, or open validation/test examples for scoring. Export only timings and diagnostic loss values; no model weights are changed or exported. The four-thread measurement has the same six-CPU quota as the six-thread measurement, so this directly measures thread-count effects, not every difference from the earlier four-CPU quota.

Before observing benchmark results, choose six threads for subsequent training only if its mean total time over the three examples is at least 10% lower, with no memory-limit failures. Otherwise keep four threads. Generation remains at four threads because this benchmark measures training, not autoregressive inference. Record CPU settings and boundary receipts for every cohort. Dataset, adapter targets, optimization schedule, decoding, and success criteria remain unchanged. This is an engineering comparison on one laptop, not a claim about hardware in general.

Reproduction after the active training stage ends:

```sh
sudo python3 setup_benchmark.py
sudo python3 /opt/sp-lense-r3-thread-benchmark/code/isolation/supervisor.py suite thread-gate01
sudo python3 /opt/sp-lense-r3-thread-benchmark/code/isolation/supervisor.py run threads46 --spec benchmark.json --seconds 1800
```
