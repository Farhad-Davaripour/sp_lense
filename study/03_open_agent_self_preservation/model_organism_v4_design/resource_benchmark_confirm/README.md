# Follow-up 4-versus-10-thread confirmation

The [initial isolated benchmark](../resource_benchmark/README.md) tested 4, 6, 8, 10, and 12 threads after checkpoint 3. Ten threads had a 22% lower two-block mean than four, but its own blocks differed by 21%, above the predeclared 15% stability limit. That benchmark therefore chose four threads. This **separately declared** confirmation checks the concrete remaining risk that ten threads offer a repeatable training gain after sufficient warmup. It does not retroactively change the first decision or any checkpoint-3 run.

**Completed result:** ten threads passed the confirmation rule. Its median three-example training time was 45.23 seconds versus 52.54 seconds at four threads (13.9% lower), all four ten-thread block totals beat four's median, and its range was 7.0% of its own median. The isolated worker had a 6.0-GiB peak under a 12-GiB cap with no memory-limit events. Choose ten Torch training threads under the measured 12-core quota for the next separately frozen fit; this benchmark makes no inference-throughput claim. The post-run reader needed one path-only fix recorded in [ANALYSIS_FIX.md](ANALYSIS_FIX.md).

Under a common 12-core cgroup quota, fixed order is 4, 10, 10, 4, 4, 10, 10, 4. Each block has two discarded warmups and timed forward, backward, clipping, and AdamW updates on the same longest three *training-only* examples. The original Qwen base remains frozen, and no changed adapter is exported. Memory stays capped at 12 GiB, zero worker swap, with the full isolation suite required again on this fresh root.

Before seeing results, choose ten threads for a future fit only if all are true: its median three-example time is at least 5% below four's; at least three of its four block totals are below four's median; its four-block range is at most 20% of its median; and no memory-limit/worker failure occurs. Otherwise retain four. This resolves the pilot's instability with more warmup and repeated blocks, while accepting that short engineering timings still may not predict every full training run. Generation speed remains untested. Do not run more optional thread sweeps unless a specific later failure requires it.

After checkpoint-3 workers and the first benchmark exit:

```sh
sudo python3 setup_local.py
sudo python3 /opt/sp-lense-r3-capacity-confirm/code/isolation/supervisor.py suite r3confirm-gate01
sudo python3 /opt/sp-lense-r3-capacity-confirm/code/isolation/supervisor.py run r3confirm-threads --spec benchmark.json --seconds 1800
sudo python3 analyze.py
```
