# A100 batching measurement and applied execution revision

The GPU benchmark ran on the same Colab A100 after the user requested immediate
optimization. The completed preservation adapter and all its first evaluation
records were retained. The unfinished continuity fit was deliberately interrupted
after its last logged microstep 352/856 and restarted fresh in a new run directory;
it was not resumed from the unvalidated training-state path.

Each configuration used the same four-example optimizer update, the same fixed
input groups, one warmup plus three measured updates, and two mirrored repetitions.

| Physical microbatch | Effective batch | Median update time across measurements | Peak allocated GPU memory |
|---|---:|---:|---:|
| 1 | 4 | 11.51 s | 28.37 GB |
| 2 | 4 | 6.33 s | 29.27 GB |
| 4 | 4 | 6.53 s | 32.12 GB |

Batch two reduced measured update time by about **45%** (about **1.82× throughput**).
The first 32 examples of the restarted control fit took 50.97 seconds, versus
93.65 seconds in the serial fit. More occupied memory is not itself a throughput
measurement; batch four used more memory and was slightly slower on these groups.

Initial per-example losses passed the declared closeness check. The strict
gradient-cosine diagnostic did **not** pass for batches two/four: batch two was
approximately .994 versus the .995 threshold; the repeated serial baseline was
.999919. These failures remain recorded, and the configuration is not described
as numerically equivalent. It was adopted as an explicit performance revision
with the same mathematical per-example objective, optimizer-update membership,
effective batch size, learning-rate schedule, data, and scientific criteria.
Within each four-example update, examples are sorted by length for batching;
the phrase `training_order_and_update_membership_unchanged` in the initial launch
metadata refers to update order/membership, not identical within-update ordering.

The new run uses microbatch two/accumulation two for fresh continuity and neutral
fits. It retains the completed microbatch-one preservation checkpoint and repeats
its evaluation with the same batched inference used for all four arms. Independent
inference requests use batches up to four, with separate histories/states, left
padding, per-row EOS handling, and the original token/turn caps. All physical
batching and numerical differences must be disclosed with the exploratory results.
Any confirmatory training replication should use one configuration for all fits.

Original private run:
`/content/sp_lense_work/qwen38_simple_v1_20260930T124815Z_1f3e2372`.
Benchmark:
`performance/batches_1790781447283610861` within that run.
Applied run:
`/content/sp_lense_work/qwen38_batched_v1_20260930T153316Z_4460df35`.
Raw timing/loss/gradient records remain in the private run. No laptop model test
was used. System RAM was not a data-loading/offload bottleneck in this workload.
