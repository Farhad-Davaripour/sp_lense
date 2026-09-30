# GPU batching follow-up

The active comparison uses microbatch 1 with four accumulated examples per
optimizer update. The A100 reports about 28/80 GiB allocated during training.
That is memory use, not a measurement of GPU compute utilization. System RAM is
not currently a model-offload or data-loading bottleneck: weights and prepared
training tensors are on the GPU.

The candidate processes 2 or 4 examples in parallel, retaining effective batch 4,
the same update membership, scheduler, and equal-per-example mean loss. Padding
is masked; examples are sorted only within each four-example update to reduce
padding. Do not replace the loss with a global token mean, which would alter
example weighting. Do not modify the current trained adapter or mix performance
settings across its matched controls.

After the active campaign exits, run `benchmark_batches.py` on the same Colab GPU
within the remaining compute budget. It compares 1/2/4 in mirrored order, using
one warmup and three timed updates per setting. Record speed, peak memory, and
OOM failures; discard benchmark adapters. It is prepared, not yet executed.
Validate batched losses/gradients on Colab before adopting the faster candidate
for a new consistently configured comparison. No laptop model/CPU prerequisite.

The runtime also reports reference implementations for causal convolution and
gated delta attention. Optimized kernels and BF16 LoRA without 4-bit quantization
are possible later performance comparisons; neither is being changed underneath
the active experiment. More allocated RAM alone is not a speedup claim.
