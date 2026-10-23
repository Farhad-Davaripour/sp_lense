# Kaggle composite runtime

The user selected Kaggle after direct GPU, CUDA, BF16, checkpoint persistence,
and pinned-package engineering checks passed. This changes execution and
artifact transport; it does not change the frozen recognition recipe.

Set `RESEARCH4_EXECUTION_PLATFORM=kaggle` only inside the Kaggle container.
The scientific child exposes `CUDA_VISIBLE_DEVICES=0`: Kaggle provisions a T4
pair, while this stage retains one visible computational device. T4 has no
native BF16 acceleration; actual BF16 and tiny NF4 operations were tested.
The full Qwen3.5-27B memory fit remains unverified.

Freeze `inputs/EXECUTION_PLATFORM.json` with the Kaggle account, notebook,
source commit, stage, input root, transport-manifest SHA, and fresh destination.
Only `detector-rank32-v5` is admitted by this version. Behavior fitting,
evaluation and held-out access are unavailable through the Kaggle route.

The transport manifest lists exact SHA-256 digests of the required existing
training/development artifacts. `Store` reads only registered files, rejects
held-out paths and changed bytes, and cannot write into the scientific inputs.
The worker writes exclusive output files below
`/kaggle/working/research4_campaign/runs`, verifies readback, and retains the
same run receipt, checkpoint, failure and evidence-archive formats used on
Databricks. An existing destination is never reused or automatically retried.

Keep the 45-minute external deadline, 2400-second worker bound, protected export
reserve, 30,000-forward ceiling, full data splits and all-block rank32/alpha64
ModernBERT coverage. Save and checksum-recover Kaggle outputs before releasing
the owned session. Keep source/PR admission separate from scientific admission;
a completed fit does not establish validation or held-out success.
