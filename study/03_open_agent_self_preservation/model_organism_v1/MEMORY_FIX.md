# Pre-evaluation implementation correction

The first attempted fit, `p1-fit-preservation`, was terminated by the independent 12-GiB memory cap after 133.46 seconds. It exported provenance and a first-step loss report, but no adapter. The controller stopped the campaign. No validation or behavioral output had been generated or inspected. The failed run and original source snapshot are retained.

Before restarting from the same seed and baseline, two memory optimizations were made for every arm and both candidate checkpoints:

- Enable non-reentrant activation checkpointing, recomputing forward activations during backward propagation.
- Calculate output logits only at positions with supervised next-token labels. Prefix positions had ignored labels before this change, so the supervised cross-entropy objective and target-token denominator remain the same.

The installed Qwen implementation explicitly supports a tensor of `logits_to_keep` positions. The checkpointing option follows the [PyTorch documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html). This changes the training implementation, not the corpus, optimizer schedule, parameter targets, seed, evaluation rubric, or success thresholds. The new jobs carry a `-mem2` suffix; no failed run is overwritten. Memory, CPU, filesystem, network, and stop boundaries remain unchanged. Only source implementation files are replaced in the operational mount; both source versions remain archived and in Git history.
