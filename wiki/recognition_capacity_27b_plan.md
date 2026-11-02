# Frozen 0.8B versus 27B recognition comparison

The completed 0.8B campaign does not establish that the failures extend to more
capable models. Test this alternative explanation directly with the same frozen
revision and numeric cases. Select Qwen3.5-27B rather than Research 3's Qwen3.8-27B
to keep the model family fixed. Do not load a research fine-tuning adapter.

The official Qwen3.5-27B checkpoint is pinned to
`fc05daec18b0a78c049392ed2e771dde82bdf654`. Its eleven safetensor files total
55,563,022,432 bytes. The reference 0.8B pin remains
`2fc06364715b967f1860aea9cf38778875588b17`. Architectures have 24/64 text blocks
and hidden widths 1,024/5,120. The model source and file identities are preserved
in the capacity packet. No Research 3 worker, checkpoint or file is changed.

## Comparison and scoring

Run the 0.8B control and 27B sequentially in one GPU environment, both unquantized
BF16, with the same exact prompts, raw greedy numeric decoding, native EOS,
eight-token numeric cap, no cache, disabled thinking and SDPA math attention.
This controls the device/precision change relative to the old FP32 CPU results.
Model training histories still differ; this is a matched model-size comparison,
not a controlled causal experiment on parameter count alone.

Each model receives 288 binding queries: the same 72 revision observations
(12 identity cases, simple/chronological/reverse histories, both role assignments),
two option wordings and two answer orders. Each also receives the same 36 numeric
prompts. There are at most 576 forwards per model, 1,152 total. The originally
defined canonical role contrast is primary; reversed/mean contrasts and direct
semantic means are all reported. Numeric completion-only exact integer scoring
is primary; complete first/last integer parses are both secondary. Ties and
truncations stay in denominators. Do not tune the question, equation, parser,
tie band or answer order on these outcomes.

The preparation audit confirms all 288 binding prompt hashes match the archived
questions, all numeric case bytes are unchanged, rendered chats are identical
across model sizes, and all 144 role pairs across interfaces/orders align in
total length and notice-start position for each tokenizer. No model forwards
were required for this check. These cases are already consumed diagnostics,
not a new held-out generalization test. Original f09/f10/f11 remain unopened.

Capture the final whole-block state at the last input position with the model's
native hidden width. Do not pretend that block 23 of a 64-block model is the
same endpoint as block 23 of a 24-block model. Preserve raw letter scores,
generated tokens, decoding, per-prefix state keys and before/after base hashes.
An independent CPU audit reconstructs decisions, paired-case counts, stopping,
numeric parses, tokenization and feature alignment from the exports.

## Resource admission and cleanup

Native 27B BF16 requires a larger GPU than the former A100-40GB runtime. Admit
only at least 72 GiB total / 64 GiB free GPU memory, 12 GiB free host RAM and
75 GiB free ephemeral disk. A Colab G4 is the candidate route; availability and
its actual memory must be checked before model loading. No automatic fallback
to quantization or another paid platform is allowed. Hardware rejection is a
failed admission, not a model result.

The worker permits 40 minutes including its download/load work, reserves three
minutes for export and accepts an earlier absolute allocation deadline. Save
results in a dedicated Research 4 Drive folder, verify the result archive and
release the owned GPU promptly. No recurring monitor, retry, purchase, serving
endpoint, adapter training or model action loop is part of this experiment.

The repository's no-paid-compute default remains in effect until the user grants
a specific existing-credit GPU budget. Preparing this packet, notebook and tests
does not constitute a GPU run or authorize spending. Prepared results are unknown.

## Interpretation

If matched 0.8B still fails but 27B succeeds, capacity/model capability becomes a
supported explanation on these cases. If both fail, the narrow protocol or the
readout itself remains a candidate limitation. If the 0.8B GPU control changes
substantially from its archived FP32 CPU outcome, report that numerical/interface
sensitivity before attributing the difference to size. A positive result would
justify a fresh frozen confirmation cohort; it would not establish a universal
energy equation or a broadly reliable detector by itself.
