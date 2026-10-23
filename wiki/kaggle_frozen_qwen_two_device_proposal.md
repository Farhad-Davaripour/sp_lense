# Frozen 27B Qwen placement proposal

The unchanged single-device conditional-generation loader cannot fit the
observed Kaggle T4. A metadata-only lower bound is 17,492,559,480 bytes before
NF4 metadata, normally unquantized tensors, adapters, activations and runtime
workspaces. The observed usable device capacity is 15,636,037,632 bytes.

The first 17.7 GB checkpoint-wide calculation included checkpoint-only `mtp.*`
keys. The pinned Transformers 5.19.0 reference class already ignores those
unexpected keys; excluding them corrects the runtime lower bound to 17.5 GB.
This follows the original loader and does not remove a runtime model component.

Pinned tensor headers suggest a balanced candidate: complete decoder blocks
0–17 on device 0 and 18–63 on device 1. Vision, input embeddings, rotary module
and output head remain on device 0; final text norm is on device 1. Optimistic
weight bounds are 8,743,381,400 and 8,749,178,080 bytes. These are lower bounds,
not measured runtime memory or evidence that the full context fits.

The model revision, all 64 blocks, retained rank32/alpha64 capacity definition,
NF4/double quantization, BF16 compute, reference kernels, complete tokenization,
6144-token ceiling and 160-token greedy generation semantics remain unchanged.
No CPU/disk offload, context reduction, behavior fitting or reward training is
introduced. The profile is limited to frozen-base activation analysis.

`composite_placement.py` supplies a deterministic device map, metadata coverage
checks and a raw activation/fingerprint/cache-boundary auditor. Its tests use
small synthetic parameter records and vectors; they never instantiate Qwen,
load pretrained weights, allocate a GPU or execute a model forward. They do
not establish runtime fit, numerical parity, native generation correctness,
full scientific validation or held-out admission.

No loader or scientific worker admits this candidate. Before any allocation,
the retained one-visible-GPU limit needs an actual human exception, and a
reviewed collector/loader must bind the new source, device map and training-only
engineering prompts. Actual qualification must measure both devices' free and
peak memory, verify every original parameter and quantization buffer before
and after, export raw finite logits/activations from all 64 blocks, independently
replay tokenization and native output decoding, and verify fresh response caches.
Complete later evaluation must use its own exact model/config/platform freeze;
single-device or reference-checkpoint admission cannot be inherited.

This source preparation grants no execution authority. No additional GPU run,
scientific retry, held-out opening, paid resource or stopped schedule is started.
