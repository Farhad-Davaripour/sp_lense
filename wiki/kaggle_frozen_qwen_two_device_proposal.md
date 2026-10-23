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

## Frozen loader and collector library

`composite_frozen_probe.py` supplies the next library layer. It validates a
separate `FROZEN_QWEN_AUTHORIZATION.json`, source/input/profile hashes and the
original study transport before loading. The recognition recovery exception
does not authorize this two-device attempt. A proposal, missing authority or
changed devices/bounds fails before model loading.

The library reuses `Actor` for the original tokenizer, NF4/double/BF16 loader,
reference kernels, greedy native generation, EOS decoding and cache boundaries.
The normal single-device path is unchanged. The optional frozen path checks the
entire logical parameter set and explicit device map, rejects missing or
mismatched pretrained weights, and forbids adapter installation/restoration,
training contexts, teacher-forcing losses, ON-branch claims, shorter generation
budgets and alternative kernels. The per-instance forward ceiling is 400 and
the guard checks a 512 MiB free export reserve on both devices before forwards.

Two predeclared full training observations are selected in immutable dataset
order: initial quiet-pending and initial own-pending. Selection starts from the
complete original 720-row cohort, not a shortened or relabeled input. This is
an engineering qualification, not a reduced version of full study validation.
No returned native action is executed in the simulator or on the host.

The read-only collector exports raw last-token vectors from all 64 blocks,
full-vocabulary logits, original token IDs/native output, actual per-device
memory observations, fresh-cache identity provenance, and frozen parameter /
quantization-buffer fingerprints before and after. Prefill evidence is persisted
before generation; a later failure must preserve completed artifacts. The raw
auditor replays tokenization, EOS/native parsing, activations, ownership,
fingerprints and exact bounded forward accounting without a model forward.
It explicitly leaves cross-device numerical parity and full validation unproven.

The original public 17-file model pin is retained as a reviewed control file,
`config/composite_qwen_file_pin_fc05.json`, with a fixed canonical digest. Both
the source/runtime manifests and actual downloaded config/index/weight proofs
must match it. The shared byte verifier is reused at the loader boundary. Small
metadata files are exported as exact base64 bytes and checked against their
original Git blob and SHA proof; pretrained tensor files are never exported.
Mutually editing two mutable metadata documents is not an authorization to use
another model or configuration.

The layout proof explicitly requires unquantized BF16 input embeddings and
output head, and records the NF4/double-quant state, BF16 compute, UInt8 storage,
64/256 block sizes and device ownership of every expected quantized projection.
These checks follow the pinned bitsandbytes 0.49.2
[QuantState](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/functional.py#L412)
and [Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/nn/modules.py#L391)
interfaces. Raw generation replay rejects tokens after EOS, an early non-EOS
stop, a missing/nonempty initial cache and disagreement between flattened and
native cache provenance. Such checks verify evidence fidelity, not task success.

Model files must use the source-bound `/kaggle/temp/research4_frozen_probe_cache`
namespace rather than persistent notebook outputs or another study. A future
worker must own the whole-stage protected timer, external hard timeout, disk
admission, atomic single-use submission allowance, exclusive output directory,
failure exports and cleanup. Those worker/submission paths are not enabled by
the library; existing Kaggle admission remains recognition-only. No actual
27B model or second-device scientific run has been loaded or qualified yet.
