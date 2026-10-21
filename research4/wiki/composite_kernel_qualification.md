# Optional runtime qualification

The pinned Qwen runtime currently uses PyTorch reference Gated DeltaNet cores.
The official [Transformers documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)
describes optional acceleration, and the installed 5.19.0 source exposes explicit
dispatch/fallback wrappers. The [FLA installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
separates kernel packages from model layers and backend dependencies.

The prepared candidate pins `fla-core==0.5.2`, `einops==0.8.2` and
`triton==3.6.0`, retaining Torch 2.11.0+cu128, Transformers 5.19.0, NF4/BF16,
SDPA and the exact Qwen/model-adapter revisions. Wheel digests are frozen in the
qualification config. A matching causal-convolution wheel was not found in the
checked official v1.7.0 release; both profiles therefore explicitly use the
reference convolution. No kernel package is installed locally for model tests.

One Databricks worker will use one frozen Qwen backbone and its existing
behavior adapter. It switches between explicit reference and verified FLA GDN
cores only between responses, with fresh caches. Unavailable FLA dispatch is an
error rather than a silent fallback. The qualification uses two existing
training observations, both adapter states, separate warmups, complete bounded
generation records and saved logits. It executes no simulator actions or
optimizer updates and reads no validation/held-out content.

Adoption requires identical bounded generated token IDs/EOS status for every
pair, at least 1.2x measured speed in every pair, unchanged pretrained parameter
bytes/buffers/quantization state, and same-instance exact OFF checks in each
profile. Timing excludes recorded warmups. Token identity under these bounded
checks does not prove universal numerical/behavioral equivalence. Even an
admitted qualification does not automatically adopt the candidate: any future
full validation and held-out package must explicitly freeze its kernel profile.

This is prepared engineering work. It has no scientific results yet and does
not alter the earlier frozen actor diagnostic or detector failures.
