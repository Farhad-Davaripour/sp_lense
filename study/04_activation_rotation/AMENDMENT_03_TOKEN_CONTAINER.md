# Native collector compatibility correction — October 9, 2026

First native attempt stopped before baseline generation or residual collection:
RuntimeError: Could not infer dtype of tokenizers.Encoding. Model pins passed;
the frozen 0.8B BF16 checkpoint loaded. Preserve its complete fault receipt,
interventions and logs under the original attempt directory and private Drive.

Normalize apply_chat_template output containers to the exact existing integer
ID sequence before constructing the Torch tensor. Cover Encoding, mappings,
single-row and plain-list containers with model-free regression checks; reject
multiple examples. No prompt/token content or generation/intervention/analysis
settings change. A new output directory is mandatory for the corrected attempt.
Continue in the same owned session with the verified cached snapshot and original
allocation deadline; do not redownload weights or allocate another runtime.
This correction is announced prospectively, before any Qwen data was generated.
