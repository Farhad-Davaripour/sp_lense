# Human-authorized replacement before collection — October 9, 2026

The user requested using the faster existing Research 2 checkpoint. No 27B
download worker, model loading, baseline generation or replay had begun. The
27B preparation is preserved in superseded_27b_preparation; it is not a result.
Reuse the connected owned A100 and the original allocation deadline.

Use only Qwen/Qwen3.5-0.8B, revision
2fc06364715b967f1860aea9cf38778875588b17, 24 blocks, residual width 1024.
Load frozen base weights in BF16 without quantization, SDPA, local files only,
Transformers 5.19.0, no adapter. Restore the exact public snapshot only if no
complete reusable Colab/Drive copy is available. Verify the pinned shard and
configuration/tokenizer hashes before loading.

The existing Research 2 vector.json has SHA256
3c611477efaa959314f4119d5b10b4dc9e3d5ebbfbfb49c4d5af0ef1c26e75aa.
Its original recorded_vector_fit.py registers on layers[10] (line 136): this
means zero-based decoder block 10, ordinary block 11. The direction has all
1024 coordinates and unit norm. It was previously fit using equal subtype means
of raw TRAIN action gradients, with positive sign increasing KEEP minus STOP.
Its original alpha was unselected; this pilot prospectively fixes strength 0.5
without calibration. The condition is labeled existing_vector. It is not the
27B JEV instruction vector and its provenance does not imply behavioral efficacy.
Apply a fixed additive 0.5 times unit direction, all positions, persistently
during each replay, matching the pilot intervention convention. The random
control uses the same block, strength and convention.

Primary token axis fixes ordinary block 16 (zero-based 15), downstream of the
injection. Depth uses all 24 blocks at positions 0, 15, 31 and 47 separately.
Eight prompts, fixed 4/4 split, 64-token cap, matched cached replay IDs, seeds,
ridge, 19 permutation controls per axis and 24 replay count remain unchanged.
No new vector fitting, training, JEV, sweep or additional model is authorized.

Generalize shard verification to the exact manifest shard set and the temporal
report key to its configured block. Synthetic and partial-output checks must
pass and final source hashes must be frozen before collection. Original
Research 2 and Research 3 assets remain read-only.
