# Exact-head v2 lossless logit publication note

After the frozen v2 evaluation completed, inspection found that each serialized
one-dimensional final-token logit tensor still shared the full-sequence transformer's
backing storage. Although every saved tensor had the intended shape `[248320]`,
PyTorch preserved the larger backing storage and produced files of roughly 160–174
MB each. The 128 immutable source files total `21,312,040,064` bytes.

This was a storage defect, not a scoring or experimental defect. The frozen
checkpoint had already bound both each source file SHA-256 and the SHA-256 of the
248,320 float32 values used for every metric.

The post-outcome publisher in
`scripts/compact_interface_equivariant_exact_head_logits.py` did not overwrite the
source files. It loaded each source vector, cloned the same float32 values into an
independent minimal storage, wrote a separately named compact artifact, and verified:

1. exact elementwise equality;
2. the same frozen raw-tensor SHA-256;
3. the same exact argmax and semantic choice;
4. the same preserve-minus-comply log-odds, pair probability, answer-pair mass, and
   full-vocabulary KL; and
5. a manifest link from the original file path/hash to the compact path/hash.

All 128 vectors passed. The compact copies total `127,364,224` bytes, or
`0.005976162939705705` of the original serialized size. The originals remain local
and untouched; the compact copies and their manifest are the publication artifacts.
No model call, retry, steering change, score change, or claim change occurred during
this post-outcome operation.
