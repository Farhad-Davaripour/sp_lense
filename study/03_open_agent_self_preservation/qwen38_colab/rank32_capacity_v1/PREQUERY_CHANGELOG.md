# Pre-query candidate change log

The unqueried candidate bundle `95a832694eaee75c09b73e0c555699224613b9d9c508a5abcde7b6a205a6e4da` and its source freeze/checksum were archived byte-for-byte under `unqueried_candidate_archive/95a832694eaee75c`. No rank32 neural query or update used it.

Two metadata corrections supersede that candidate: the final exporter includes both the unexported parent run and the capacity run before releasing the runtime; the embedding receipt reports old factor capacity, old frozen trainable count, new factor capacity and actual rank32 trainable count separately. Weights, data, dose, decoding and scientific gates are unchanged. Run `build_notebook.py --bundle-only` explicitly to create a new candidate; the notebook pinning mode refuses silent regeneration.
