# Windows newline correction before any model run

The first version-4 source/data freeze was committed at `7a48884eaa9828fd9f27085f7b1c78947b2dc210`. A subsequent read-only reproducibility check found that Windows had written its JSON with CRLF line endings, while Git normalized the blobs to LF. The freeze files contained hashes of the local CRLF bytes and would fail on a Linux checkout. This was caught before version-4 setup, model inference, fitting, or trajectory collection.

The builders now write LF explicitly and `.gitattributes` fixes LF for version-4 JSON, Python, and Markdown. `normalize_frozen_json.py` performs one guarded migration: it confirms each JSON value is unchanged, normalizes line endings, refreshes source/data hashes in the freeze files and tokenizer-only preflight, then writes `data/NEWLINE_FIX_APPLIED.json` with old and new hashes. The prior exact bytes remain in the original Git commit. No training target, input meaning, case count, success criterion, model state, or experiment result changes.

The corrected source/data hashes are the version-4 pre-run freeze. Rerun all source/data tests and verify Git blob bytes against the committed hashes before launching any model worker. If any later source/data correction is needed, record it as another explicit pre-run revision; never silently replace a completed experiment's inputs.
