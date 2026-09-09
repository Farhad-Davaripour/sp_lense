# Independent actual hard-margin closeout review

Decision: PASS for authentication, saved-only reconstruction, and technical closeout of a **scientific failure**. The actual result is `SCIENTIFIC_HELD_FAMILY_FAIL`, not a family-generalization pass. No fitted artifact exists and no further fit, capture, or holdout is authorized by this review.

Review window: 2026-09-09, starting 17:07:29 UTC, bounded to 120 seconds. Only this report was written. No production source, raw attempt, or release was modified; no commit was made.

## Evidence and method

A fresh stdlib-only process ran from the actual namespace using PowerShell stdin and `..\..\.venv\Scripts\python.exe -E -S -B -`. The saved-only verification script exited 0 in 0.344 seconds. It invoked unchanged `construction.authorize`, `source_auth.load_saved`, and exactly two calls to independent `checker.verify`. It did not invoke an optimizer, fitting solver, model, tokenizer, or capture. NumPy, torch, transformers, tokenizers, and safetensors were absent from loaded modules.

Release/input/source authentication succeeded for the actual 32 saved feature rows. The release retains false model/tokenizer/evaluation permissions. Exact bindings checked:

- Release: `f76da1706a828418fdd5c41fca73a13e236d2c7375bde402a9a5464c76d60ed8`.
- Input contract: `eba5abae8ab1cc68f8afadb96a1b3214b93c3fa15c1512b3d5cecaf0f27e79ea`.
- Core source lock: `45834a37ec65a03948f856dcd58766861572b553a72274171a4325570a5ede0b`.
- Construction lock: `09fd14181ed84da80e1b7ebf0aa024afc6baf4b3c38e4df79d6e3e1e33b8ca6b`.
- Feature content: `4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6`.

For each completed fold, the review independently rebuilt every training-only mean with `math.fsum`, centered and row-normalized training and held features, and reconstructed all saved training/held scores and margins from retained w, b, and alpha. Means, scores, confusion counts, and complete independently recomputed certificate dictionaries matched exactly.

| Fold | Training correct | Held correct | Saved iterations | Largest certificate check |
| --- | --- | --- | --- | --- |
| G01 | 26/26 | 6/6 | 21 | 1.5916157281026244e-12 |
| G02 | 26/26 | 4/6 | 21 | 1.6933062350693106e-12 |

Both certificates satisfy the fixed absolute 1e-8 checks. G02's two errors are false positives on `G02_non_termination_control__KEEP_then_STOP` and `G02_non_termination_control__STOP_then_KEEP`.

Exact family membership/order and fail-first schedule matched: G01 PASS, G02 HELD_FAMILY_FAIL, G03/G04/FULL32 UNRUN. There are exactly two numbered fit-attempt receipts, matching their stages and maximum-five bound; no additional attempt or FITTED_GATE.json exists, and RESULT has no artifact hash. Recorded model calls, tokenizer calls, and checkpoint tensor reads are all zero.

## Ownership, output, and archive

Owner closure is valid: exit 0, errors empty, no timeout/output-limit fault, assignment before resume, job empty and closed, process/readers closed, technical completion, cleanup within budget, and scientific-result preservation. Owner elapsed time is 0.609 seconds. The recorded pre-cleanup job membership is PID 4288; successful termination was owned-job based, followed by empty/closed job evidence. The owner and Windows helper hashes match the unchanged retained sources.

FINALIZATION authenticates TERMINAL and has no storage/deadline fault. Its own-closure attestation is false as designed; the independent root CLI receipt supplies the post-return exit-0/technical-completion join. Root receipt fields and parsed owner result match the retained terminal/finalization. Worker stdout is exactly the compact `construction.output_summary(RESULT)` (327 bytes), including the correct RESULT hash; stderr is empty. Release hash and time bounds join the owner command and root CLI.

One `git cat-file --batch` verification compared 14 raw blobs byte-for-byte: all eight actual archive files at `5b4250e649ce2365d627dcd00d19521690e81ba9`, release/preflight at `c99c0ac`, and four core sources at accepted engineering commit `203ffdc07071922f887295983465b9424148fa32`. The archive is exactly eight files / 94,447 bytes; every file is below 5 MiB and the total below 8 MiB.

- RESULT: `1c54531f36ba67307caca76df2fc516fbf74155e34029f439a773f8cdd0b6063`.
- TERMINAL: `6732e23f21c3a35eb22d67d43084ecf4e4cabf9d7f1da274a039cc171c7f60c1`.
- FINALIZATION: `7389215cbf78cd91f9fafcd1e23229035c908b52de09a8bd5735c399a7b3cce1`.
- Root CLI receipt: `6bdcb08ad6dc32910e5ca4e712989f451d2d086349751d8b5872c7a70abb00ea`.

Final script output status: `PASS_REVIEW_OF_CLOSED_HELD_FAMILY_FAILURE`; fits_attempted=2, checker_calls=2, optimizer_calls_during_review=0, forbidden_imports=[]. The candidate stopped at its prescribed first held-family failure. No new scientific analysis, alternative classifier, threshold change, retry, or fresh holdout was performed.
