# Independent saved fit closeout

The construction candidate scientifically failed: 19 of 32 labels are correct,
below the frozen requirement of all 32. The independent review PASS means this
failure is authenticated and technically closed; it does not convert it into
a scientific pass. The candidate permits no retry, tuning, replacement fit, or
fresh holdout under its fixed continuation condition.

Saved-only review on 2026-09-09, 16:12:59–16:14 UTC, completed within the
authorized 120 seconds. A fresh standard-library process ran `python -B -`
from the gate namespace. It authenticated the exact release, construction
lock, core sources, training manifest, and actual capture/prepared feature
joins using unchanged production admission functions. All 32 rows were
reauthenticated: eight positive, 24 negative, native width 1024, unedited finite
float32 features with exact stored hashes.

The saved artifact was loaded using its independently expected artifact hash
and manifest/lock/feature/source bindings. The unchanged independent checker
was called exactly once on that saved artifact and those authenticated rows.
Its complete output exactly equals RESULT.check:

- Correct: 19/32; verified true; tolerance 1e-10.
- Maximum parameter difference: 2.0816681711721685e-17.
- Normal-equation residual: 6.765421556309548e-17.

No call to the fitting solver occurred during review. RESULT records exactly
one real-feature fit, CONSTRUCTION_FIT_FAIL, scientific_pass false, and zero
model calls, tokenizer calls, or checkpoint tensor reads. There was no altered
threshold, parameter search, alternative classifier, or extra analysis.

RESULT equals captured worker stdout after canonical serialization; stderr is
empty. FINALIZATION authenticates TERMINAL and reports no storage/deadline
fault. The retained root CLI result joins every terminal field plus the exact
finalization object, reports technical_complete true, and exits 0. The owner
records successful suspended assignment, empty and closed job, closed retained
process handle, joined readers, no timeout/output-limit/errors, preserved
scientific result, and cleanup within budget. The owner/helper hashes match
their exact source bytes. Worker elapsed is 0.390 seconds; owner elapsed is
0.578 seconds; cleanup is zero at the recorded clock precision. The CLI's
post-return receipt supplies the final closure observation rather than relying
on FINALIZATION to attest its own future close.

Seven saved fit/owner/CLI files total 49,099 bytes, below the 8 MiB total and
5 MiB per-file ceilings. All seven matched raw Git at
`c18c09b3ed31f5d1af14c32ed6286306e0aab97f`. One batch verified 11 Git blobs:
those seven plus the release, training manifest, core source lock, and
construction lock at prelaunch commit
`c5811a16e49a438b8fb21ad0824cf1276c89df63`. The complete review command exited
0 in 0.453 seconds internally with PASS_REVIEW_OF_CLOSED_SCIENTIFIC_FAILURE.

## Exact verified SHA-256 values

- Fit release: `4813dcc5cd732c3bc05b7fb20c0efd09c6a0acb6f76d217fa0e042477c57ce4b`.
- Training manifest: `c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15`.
- Construction lock: `0cca7c93ea7b0377220638fb37f12b38dfded37b93b5c3486e26c6e7260ee6f9`.
- Core source lock: `6ba1c6d2265ca585d17657359abfde7212ff67a0b59bf282cb7b13b49c179d2b`.
- Exact feature/label payload: `4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6`.
- Fitted artifact: `4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb`.
- RESULT: `943bc0c9409e333823fbfbda41cd34feafc79303666328870fbe46ac8a9b6993`.
- TERMINAL: `1f058f6fd5759cd373c0e2e6b4f1aaacb473ec1d954fbedab4fee7d699702b21`.
- FINALIZATION: `7a188c967a165ea86bbe84668c42ee913b889052cfd0efd38d42b818ea508a8b`.
- Root CLI receipt: `ae21cc2eb3ffa16eb5f67891e6251983fdf52789791f9550f076f6eb69dcc71c`.

The release explicitly denies model, tokenizer, and evaluation permission.
The frozen lock allows one fit, retry false, and separate clean authorship only
after an accepted artifact. This artifact failed that acceptance condition;
no fresh holdout is authorized by these records.

Review-only fits/refits, model/tokenizer/provider/tensor operations, launches,
source/raw-artifact edits, and commits were zero. None of torch, transformers,
tokenizers, safetensors, or numpy were imported. Only this report was written.
Prior unchanged preparation, capture, native, ownership, and numerical proofs
were reused within their limits.
