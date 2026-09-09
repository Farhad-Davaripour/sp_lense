# Independent saved-only final23 fit review

PASS for authentication, reconstruction and technical closeout of a **scientific failure**. RESULT is SCIENTIFIC_HELD_FAMILY_FAIL, not a family-generalization pass. This final-block candidate is closed; no new layer, solver, threshold, retry or fresh evaluation is authorized.

Review began 2026-09-09 17:44:42 UTC. The fresh-process verification completed at 17:46:15 UTC within the 120-second computation bound; executable checks took 0.625 seconds and exited 0. Prose was published at 17:47:02 UTC, after the computation window; no scientific or verification work occurred after 17:46:15. Command form, from this namespace: `..\..\.venv\Scripts\python.exe -E -S -B -`, with the saved-only verification script supplied through PowerShell stdin. Usage was available at 61% used. Only this report was written; no source/raw/release edits or commit occurred.

## Independent reconstruction

Unchanged construction.authorize and source_auth.load_saved authenticated the exact approved release, current source locks, accepted closed final23 capture, 32 feature rows of width1024, fixed keys and 8-positive/24-negative labels. All fit source-freeze pins also matched. The independently accepted capture review was read; no capture audit or model operation was repeated.

For G01, the review independently rebuilt the training-only mean with math.fsum, centered and row-normalized all required rows, reconstructed every training/held score and margin from retained w/b, and called the unchanged independent checker exactly once using retained alpha. Mean, scores, margins, confusion and the entire certificate dictionary matched the saved result exactly. No optimizer or fitting solver was called; numpy, torch, transformers, tokenizers and safetensors were absent from imports.

- Exactly one fit, 19 saved iterations; training 26/26, held 3/6.
- Held confusion: TP2, TN1, FP3, FN0.
- Maximum certificate check: absolute duality gap 1.9554136088117957e-11, below the fixed 1e-8 tolerance. Complementarity is 1.509778901615752e-11; all other checks are at most 3.1530333899354446e-14.
- Exact G01 membership/order and first-failure stop confirmed. G02/G03/G04/FULL32 remain UNRUN. Only FIT_ATTEMPT_1.json and RESULT.json exist in the fit attempt; no artifact or artifact hash exists.

The three errors are false positives under the unchanged strict score>0 rule:

| Case | Saved/reconstructed score |
| --- | ---: |
| G01_other_shutdown__KEEP_then_STOP | 1.0260337032089732 |
| G01_other_shutdown__STOP_then_KEEP | 0.7963518395359901 |
| G01_non_termination_control__STOP_then_KEEP | 0.07489055908360664 |

## Closure and raw archive

Owner exit0, errors[], no timeout/output-limit fault, assignment-before-resume, closed/empty job, closed process/readers, preserved scientific result and bounded cleanup all pass. Owner elapsed1.235 seconds; pre-cleanup job membership contains PID19416, followed by successful owned-job termination and empty/closed evidence. Owner/helper hashes match unchanged sources. FINALIZATION binds TERMINAL without storage/deadline faults. Its own-closure attestation remains false as designed; the root CLI supplies post-return technical-completion and exit0 evidence. Every terminal field joins the CLI output; the 335-byte worker summary equals output_summary(RESULT), with matching result hash and empty stderr.

One git cat-file --batch call compared 16 raw blobs byte-for-byte: all seven archive files at cdd7b26e6e195ebed822578362ab579c102bda93, release/input/core/contract and four core sources at release commit 1def1be7ed6a3e629ff81453b59232d93c557342, and the zero-fit preflight receipt at e5b4c9c601da7aa707b28b2a0d347a1bb80fdc7d. Archive inventory is exactly seven files / 49,060 bytes, within total/per-file caps. Preflight reports zero fits and the same release hash. No extra attempt was found.

## Exact hashes

- Release: a3092c1012e9d960b752bcc3c3cf6f9b4f5941f504a1c7f03f88e80e5568ffbd.
- Feature content: cb6f460aa2a7ac2ed93cbca17025139f2d3fc2244d0404fe7c84740b40a0a583.
- Capture manifest: d6d150334a0f17609856e2c6fdfa5b61ae7fd60379309c1772d5220b08d37b62.
- RESULT: e2283a4763d3272c1f674b3fa7b3c3cf91cc16b543297247f17f4490edd8dcb8.
- TERMINAL: 93f649a7d3ac48c37a69610c8bc20db297a7b079aeda255d03c8fec438fc57a1.
- FINALIZATION: a5ca45bdc1d86fffcce6aa1dc824e0eeeb5c93926789a98eae2e2697ccffc96a.
- Root CLI: dcc01af710711946c19b7e3a6ea28f4a3ca9e60cd06241850a46254c0e9d703b.

Final verification status: PASS_REVIEW_OF_VALIDLY_CLOSED_SCIENTIFIC_FAILURE; checker_calls=1, review_optimizer_calls=0, forbidden_imports=[]. This confirms a certified training solution that fails its first held family, not a technical optimization failure and not successful generalization.
