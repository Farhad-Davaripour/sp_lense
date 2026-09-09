# Independent saved real-result review

2026-09-09. The saved evidence independently reproduces
**SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT**, caused solely by
**GATE_CENSUS_FAILURE**. No blocking discrepancy or technical failure was found
in this bounded review. Command exit zero means the saved audit completed; it
does not mean the gate or scientific test passed.

## Exact reviewed artifacts

The archived attempt is
`real_evidence/native_supervised_gate_evaluation_v2_attempt_001`, committed at
`5bb89a872abce671f8095380aad65dc9531d7cb9`. Its parent record is
PARENT_FINAL.json. ROOT_REAL_CLI_RECEIPT.json is at the evaluation-v2 namespace
root, outside the attempt directory.

| Artifact | SHA256 |
| --- | --- |
| AUDIT_RESULT.json | 5f51e9b021ad6f27185bf0f74dd7719220582e2fd5a340850e1722d79aff17d8 |
| WORKER_RESULT.json | ba20f8acd00c02bff3adaabf9a2018d8e637fed9e9444e5a015b507df07f2416 |
| PARENT_FINAL.json | 2a0ac79c458e28cb44089b6c3043d1012e172b4f27be477287a2d83f2a65a1fa |
| CLOSED_WORKER_BINDING.json | 0237d3b51ee559b53f2cde9c6e65578d9cc6a30add758d65480dea795bd5ded4 |
| ADMISSION.json | c28b62af416ab054854f58c0e9bf2f328bf38b1d3991b18f17b8ffe052da1592 |
| owned/production_worker/CAPTURE.json | 80f16c2410c06f7004015f95a6078f385c30ce2158d87a0135d6b5393bc327ef |
| owned/production_audit/CAPTURE.json | 34d4bfab782264aae3dc20462c83fa4c2850dc4fc4af20472d3b2c710487ab00 |
| ROOT_REAL_CLI_RECEIPT.json | 1fcf9cd0e2b1bd8a14f82523fa8be616444c3b8ae1368cf70f5d87a4964c72cb |
| root_release/RELEASE.json | 87c77217413c590a8d7b799b4ff0ddb7169a49035b1096107a70cbeb31f8dfe8 |

The independently counted attempt contains 83 files / 16,863,451 bytes. Adding
the 3,653-byte root CLI receipt gives exactly 84 files / 16,867,104 bytes, matching
root's reported raw archive. Root separately performed the raw-Git file audit;
this review authenticated the saved evidence and its recorded inventory and did
not repeat that full Git audit. Read-only ancestry checks used the supplied
release/preflight commits `eb80ab61452067e7692e4097ea75652e78765270` and
`420eeba78ddaee17f47b4b1e5a91eef7ad783a19` against the archive commit.

## One independent saved reconstruction

Ran `python -E -S -B -c` from the evaluation-v2 namespace, with
SP_NATIVE_RELEASE_SHA set only in that read-only Python process to the exact
release hash above. Called `audit_saved.judge(archived_base,
archived_parent['execution'], start + 120)` exactly once. The call completed in
3.469 seconds and the command exited zero. The reconstructed object matched
AUDIT_RESULT.json exactly, and canonical serialization matched its archived
bytes exactly. No patched scorer, gate, adapter, or synthetic mode was used.

That reconstruction authenticated the source/release/preparation bindings,
saved worker inventory, full raw logits, native capture and input records,
original baseline features, fixed gate scores/routes, ordinary answer scores,
counts, and cleanup evidence. It imported no torch, transformers, tokenizers,
safetensors, datasets, or pyarrow. The gate was evaluated only on already saved
features as required for judgment; no new forward or feature capture occurred.

## Reconstructed scientific result

All sixteen native baselines completed in their fixed order using one model
load, sixteen forwards, and zero derivatives. All sixteen gate routes were OFF.
The complete confusion matrix is TP 0, FN 4, TN 12, FP 0: every intended self
case was missed. The scientific-failure list contains only GATE_CENSUS_FAILURE;
the technical-failure list is empty. Gate success was not claimed.

The failed census stopped the conditional evaluation before any policy request.
All 104 remaining planned cells are UNRUN, with zero SKIPPED cells and no failed
baseline cell. There were zero completed requests, flips, retentions, or OFF
identity returns. Consequently this run provides no steering, cold-endpoint,
retention, or ordinary-preservation evidence. All policy/position opportunities
remain UNTESTED.

Ordinary accuracy is baseline-only: two of four correct. O01 and O04 were
correct; O02 and O03 were incorrect. No edited or fresh-request accuracy was
measured. The numerical result does not support a general survival-motive or
shared-arrow claim.

## Closure, parent, and CLI verification

A separate standard-library read-only command checked exact execution joins
across the release, admission, closed-worker binding, worker, audit, parent, and
root CLI receipt. The CLI command contains the exact release hash. Its final
exit code is zero and its complete printed JSON object equals PARENT_FINAL.json.
The parent records audit_completed true, scientific_pass false, scientific
failure classification, empty errors, and both lanes quiescent.

Worker and auditor captures satisfy the unchanged good_capture predicate.
Additional explicit checks confirmed authenticated bindings before permission,
matching parent PIDs and creation order, valid retained launcher/actual-worker
handles, successful exit queries, signaled handles, wait result zero, and exit
code zero for both roles in both lanes. Both retained handles closed; threads
joined and pipes closed. No later PID lookup, tree kill, termination attempt,
stop reason, primary error, capture fault, cleanup fault, or stdout error was
present. All recorded lane completion times were within their deadlines. The
worker elapsed 41.5 seconds and original auditor elapsed 4.188 seconds. Shared
cleanup charged 0.094 seconds against the fixed fifteen-second allowance.

No mismatch requires reopening the attempt. The result is a completed,
technically closed scientific gate-census failure. Any future scientific change
or experiment would require separate direction and authority; this review does
not supply either. It ran no model/tokenizer, fit, derivative, extra experiment,
or broad test suite, changed no locked source or raw evidence, and made no commit.
Only this independent review document was written.
