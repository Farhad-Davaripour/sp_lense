# Independent prospective storage revision review

**PASS for the authorized storage-only engineering revision.** The earlier combined-write blocker is resolved for the exact source candidates and tests below. No concrete blocker remains in the reviewed storage paths or affected source bindings. This is not a text lock, preparation release, permission to tokenize, model authorization, or proof of an actual preparation process closure.

I read `ROOT_STORAGE_FIX_AUTHORIZATION.md` completely and stayed within its prospective scope. Usage was checked before jobs: 36% used. The prior blocked `INDEPENDENT_REVIEW.md` remains unchanged at SHA256 `e7b40e4aa4e0b67f26cd24e8c6af6bb4e1c061a147e2ba7c4ff0440bf20f831b`.

## Exact reviewed and tested hashes

| Source candidate | SOURCE_FREEZE.json SHA256 |
| --- | --- |
| `native_final_preparation_v1` | `247954f283b1a47b98e9067c2b01d41d836ef29576793066b42dd4beb912d3f6` |
| `native_final_execution_v1` | `26100ab792bcf2379cb7a4960bd5aa1a54f5e69477043bf132e4f09ee0540af3` |
| `native_final_preparation_owner_v1` | `1264e672cadc91ebe7c4757135178d2e6cb3072bf1b750d23ff7df63b87e246a` |

These three hashes matched before and after the independent tests. Adapter and owner source verification checked the frozen local and external bytes, including the revised preparation code. Relevant individual source hashes are:

| Source | SHA256 |
| --- | --- |
| Preparation `storage.py` | `687f84e275b58cf81c01bfe76733519ce6f81736fadeaab864c6af36f2abce31` |
| Preparation `prepare_core.py` | `ff50c632ff92200519119ec31f5456e262b5df6d449d7a44c56051252c0f0dfc` |
| Preparation `prepare_offline.py` | `790750293d139c854e8df6a79141005aa1696160777c96d283b1491b90cc15fa` |
| Preparation `plan.py` / adapter `prep_plan.py` | `50937ed964fed89a7bee9adf29b04c56d5ce376ff8d9b68b472569a000e25a32` |
| Adapter `input_reader.py` | `293154caba0fdfd211f2f30402414e343dd6bd9b11658c201588e6f845ec5c66` |
| Owner `owner.py`, including readiness flag true | `d274422cc4c732259001466984081eaa41e89e3c155e0439b1e1bdcf02768fc4` |
| Preparation `test_storage.py` | `21bc6b096c99067d0183c001b452c25eb1ad94013f8559589784a5d540400d45` |
| Adapter `test_storage_binding.py` | `b8ca454447927ecda21618f81410cba0f2cf3df16c22e2182490ef507e8fa61d` |
| Owner `test_owner_storage.py` | `691ea24b508b649b8b9a29cc5a3e3afed223f0251164f79580a632686d45f77d` |

## Deterministic bound and write-path review

The combined cap is unchanged at 16,777,216 bytes. Preparation has a disjoint allowance of 16,744,448 bytes and owner evidence has 32,768 bytes. Their maximum sum is exactly the required combined cap; unused allowance in one partition cannot be borrowed by the other.

Every preparation output write now passes through the same stdlib `Publisher`: ADMISSION, appended STARTED/COMPLETE/FAILED operation rows, completed case receipts, final inputs, normal core RESULT, and entry-failure RESULT. Before opening a file, it checks the prospective per-file size and total preparation bytes. Ordinary writes preserve 65,536 bytes of terminal capacity inside the preparation allowance. Critical writes cannot cross the preparation allowance. RESULT must be critical, non-appended, and at most 8192 bytes. The per-file ceiling remains 5 MiB. The entry no longer has an unchecked direct write path.

Exception types are limited to 64 ASCII identifier characters; exception codes are limited to 128 characters from an explicit ASCII set. Oversized, invalid, or broken string conversions produce fixed failure fields. A failure after an operation journal exists reports attempted/unrun counts as unknown with a pointer to that retained evidence; it does not falsely report zero work. Core counters and failure/UNRUN semantics remain intact.

Every owner evidence write uses an exclusive prechecked writer: ADMISSION at most 4096 bytes, stdout and stderr at most 8192 each, and CLOSURE at most 12,288. Those maxima sum to 32,768 bytes. Each write also checks the current owner total before opening the destination. Runtime monitoring now checks the two disjoint allocations independently and no longer double-counts actual owner bytes plus its full reservation. Closure retains an additional combined-size check, but the guarantee follows from the two pre-write allocations rather than polling or the postcheck.

The earlier ownership primitives, hidden suspended launch, job containment, retained identities, capture/EOF/handle closure checks, and one-shot behavior were not changed by this storage repair. The worker kept the readiness guard false through the source pass and its local focused checks, then included the true flag in the final frozen candidate before my independent test set. The flag indicates storage readiness only: exact source verification, an explicit root release hash, release approval, and the owner binding in the pre-tokenization text lock are still required. The final fake-child test confirmed that an absent real release still prevents a production attempt.

## Scope and source-binding checks

Diff review against the preserved pre-repair commit `3da6385` found preparation changes limited to the publisher/error-serialization paths and storage metadata. The 313-operation plan, prompts/IDs/labels, model, 320-token ceiling, tokenizer boundary checks, 180-second preparation envelope, and `STUDY` metadata are unchanged. The final scientific worker, gate, thresholds, receiver, scores, geometric constraints, and final study budgets retain their previous hashes. No old scientific attempt was rewritten.

The adapter's preparation source constant now equals the revised preparation freeze. Its preparation plan is byte-identical to the revised plan. It independently enforces the 8192-byte RESULT bound and the 16,744,448-byte prepared-artifact allocation, in addition to the existing raw-hash, result/input, source, text-lock, category, gold, and closure joins. The owner freeze pins the same revised preparation source, storage implementation, and plan.

The binding graph remains acyclic: the preparation candidate is frozen first; the adapter binds that preparation freeze; the owner pins the preparation source and has its own freeze. The future TEXT_LOCK includes the adapter binding and owner binding, and preparation hashes that entire lock. No source candidate needs the future lock's own bytes or hash. The unchanged dependency and tokenizer-pin identities remain `fc729121734b4e0f0118319a23e5e193848c81e2bab8c03ebfadbbfe2b7d9222` and `c5ae7cbd5356b1df1f6590045eed010f45acc70d2ccb9c39337504dd2d1f4135`; affected source identities have been rebound rather than left stale. Root must use the three exact revised freeze hashes above in its future combined lock and releases.

## Independent focused tests

Ran each command exactly once, sequentially, with shell `login:false`. All exited 0.

```powershell
.venv/Scripts/python.exe -B development/native_final_preparation_v1/test_storage.py
.venv/Scripts/python.exe -B development/native_final_execution_v1/test_storage_binding.py
.venv/Scripts/python.exe -B development/native_final_preparation_owner_v1/test_owner_storage.py
```

All **18 focused groups passed**: six preparation/storage groups, three adapter groups, and nine fake-child owner groups. The preparation checks accepted exact ordinary and critical boundaries, rejected the next byte before writing, refused an 8193-byte RESULT before creation, bounded a one-million-character error message and a 100,000-character exception type, handled a broken `__str__`, published compact entry/core failures inside reserve, preserved unknown counts after a journal, and completed the unchanged fake 24-case/313-operation interface. They also filled the owner allocation exactly and demonstrated preparation plus owner equaling exactly 16 MiB, followed by refusal without creation/overwrite.

The adapter accepted the revised fake 24-input binding and rejected coherently rehashed terminal and total-partition overflows. The owner suite exercised natural success, nonzero exit, timeout, stdout/stderr overflow, retry refusal, suspended pre-assignment cleanup, missing release arguments, and missing real-release refusal. These were fake children only; no real preparation was launched. The small AST wrapper changes only the old test's report filename and the now-accurate missing-release group label, preserving the historical report.

| New independent report | SHA256 |
| --- | --- |
| Preparation `TEST_STORAGE_RESULTS.json` | `46cd87204720607fffaf5f5ba4d7097d5b5c20abda392ecc665af91597261206` |
| Adapter `TEST_STORAGE_BINDING_RESULTS.json` | `57cae0b06c969225c10306e7a7eda67bd6cefb357b30b7ca7f2333608740754d` |
| Owner `TEST_STORAGE_OWNER_RESULTS.json` | `37fb90fab8084b4a105a7d96ea2f02d2a48f7ab80cc2f5e6ab0de7d106f06f85` |

The new preparation evidence is under `test_evidence/storage_1788916513964670300/`; new owner evidence is under `test_evidence/owned_1788916517411975500/` in their respective namespaces. Earlier `TEST_RESULTS.json` files in all three namespaces, the blocked review, and `native_final_binding_v1` are unchanged versus `3da6385` (`git diff --exit-code` returned 0).

I made no source edits, commit, approval, release, actual text/cohort access, real tokenizer/model call, or broad inherited-suite rerun. Historical results and the blocked review remain preserved rather than relabeled. Root alone may approve the future text/source lock and a separate one-shot preparation launch; authoritative closure of that actual process remains a future observation.
