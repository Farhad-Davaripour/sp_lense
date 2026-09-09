# Independent final execution adapter review

**PASS for the bounded engineering review; no blocking defect found.** Reviewed and independently tested commit `3af568f5bac8825b2605986cf7120105aa4c6e95`. The prospective text-lock -> one-shot preparation -> root-attested process closure -> separately approved final execution chain fails closed under the contract's explicit root-authority boundary. This finding does not admit any text, authorize preparation or model execution, establish a real process closure, or claim final-study success.

## Exact source and evidence identity

All 23 local files in the final adapter's source inventory matched their frozen SHA256 values. After the independent run, `git diff --exit-code 3af568f5bac8825b2605986cf7120105aa4c6e95 -- development/native_final_execution_v1` exited 0 with no output. The report itself is outside the frozen source inventory.

| Reviewed or tested artifact | SHA256 |
| --- | --- |
| Final adapter `SOURCE_FREEZE.json` | `ff8d9b631ad9a96005554b9e2abfa9f2b7e3f2112da799cae9294af62fc21ab0` |
| Frozen binding reference `SOURCE_FREEZE.json` | `1bf2a680caa2f2b4a16191e7376f3516388a9da5764f687b1ce1a8d14659c64b` |
| Pinned preparation `SOURCE_FREEZE.json` | `bbc9a2587a766e1c1c3260d34add7c621ce32e030ae23b9e141e1b3d52b9feef` |
| Final `CONTRACT.md` | `270fb20acc20147f6f4c30cb257686fb83f3e64772d0eca00ab86af8adf22963` |
| Final `authority.py` | `8fd7adc97a04f8b0d406013aa76eaecfb5d49555d334e5a3c304f227655ee22d` |
| Final `input_reader.py` | `c41b5e41fdabf1c472a477a70fcb9186b6eb40bb7fb9de317351f75944541d88` |
| Final `loader.py` | `e9eeaba724b024f2a27e140a67e7005b6e2900e321abc913d0b5a90be5ff684c` |
| Final `support.py` | `4c374cae67fdd4818134c6627dafd3728157f5c867374a5f55e06b1a481f1ba6` |
| Final `audit_saved.py` | `ebca73f1d2de07f828888f32aed7e36ee548fbc499d7dffb65fa58ae8ca21dda` |
| Final `prep_plan.py` | `e9d5dadd49b6d0e117e1a61903582dde9b703ad2fedf157bf3b8e66b3ba21f90` |
| Final `make_candidate.py` (reviewed, not executed) | `5378c2f920b7ad9337e79333175a6b3f82b9ee34b3a3a9c3a9867ec896a8452e` |
| Final `test_binding.py` | `bd56f84083bec6566c1feb72f7a3f1acdbb6d137ee0eb1fe845307a5e0d64439` |
| Independently reproduced `TEST_RESULTS.json` | `daf73024ca548ac9288db09f0b5812919d46e6e49e321ac3dbeaba62457c574f` |
| Preparation `prepare_core.py` | `1af14fb9a43e41fe47aa54a96d5cdab0f904394dd9353be301938e309f98f026` |
| Preparation `prepare_offline.py` | `0a29f47ad335c91bd1bda6afe45eea3f9f4095b19d182c6b01591f5fef0bf98a` |

The final freeze pins the unchanged earlier 32-group binding proof at `TEST_WORKFLOW_RESULT.json` hash `f57e48653fc8d8aa22d093b2505dfd3fb2b4d7267f072ed8f721d348e042840b` and the 20-group preparation proof at `TEST_RESULTS.json` hash `d0956417d357590bcade83248eb6f0a26c3fea2d7d3a953c8562fda937fad10e`. Their source/ownership/scientific coverage is reused, not claimed as newly exercised broad testing.

## Findings

1. **Prospective lock and authority joins pass.** The adapter demands `final_execution_binding` with its exact namespace, the actual release-bound adapter freeze hash, and the fixed preparation freeze hash. Preparation validates the cohort and exact renderer output before its tokenizer factory is called, then hashes the complete canonical lock, including this binding, into its input artifact. Execution checks that canonical hash, the raw lock hash, and the exact preparation source/dependency/tokenizer-pin identity. Root's pre-tokenization commit remains the chronological evidence; the adapter does not infer chronology from a hash alone.

2. **Preparation completeness and closure joins pass.** Execution requires all 27 prepared artifacts by exact relative names and raw hashes, the result-to-input hash join, PASS and complete publication, exactly 24 cases, all 313 planned operations complete in 626 ordered STARTED/COMPLETE rows, no failed or unrun operation, no model work during preparation, and bounded elapsed time. Closure must state quiescence, successful exit, no timeout, and one-shot preparation, and must join the exact result and raw text lock. The closure is an explicitly trusted root attestation. The code cannot independently establish that the underlying retained process was actually quiescent; root must inspect and retain the authoritative process evidence before signing the release, as the contract requires. A synthetic closure test establishes only rejection/aggregation behavior.

3. **Input, boundary, category, and gold joins pass.** The fixed slot sequence binds 24 case IDs, families, layouts, expected gate categories, and each ordinary item's gold label from its own locked proof. Complete IDs and integer masks, exact declared lengths and final indices, the 320-token ceiling, answer-token IDs, and int64 byte hashes must agree across prepared inputs and case receipts. Prompt/chat hashes, fixed template identity, complete assistant header and end-token suffixes, final decoded input, no thinking, no truncation, and single-content-token receipts are checked. The adapter performs no encoding, shortening, replacement, prompt generation, gate fitting, or scientific selection. It relies on the approved unchanged preparation engine's renderer/tokenizer work and the signed raw-artifact inventory, rather than reconstructing evidence at execution time.

4. **Release, paths, and loader boundary pass.** An explicit release SHA256 is required; the exact release schema, approval, attempt, output path, limits, checkpoint lock, owned identity, source freeze, and every local Python trace source must match. Prepared relative paths reject traversal, backslashes/colon aliases, symlinks, and reparse points within the checked bundle. The input bundle has per-file and aggregate caps. Production `read()` does not enable synthetic scope. Admission exclusively creates the fixed attempt directory and binds the release execution record to both retained children. Before real provider imports, the loader authenticates the same execution, validates the owned worker bootstrap/PID, and uses the new fixed output path. The saved judge now requires full native loader evidence for `ROOT_APPROVED_NATIVE_FINAL_ONLY` as well as its previous production scope.

5. **Scientific method, ownership, and resource limits remain fixed.** Hash comparison confirms byte-for-byte reuse of `core.py`, `counts.py`, `entry.py`, `forward_trace.py`, `launch.py`, `owned_production.py`, `production_run.py`, `receiver.py`, `science.py`, `setup_budget.py`, `workflow.py`, checkpoint metadata, owned identity, and reused-source pins. Support changes only the attempt name and final newline; loader changes remove the synthetic denial and bind its bootstrap to `output()`; the saved-judge change adds the final scope to its existing loader-evidence requirement. `prep_plan.py` exactly matches the frozen preparation plan hash. Thus the 24 original baselines first, 48 fresh P/C requests, 12 self cold endpoints, 36 OFF requests, maximum 180 forwards/48 derivatives/one load, zero smoke, full vocabulary and CPU float32/eager path, learned gate and threshold, refreshed-gradient method, geometric constraints, and first-failure/SKIPPED/UNRUN rules are preserved. The worker/audit/shared-cleanup envelope remains 1800+180+15=1995 seconds; storage remains 288 MiB total and 5 MiB per file, with the complete 209,068,032-byte group-plus-terminal reservation.

## Independent test evidence

Ran exactly once, using shell `login:false`:

```powershell
.venv/Scripts/python.exe -B development/native_final_execution_v1/test_binding.py
```

Exit code 0; all **19 groups PASS**. Evidence was generated under `test_evidence/binding_1788913782285770600/`; the compact receipt was reproduced at the exact `TEST_RESULTS.json` hash above. The suite covered exact 24-input acceptance, missing/reordered cases, changed declared length below the ceiling, over-ceiling input, wrong ordinary gold, wrong gate category, boundary tampering, missing operations, wrong prospective adapter binding, wrong preparation identity, synthetic-scope production refusal, raw input tampering, coherently rehashed nonquiescent closure, missing release/loader refusal before provider import, and tampered release/source refusal.

Its one complete tiny-autograd cohort and separate saved-judge subprocess also passed. It exercised 24 baselines, 48 requests, both policy signs, 12 cold endpoints, 36 OFF identities, six flips and six retentions. The 180-cell plan used 96 completed forwards and six derivatives, with 84 legitimate skipped cells and zero unrun cells. One ordinary item deliberately remained wrong at baseline and both OFF entries; all three wrong answers remained reported wrong. Coherently rehashed entry aliasing was rejected. The retained-exit/quiescence, single shared cleanup, and full storage reservation checks passed. These are constructed test observations, not real model or study outcomes.

## Scope and remaining authority

No blocker or code correction is requested. No source was edited, real release created, source candidate regenerated, commit made, model launched, or broad inherited suite rerun. All shell calls in this review used `login:false`. I did not inspect the actual new cohort, old sealed study rows, or real study outcomes. The prescribed test used artificial schema markers, fictional release/bundle fixtures, the pinned unchanged gate parameters, and tiny autograd; the provider/network/checkpoint guards remained active and reported zero real tokenizer and Qwen calls. No real owned model process was observed. Subsequent text admission, preparation authority, observed preparation closure, and a separately approved real final-run release remain root decisions and prerequisites.
