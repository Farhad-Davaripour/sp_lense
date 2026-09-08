# Closed real startup attempt: INCONCLUSIVE_SETUP

Exactly one authorized fresh_confirmation_loader_setup_attempt_001 is preserved. No retry, resume, source repair, predicate relaxation, or scientific continuation occurred. Root's session 16311 is closed, exit 0. Setup failure is not a scientific FAIL or PASS: all 24 baselines, 48 requests and 180 scientific cells remain UNRUN.

## Established result and lifecycle

The complete native diagnostic and saved independent audit identify **LD_ORDERED_WEIGHT_DIGEST at ADAPTER_CONSTRUCTOR**. The existing equality predicate compared these recorded hashes:

- Actual post-hook-setup ordered SHA256: 472c211f9000e4f2624954c75c37be377a158c4ff7e0eb3d70a89f6ef0c7c996.
- Frozen required SHA256: 6a671f0ae00398453e5b453d13b4ebe9de861eb3483f0e500e07bc9e456d06be.

They differ, so the unchanged admission predicate correctly stopped construction. This is the exact failing predicate for this startup attempt. The candidate adapter never returned; LOADER_TERMINAL retains loader_returned=false and UNKNOWN_IF_PARTIAL_REAL_LOAD. Reaching parameter enumeration/digest evaluation establishes that a real model object was available, but it is not a completed adapter or valid scientific runtime.

Measured: one reserved and actual backend load dispatch; 0 forwards, 0 derivatives, 0 prompt encoding. The original guard records 0 rejected calls and restored=true. Original inner latch.stop recorded LOAD_OR_SETUP_FAILURE; its hook view retained H_CAPTURE at cursor/check-attempt/completed-check 0. Native predicate diagnostics are complete; the hook fault correctly has no complete arbitrary error-details payload. No full 109-check scientific finalizer ran.

Parent elapsed 38.436999999918044 seconds. Worker and audit actual-worker/launcher exits were all 0, owned handles closed, threads/pipes closed and quiescent. One shared cleanup allowance consumed 1.6100000003352761 / 15 seconds. Successful audit execution retained INCONCLUSIVE_SETUP; it did not convert the failed constructor into success.

## Saved before/after parameter metadata

The two authenticated snapshots contain 474 entries each and 474 unique recorded object identities each; their identity sets are equal, with no recorded identity addition/loss. By matching entries on identity:

- Six names changed; 24 ordinal positions changed. Six moved three positions earlier, 18 moved one position later, and 450 retained their position.
- Shapes, dtypes, device type, requires_grad flags and version counters match for all 474 identities.
- All entries are CPU torch.float32, requires_grad=true and version=0 in both snapshots.

The six input-layernorm weight names change from
model.language_model.layers.L._original_component.input_layernorm._original_component.weight
to
model.language_model.layers.L._original_component.self_attn._ln1_module.weight.

| Layer L | Zero-based ordinal before → after | Recorded identity |
| --- | --- | --- |
| 3 | 205 → 202 | 2072197566368 |
| 7 | 258 → 255 | 2072197701040 |
| 11 | 311 → 308 | 2072197875264 |
| 15 | 364 → 361 | 2072198055248 |
| 19 | 417 → 414 | 2072198156672 |
| 23 | 470 → 467 | 2072199762384 |

These facts establish an enumeration/name permutation across hook setup. They do **not** establish unchanged parameter contents. Identity, shape, dtype and version metadata are not a byte-equality proof; no pre-setup weight-content digest or before/after per-parameter content fingerprints were recorded in this attempt. The actual post-setup digest cannot be algebraically reordered using only metadata.

## Why ordering is a supported hypothesis, not a completed diagnosis

Pinned legacy editor.py lines 185–189 hash only each parameter's contiguous raw bytes in traversal order, concatenated into SHA256; parameter names are not hashed. The legacy editor captures list(model.parameters()) at line 201 and hashes that retained list at line 215, before HookRecorder construction at line 216. Its final weight check also uses the retained list (line 601).

The production loader constructs create_recorder before RealAdapter (production_v2/loader.py lines 54–56). RealAdapter captures list(model.named_parameters()), builds its parameter list, and hashes that new traversal afterward (production_v2/real_adapter.py lines 117–125). The approved diagnostic candidate preserves these expressions and sequencing exactly. Thus the saved six alias/name changes alone do not cause the hash difference, but their demonstrated traversal permutation can change the concatenated byte sequence.

This makes an order-only mismatch a concrete supported hypothesis. It remains unproved because this attempt did not measure whether the pre-setup traversal already matched the frozen digest or whether contents changed across setup. The cause of historical real attempt 002 remains unproved; its missing predicate/stage evidence is not retroactively supplied by a later attempt.

## One minimal next proposal — still requires separate approval

Prepare one separately named, prospectively locked **setup-only fingerprint diagnostic**, with one local load, 0 forwards/derivatives/prompt encoding and unchanged weights/config/input/scientific predicates/caps. Before hook materialization retain the original ordered parameter references and record A = the unchanged digest algorithm on that list. After setup record B = the same algorithm on the same retained references, and retain C = the existing post-setup current-traversal digest plus both enumerations. Keep the current constructor equality predicate and original cleanup unchanged: the diagnostic does not admit an otherwise failing model.

If A = B = the frozen required hash, while C differs under the recorded permutation, that would support restoring the legacy fingerprint's original capture order as a narrow bookkeeping remedy while preserving current post-setup registry/parameter identity checks. No such fix is authorized or implemented here. If A already differs from the frozen hash, or A differs from B, do not apply an order-only remedy. Missing/incomplete measurements remain inconclusive. This is one bounded question, not a new prompt or parameter search.

## Preservation and authentication

Original model-free source/test commit: 3ce0fef3108893384c7b01f6763b10738bd6bee3.
Original packaged result: ec6418dd42e30befa595c393bacbe14ce1a6eb68.
Source manifest SHA256: 4e564d80b6fd55825f4eac814e1afc92f5ef9e3cf0d2bbb043a25110a556c8a0.
Root release commit: 7581e8f (all four release files authenticated against its raw Git bytes).
Caller authority-lock SHA256: 8cf0aa65ab7f1781a9122c40915df7f3d4b05b861443574bb6573b043b4bc5f9.
Admission SHA256: ca79cc51c8383f1285821e8a540350b628f32990f2e367ecbcb405c6fca318d0.
Actual usage age at exclusive admission: 35.53725457191467 seconds.

Read-only packaging checks authenticated all 337 original files against raw Git and every entry in the unchanged FINAL_INVENTORY.json (SHA256 5cb8295a039f04e2e2bed8fe0852b8d3ad5df76ff52541f778f01beebd2f2b9b). They authenticated source→release→authorization→caller lock→admission, attempt→parent→worker/audit/owned capture joins, native diagnostic acknowledgement, setup terminal, raw writer index and every indexed file. The index is honestly INCOMPLETE after hook failure, with no reconciliation issues. No original/frozen file or real evidence byte was modified.

All nine saved setup chunks were independently decompressed and authenticated without importing any model packages: setup-before 3,290,529 raw bytes, reference 3,325,811 bytes, setup-changes 84,920 bytes; complete manifests/hash joins match. This reconstructs saved hook metadata, not weight contents and not a successful 109-check run.

The 60-second-capped stdlib read-only authentication/comparison batch completed in 0.7659999998286366 seconds (command wall 1.0990725 seconds), with 0 model/backend/tokenizer imports or calls and no test-suite reruns. Actual standard Codex usage was freshly available at 4% before checks. The remaining operations are native hashing/inventory/Git packaging only.

Key evidence SHA256 values:
- Native LOADER_DIAGNOSTICS.json: 754e6dd44bea08690cfebdf602ac55da80983e68f7ae50389b0cae6329d0502b.
- SETUP_TERMINAL.json: f7b36a7fb18352848f5191ae8cca6db2398a17aed23a74c505d0676807d37131.
- Writer native index: 898442af85713b0c83f4a306fe7f604f53a09719ce055622e5e5d8672bf7e9f3.

Real evidence consists of 42 files / 828,596 bytes. The new REAL_ATTEMPT_INVENTORY.json covers all current namespace files except itself, including the original 337, four release files, all 42 real files and this report. It does not replace or edit FINAL_INVENTORY.json. Existing preparation plus fake evidence remains within 32 MiB; real evidence within 288 MiB; every file within 5 MiB. No actual root authority is reused; the consumed output remains exclusively occupied.

Stop for root review. No scientific conclusion follows from this setup diagnostic.

