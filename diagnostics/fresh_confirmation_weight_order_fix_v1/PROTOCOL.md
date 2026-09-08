# Prospective retained legacy weight-order bookkeeping correction

Question: can the exact pre-hook legacy parameter ordering make the unchanged fingerprint predicate correct while still rejecting changed weight bytes?

Historical order diagnostic is immutable at d2296bed4fd300854499758c38d3de24c9e42543. Its A=B=frozen/Cdifferent supports ordering only for that capture; its LD_ORDERED_WEIGHT_DIGEST/INCONCLUSIVE_SETUP remains final. Attempt002's missing original cause remains unproved.

## Minimal executable delta

The loader captures the complete strong immutable tuple(model.parameters()) immediately before create_recorder, at the legacy pre-hook capture site. It freshly computes the unchanged parameter_digest once on that tuple and requires the exact frozen expected6a671f0ae00398453e5b453d13b4ebe9de861eb3483f0e500e07bc9e456d06be before setup. Initial mismatch is only INITIAL_FINGERPRINT_MISMATCH, not automatic wrong-checkpoint proof.

One explicit legacy_weight_parameters constructor argument passes that SAME tuple. The adapter leaves self.named_parameters, self.parameters, all flags/versions/gradients and current registry/cache/hook checks in their original current-traversal order. A separate legacy_weights object requires complete same-object multiplicities and per-object shape/dtype/device/numel/element-size/byte-boundaries, flags and versions across setup. No sorting, alias list, observedC replacement, new expected value, model mutation or threshold change occurs.

Only the two original content-hash call arguments change to self.legacy_weights.parameters: constructor and parameter_state(digest=True). Both freshly execute the unchanged raw-byte algorithm; closeout never uses a cached expected result. An additional boolean metadata/bijection identity is required by the unchanged all-boolean cold-receiver/cleanup predicates. Subsequent current name/order/object identity checks remain original. No subsequent digest uses current traversal.

All14 original predicate expressions/messages and original inner latch.stop/guard.restore cleanup are AST-identical. Science/gate/editor/scorer/input/model/weight/config pins and production ceilings are unchanged. Twenty-one checked infrastructure/source files including the base saved setup reader are raw-byte identical to the predecessor. The separate saved proof reader only preserves or downgrades the original setup judge; SETUP_DIAGNOSTIC_COMPLETE is not scientific PASS.

The finite diagnostic retains source/execution phases, exact before/retained/current metadata, one extra initial hash and already-computed constructor hash within2MiB; no parameter contents or exception text. All future hashes still use fresh loaded objects, never historical A/B/C. The native publisher, first-cause diagnostics, recorder retention, exclusive authority/output occupation, owned controller and reserved closeout are inherited, not rerun or redesigned.

## One frozen model-free regression

TEST_PROTOCOL.json fixes11byte-standin cases in4groups before any batch: unchanged order and alias permutation accepted; version-silent before/closeout content changes rejected; wrong initial fingerprint rejected; missing references, omitted/replaced/duplicated objects and byte-boundary drift rejected; later current-registry drift rejected even when retained content is correct. Boundary fixture preserves concatenated bytes while changing individual boundaries. Exactly three full hash calls on normal and late-fault fixtures (one new initial, inherited constructor, inherited closeout); two on constructor content mismatch; one on early reference/initial failures. No actual model/content is loaded. Two static cached-hash substitutions must be rejected by the exact source delta check.

The actual candidate constructor and parameter_state are AST-extracted without module imports into finite array stand-ins; native exclusive publication/readback plus independent saved joins are exercised. This is not the full owned real workflow, actual Qwen setup, tensor autograd, scientific engine, or old eight-case suite. Reused proofs are pinned by exact hashes.

ONE45s substantive deadline includes all saved readers; ONE15s closeout inside60s, terminal timer59.5s.32MiBpreparation/8MiBcombinedfixtures/5MiBfile/2MiBnative receipt. No cap reset, retry, post-outcome source repair or denominator shrink. A failed fixed case remains FAIL and the exact later suffix UNRUN. Actual standard-Codex usage is tool-read and must remain available/nonexhausted/<100 and fresh<=120s at batch start. No purchases/resets. Source/test input lock is committed before execution.

## Future boundary remains disabled

Distinct fresh_confirmation_weight_order_fix_attempt_001 only. No root_release directory or actual real_evidence supplied. Future independent root release/actual fresh usage admission after validation is mandatory; no inherited consumed authority/output. Separate prospective setup-only1load0F0D0promptencoding,180worker60auditONE15sharedcleanup255total,288MiBevidence/5MiBfile and native2MiB remain inside original1800/180/15/1995 and180F48D1load/160token ceilings. Successful setup cannot continue to any question; all24/48/180scientific slots stayUNRUN. Remaining actual setup success/capacity/timing is not inferred from byte fixtures. Stop for root review.
