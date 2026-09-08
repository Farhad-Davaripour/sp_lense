# Post-bridge selection correction v2

This is ONE model-free selection-only successor to immutable handoff v1 result 9eda48d356f9725dbf88908cbba67b0bab8ff8f2 (inventory b8a27d6df3fcf812e5c85a61b11a9d0bdb84bab19324ac447cb4e7fef739f053). V1's inert PASS remains limited; its actual-loader selector is BLOCKED. Compatibility 29393c717251212f4afc1ddfd9808fa0ca5bc058 and diagnostic 61a89ba171f471f8aedccea857dc6418a65b7f3e remain unchanged.

## Exact source-derived graph

Pinned component_setup.set_original_components/setup_components/setup_blocks_bridge replaces HF model.layers with the same ModuleList registered at TransformerBridge.blocks. Each of 24 entries is a BlockBridge; _original_component is the original HF decoder, whose discriminator is block_type. The optional linear_attn setup registers a GDNBridge at BOTH decoder.linear_attn and BlockBridge.linear_attn, and its _original_component is the original GDN. Three linear / one full repeated six times is fixed; actual config must agree.

Per root, require all 24 block paths and 24 decoder paths; each of 18 originals has two wrapper and two original paths. HF paths start model.layers.N; bridge paths start blocks.N. At each linear position:
- P._original_component.linear_attn._original_component
- P.linear_attn._original_component
Both must reach the SAME original GDN through the SAME GDNBridge. Therefore 36 original occurrences and 120 tracked paths per root, 72 original occurrences / 240 tracked paths across roots. The shared ModuleList is the same object. Original_model is a non-registered TransformerBridge reference, not an additional module traversal edge.

Selection walks actual _modules insertion order without duplicate removal, matching pinned Torch named_modules(remove_duplicate=False); an added finite ancestor-cycle / 65,536 visits / depth 64 / 1,024-character path guard rejects pathological metadata. The normal pinned registries are dicts; unexpected registry schemas are rejected. Exact role types, full counts, paths, ordering, original_component properties, semantic-label mappings, same-object multiplicities and aliases all must agree. Subclass extras are detected then rejected, not dropped by exact-type filtering. No parameters are accessed.

TypeBindings authenticates loaded class/module/source identities in LIVE_EXISTING_MODULE mode without importing any installed package. Inert tests explicitly use INERT_FIXTURE bindings. This mode is not a live admission shortcut or authority. Future original runtime/source identity gates remain necessary. Preserve the 18 semantic labels model.layers.N.linear_attn; they are logical labels, NOT claims that the original is still directly registered there.

## Immutable evidence, narrow delta

Retain strong references to roots/list/wrappers/decoders/originals and canonical graph bytes at initial selection. Each existing added checkpoint compares against that original graph hash, never refreshes it from a replacement. HELPER_SETUP includes the <=32 KiB complete graph; externally admitted setup and terminal/checkpoints include its SHA. Original 64 KiB native receipt limits remain. Independent selection_reader reconstructs the exact expected alias graph and type-source pins from saved data, checks global identity uniqueness and semantic selection, and joins the externally retained admission. saved_reader adds only these AND conditions. No existing acceptance predicate is removed.

support.py and fingerprint.py are byte-identical to v1. Handoff delta only imports/records/compares selection graph evidence; original installation, callable fingerprint, original callbacks, failure precedence, rollback, guard restore and publication logic are unchanged after erasing graph-only additions. Saved reader likewise retains all old checks after removing graph-only additions. No controller, loader, model, scientific method or installed source is edited.

## One prospectively fixed inert batch

Five groups, fixed subcases in selection_tests.py:
1. Valid post-wrap graph and one clean handoff/closeout; exact AST setup and duplicate-preserving traversal parity; 24/18, 36 originals and 120 tracked paths per root; all three added checkpoints and saved proof complete.
2. Old unwrapped/separate topology, missing block_type with only old layer_type, and wrong block_type reject before original setup; restoration occurs.
3. Missing direct/decoder aliases, extra root original, extra wrapper alias, wrong-type subclass extra reject.
4. One-sided rebound wrapper, duplicated original, wrong original type, rebound shared list, and coherent post-admission wrapper replacement reject. The last preserves original graph reference and terminal failure/restore.
5. Saved omitted/extra alias, rebound path identity, wrong type-source pin and terminal graph hash reject. Repair outer native hashes, controller and externally admitted FIXTURE copies to reach intended inner predicates; these are explicitly synthetic transformations, not forged real authority. Reuse saved clean case, no second clean run.

The fixture executes AST-extracted pinned setup functions, full Qwen block mapping, decoder constructor field/branch statements with inert leaves, base original/property/access methods, Module.add_module/named_modules and ModuleList iteration/append. No manually supplied flat named list. Numerical leaf classes and nonnumerical Module registry mechanics are inert stubs. This does not claim a full real module implementation or live selection success.

Source/cases/expectations/limits and namespace commit freeze BEFORE ONE batch. Actual standard-Codex usage must be available, <100%, not exhausted and <=120 seconds old. Batch 45 seconds substantive plus ONE 15-second closeout, 60 seconds absolute; 59-second final watchdog. No spawned processes. Exclusive output, no overwrite/retry. 32 MiB prep / 8 MiB test / 5 MiB file, unchanged. Any unexpected group failure preserves bytes and later UNRUN; no post-result fix or rerun. No Torch/HF/TL/backend/model/tokenizer imports, tensors, parameter access/hash, F/D/encoding, real authority, old-suite reruns or scientific claims.
