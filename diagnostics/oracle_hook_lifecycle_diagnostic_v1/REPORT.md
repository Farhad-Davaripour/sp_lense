Model-free hook lifecycle diagnosis: mechanism reproduced; candidate guard passes

The installed bridge can change v1's registry signature after an ordinary capture context closes while leaving no diagnostic capture callback. The tiny real-library fixture retained two intentional architecture forward-pre callbacks and added an alias to an existing LayerNorm module. It did not leave the diagnostic capture callback. This is a plausible mechanism for v1's unrecorded mismatch, not proof of that historical cause; v1 remains immutable INCONCLUSIVE.

Source basis (exact installed files and SHA-256s are frozen): BlockBridge.forward invokes _maybe_wire_pre_ln_capture (block.py:114,196). That method installs ln1/ln2 pre-normalization callbacks and assigns attn._ln1_module = ln1.original_component. BridgeCore.hooks removes nonpermanent callbacks from each selected HookPoint on exit (bridge_core.py:506); it does not remove these separate architecture pre-hooks. HookPoint stores user callback, permanent flag and level in LensHandle (hook_points.py:143,181,330).

Regression | Result
--- | ---
Cold real-library capture | Signature changed;0 diagnostic captures left;2 architecture pre-hooks plus1 existing-module alias
Verified setup before strict baseline | Same exact changes;0 forwards during setup
Three repeated captures with baseline permanent callback | Strict structured identity preserved
Nested edit/gradient-context/capture | Full-context cleanup clean; inner exit removes outer nonpermanent callbacks, explicitly detected
13 deliberate faults | All rejected: capture/edit/permanent/pre/backward/global leaks; baseline removal/replacement/order; new module; alias replacement

All16 fixed cases passed their expected outcomes. Seven arithmetic forwards on a 16-parameter, width4 fixture; zero derivatives, Qwen loads or language-model forwards. Experiment7.187s; supervisor8.266s, exit0. No retry. The fixture uses actual TransformerBridge/BridgeCore registration, BlockBridge, NormalizationBridge and HookPoint classes; its forward is tiny arithmetic, not an HF language-model forward.

Candidate invariant: before the first baseline only, call the exact source-bound BlockBridge lazy-setup implementation and validate its exact callback code objects, globals, closure-bound targets, handle order and existing-module alias. Reject every other setup change. Thereafter require identical module paths/objects/child order, hook-registry bindings, per-module and global callback objects/keys/order/options, callback code/closure identities and HookPoint user/permanent/level metadata. Retained references prevent object-ID reuse from hiding replacements. No active_hooks shortcut or blanket allowance for new modules exists. Structured deltas are retained.

guard_candidate.py and candidate.patch are proposal-only. SUCCESSOR_PLAN.json retains the identical12inputs,50F/8D,600+15+60s,80MiB, original editor math/gates, weight checks, request cleanup and exact OFF identities. Hook identity does not replace those separate cleanup obligations. A separately frozen v2 needs root review/release; no real model is authorized here. Publication40%.

Next decision: review this exact guard and separately authorize a clean v2 of the unchanged integration matrix, if accepted.
