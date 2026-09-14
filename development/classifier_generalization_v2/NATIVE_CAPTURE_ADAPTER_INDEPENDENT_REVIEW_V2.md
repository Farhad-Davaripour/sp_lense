# NATIVE_CAPTURE_ADAPTER_INDEPENDENT_REVIEW_V2

Job ID: `native_capture_adapter_review_v2_20260914_0650`. Read-only independent re-review of
`development/classifier_generalization_v2`: `native_capture_adapter.py`,
`test_native_capture_adapter.py`, `NATIVE_CAPTURE_ADAPTER_SUPERVISOR_REPAIR_V2.md`,
`NATIVE_CAPTURE_ADAPTER_INDEPENDENT_REVIEW_V1.md`, `native_capture_contract.py`, and the
fabricated helpers `test_native_capture_contract.make_case` /
`test_tokenizer_input_adapter.FakeTokenizer`. No real torch/transformers/model/tokenizer/
snapshot/cache bytes were imported, loaded or run. No source edits, network, installs, Git,
capture, fit or HOLDOUT access.

## Scoped verdict: PASS (tested injected-provider surface)

V1's F1–F6 are addressed as declared. No blocking defect and no new framework requirement
found. Residual items below are low-severity defense-in-depth/documentation gaps, not
regressions.

## Native hashes (SHA256, this review)

- `native_capture_adapter.py` `9CE842374934FE7CF0DFE11BF4A03581FC2EBA4DCAC69A856BE82156D78C5831`
- `test_native_capture_adapter.py` `BC97D396C8C0F5330EEA331AA858DCA67EE833FDF489D504F44C095B59201674`
- `native_capture_contract.py` `8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391`

First two match the repair doc exactly; contract hash matches V1.

## Tests and probes

Study `.runtime` CPython 3.12.14 (`torch` absent), `PYTHONPATH` = module dir, `-W error`:

- `test_native_capture_adapter.py` **29/29 OK** (0 skip/fail/error).
- Related: `test_native_capture_contract.py` 12, `test_tokenizer_input_adapter.py` 9,
  `test_capture_executor.py` 22 — all OK.
- 19 independent probes (temp dir, deleted; repo untouched) over the fabricated providers:
  15 behavior + 4 edge.

**F1 cross-capture baseline.** Names/ids/versions are frozen into an immutable tuple at
build and compared against the full `named_parameters(remove_duplicate=False)` sequence
before every forward, independent of the public `parameters`/`coverage`/
`loading_report`/`provenance` attributes. Probes: mutate a parameter, then clear/overwrite
each public attribute, then capture — `PARAMETER_CONTINUITY` still raises with zero forwards;
name changes and in-forward parameter replacement are likewise rejected before/without a
returned result. PASS.

**F3 vision-eager.** Build rejects `vision_config._attn_implementation != "eager"` (`EAGER`);
outer/text/vision all asserted. PASS.

**F4 hooks.** `_assert_no_forward_hooks` scans every model module's `_forward_hooks` and
`_forward_pre_hooks` plus `torch_api.nn.modules.module._global_forward_hooks`/
`_global_forward_pre_hooks`, at build, before every forward, and after own-hook removal.
Foreign hooks (model/parent/other-layer/target-pre/global) are rejected `PREEXISTING_HOOKS`
and preserved, never erased; a foreign hook appearing during our forward is detected
(`HOOK_CLEANUP`) and left intact. PASS.

**F5 provider ops.** `tensor`/`ones`/`equal`/`inference_mode` must be callable;
`TORCH_API` raises before either tokenizer or model construction (probe confirms both
`from_pretrained` call lists stay empty). PASS.

**Boundary.** The adapter enforces the contract's minimum tail — `readout_index <
final_input_index`, `final_input_index == len-1`, `input_ids[readout_index] == 198`,
`input_ids[readout_index+1] ∈ LABEL_TOKEN_IDS`, length ≤ 320. Probe accepted `[11,198,33]`
and `[198,32]`, rejected the 1-token view before forward. No stale seven-token rule exists
in adapter or contract (grep). PASS.

**Cleanup/finalizers/API.** Hook registry restored on HOOK_SHAPE, forward-error and
POSITIONS_CHANGED paths; no activation reference survives (weakrefs dead even pre-`gc`).
Finalizer observer is restored in `finally` for own, inherited and pre-finalizer load
failures. `encode`/`decode` forward correctly; `capture_view` returns the
executor-compatible numeric dict; `capture_executor.execute_cases` runs 2 forwards OK;
`provenance.native_model_provenance_verified is False`; module imports remain
`{math, os, native_capture_contract}` with no `open`. PASS.

## Residual findings (low, non-blocking)

- **L1 — global-hook check can fail open.** If `torch_api.nn.modules.module` resolves to
  something without the `_global_*_hooks` attributes (e.g. a class object), the check
  silently skips instead of flagging an uninspectable API. Prefer requiring the registries
  be dicts when the path is exposed. Tested path is correct.
- **L2 — `TORCH_API` scope.** Only callability (and `float32`/`long` presence) is checked;
  not dtype-acceptance or CPU/float32 enforcement. Fail-loud later, not fail-closed.
- **L3 — F2 documentation.** The mutable-informational-attribute limitation and the
  raw-storage/version-counter limits are honestly documented in the repair doc, but the
  adapter's own docstring still does not name `coverage`/`parameters`/`loading_report`/
  `provenance` as mutable and non-authenticating. Consumer-facing clarity only.
- **L4 — superseded artifacts.** The frozen V1 handoff still says "read-only" attributes
  and 23 tests; the repair doc states these are superseded. No conflict in the live code.

Version counters are explicitly *not* cryptographic authentication and cannot detect
deliberate raw-storage writes bypassing PyTorch's counter. This is a declared limitation,
not a defect. Injected providers remain unauthenticated; the outer run owner must still
verify snapshot bytes, pinned provider/source/runtime identity, a finite execution lock and
ownership before invoking the factory.

## Limits

Fabricated providers only: no real torch/transformers import, model/tokenizer load,
forward, capture or fit; no snapshot/config/checkpoint/cache bytes read; no performance or
provenance claim; no source edits, network, installs, Git, subagents or HOLDOUT.

**Final: PASS on tested scope; F1–F6 resolved as declared; remaining L1–L4 are low non-blocking; providers stay unauthenticated.**
