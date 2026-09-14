# SPAN_CAPTURE_ADAPTER_REVIEW_V1

Job ID: `span_adapter_review_20260914_1054`. Read-only independent review of
`span_capture_adapter_v1.py` + `test_span_capture_adapter_v1.py` against
`native_capture_adapter.py`/`native_capture_contract.py`, `test_native_capture_adapter.py`
fakes and `PROMPT_SPAN_LAYER_PLAN_V2.md`. No edits, fitting, capture, network, installs,
Git, subagents or HOLDOUT; no real torch/transformers/model/tokenizer/data.

Verdict: **PASS_SCOPED** (tested injected-provider surface only; no native claim).

SHA256 pins (this review):
- span_capture_adapter_v1.py `59EDDFB20ADE6FB97C10DB709DAD9012B6C4B77E72765B75CAB186C443E20A99`
- test_span_capture_adapter_v1.py `80766AE07B26C05D1495E56A8A3312BD8629950EF791AA7123B635CEFDCA75CE`
- native_capture_adapter.py `9CE842374934FE7CF0DFE11BF4A03581FC2EBA4DCAC69A856BE82156D78C5831`
- test_native_capture_adapter.py `BC97D396C8C0F5330EEA331AA858DCA67EE833FDF489D504F44C095B59201674`
- native_capture_contract.py `8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391`

Tests: 21/21 OK, `-W error`, study `.runtime` (torch absent). Probes confirmed: one forward per
call, only `ids[:readout_index+1]` forwarded after full-view validation; blocks 6/10/18 hook
shape/dtype and exact-one-callback; finite float32 row output; no parameter/grad/eval/device/
eager mutation; own hooks removed on pre-hook, mid-registration, forward-error and hook-reject
paths with foreign hooks preserved; provider injection loads once.

Residual (low, non-blocking): successful-call count is not cumulative — the outer executor owns
the 640-forward cap; eval/float32/eager are trusted from the reviewed native build rather than
re-verified per call; a mis-wired custom model class that never executes a configured block
fails safe as `HOOK_CALLS` with no window returned.
