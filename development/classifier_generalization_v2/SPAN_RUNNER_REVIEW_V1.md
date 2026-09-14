# SPAN_RUNNER_REVIEW_V1

Job ID `span_runner_review_20260914_1120`. Read-only independent review of
`span_development_runner_v1.py` (`54E29C37F9CBBFFBAB3F81C10D43F472D7D0CE9AAB55CE4C7D0F105E36AB4ABF`)
and `test_span_development_runner_v1.py` (`8D31B13BD1CE3E0494DAA0A8FB9D78F8DF277A4B607B5125AFAF5F825EF4D110`)
against `PROMPT_SPAN_LAYER_PLAN_V2.md`, `span_capture_adapter_v1.py`,
`native_development_runner_v2.py`, `tokenizer_input_adapter.py`. Both requested pins
verified. No edits; no real dataset/model/provider/tokenizer/snapshot/feature/result/
HOLDOUT read; no native preflight, fit, capture, network, install, Git, config,
coordination or subagent. 12/12 synthetic tests PASS (`-W error`, real toy watched child).

Verdict: **PASS_SCOPED** (model-free structural evidence only; trusted
provider/run-owner boundary accepted, no native readiness claim).

Entry path verified: `preflight` checks lock caps, exact source set and runtime/provider
pins, then runs the reviewed old-lock preflight on the two 320-forward metadata locks
only (never their captures) and rejects old run_id reuse. `_combine` enforces 320 unique
cases, 240 TRAIN/80 VAL, 80 per class, no HOLDOUT overlap. `supervise` -> watched child
-> snapshot bytes verified before the single provider build -> 640 prefix-only forwards
(option tokens validated but excluded) -> streamed little-endian float32 `windows.f32`
plus JSON index (case/order/block/offset/length/token_positions/prefix and input hashes)
-> parent re-hashes all three artifacts before `supervisor_success.json`. Failures
preserve the partial binary and remove the owner marker on timeout/error paths.

Non-blocking observations, not reachable in the locked entry path:
1. `capture` never reads `caps["forwards"]` and does not assert `len(cases)==320`; 640
   holds transitively through `preflight`/`_combine`. Optional hardening:
   `need(len(cases)==320 and forwards<=caps["forwards"], "FORWARDS")`.
2. Tokenizer identity is passed expected==observed, so that check is tautological;
   tokenizer bytes are still authenticated upstream by `snapshot_verifier`.
3. Handoff "128 MiB rough raw" is 120 MiB (cosmetic).

Native SHA256s: `native_development_runner_v2.py`
`1BFDE09C4D4EE124DF4F5DDFA231CD8EE2A822B2FD9EBEB9383AC5124F1DF92A`;
`span_capture_adapter_v1.py`
`59EDDFB20ADE6FB97C10DB709DAD9012B6C4B77E72765B75CAB186C443E20A99`;
`native_capture_adapter.py`
`9CE842374934FE7CF0DFE11BF4A03581FC2EBA4DCAC69A856BE82156D78C5831`;
`native_capture_contract.py`
`8841BC0B76157936A77B973089BCE20856442DB2BCDB6BC44F756B8F616A3391`;
`snapshot_verifier.py`
`710C16F39B2FB7AA818D5B6EE5D33632555E690D745997FA0FB0F51AD134875A`;
`tokenizer_input_adapter.py`
`3883326066F2D3B322A79547BC9054588750D9D020A47543930B216183F4D164`.
