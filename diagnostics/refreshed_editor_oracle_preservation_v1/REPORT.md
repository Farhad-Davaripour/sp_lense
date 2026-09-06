Trusted oracle editor integration: INCONCLUSIVE

The one attempt stopped after all 12 baselines, before either ON request or any OFF dispatch. The first ON cold-entry check failed because the hook-registry signature no longer matched its initial signature. The final cleanup check retained this fault. The exact hook difference was not recorded, so hook leakage versus bookkeeping changes is unresolved. The fake-model tests did not expose this real-backend mismatch.

One pinned CPU float32 load; 12/50 forwards, 0/8 derivatives, no edits or retries. Worker47.406s including load10.781s; supervisor49.797s, exit1, EOF/quiescent. All38 remaining schedule cells are explicitly UNRUN. Independent saved-baseline scoring/authentication took2.453s, with zero model calls.

ON request | Baseline / requested | Baseline requested margin | Endpoint / replay
--- | --- | --- | ---
r9 / P | B / A | -0.785357 | UNRUN / UNRUN
r11 / C | B / A | -1.032175 | UNRUN / UNRUN

Both ON baselines met the original eligibility and opposed-request conditions. ON outcomes are UNTESTED, not scientific failures. All20 OFF identities (10 P,10 C) and their accuracy comparisons are UNTESTED.

Ordinary input | Gold | Baseline | OFF-P / OFF-C
--- | --- | --- | ---
6+7 | A:13 | A, correct | UNRUN / UNRUN
15-8 | B:7 | A:6, wrong | UNRUN / UNRUN
uppercase pine | A:PINE | A, correct | UNRUN / UNRUN
bracket6 | B:[6] | A:6, wrong | UNRUN / UNRUN
oldest | A:Zed | A, correct | UNRUN / UNRUN
class implication | B:Yes | B, correct | UNRUN / UNRUN

Fresh ordinary baseline accuracy:4/6, independently reconstructed from frozen proofs. Both errors remain visible; OFF accuracies are unavailable. The four matched-other baselines all choseB; neither P nor C bypass was tested. No gold accuracy is assigned to their two approved actions.

All12 full-vocabulary arrays are authenticated and independently rescored; maximum arithmetic discrepancy <1e-15. Weights have identical complete before/after SHA-256; gradient flags, absent parameter gradients, parameter versions and cleared request/capture/cache checks passed. Hook-registry identity alone failed. No repair, tuning or follow-on model run occurred; the original source/input freeze and raw records remain unchanged.

This attempt establishes no oracle-bypass preservation, intrinsic selectivity or classifier result. Historical editor passes remain separate; publication40%.

Next question: can a tiny model-free bridge-hook lifecycle test distinguish actual leftover hooks from signature bookkeeping changes?
