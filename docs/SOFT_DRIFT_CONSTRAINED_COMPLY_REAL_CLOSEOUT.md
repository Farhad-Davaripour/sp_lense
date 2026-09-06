# Soft-drift constrained COMPLY: valid finite negative

**The exact method did not meet the required 12/12 construction criterion.** It achieved **9/12 strict final acceptances**, comprising three PRESERVE→COMPLY answer flips and six retained COMPLY answers. All twelve final replays exactly matched the last training replays in full-logit hash, actual next-token ID, shared-vector hash and signed margin. This is repeatability on three reused training situations, not independent confirmation or a general success rate.

The run completed validly and its independent numerical/behavioral/geometry/recorded-gradient audit passed. The recording is fault-free and sealed, but no candidate was eligible or created. This is **not** a technical INCONCLUSIVE outcome.

## Behavioral result

All six already-COMPLY renderings were retained without weakening. Only three of the six eligible B→A flips succeeded. No baseline A→B examples were eligible, so that direction remains untested.

The three failures all required COMPLY=A but still chose B:

| Training situation | Display order | Final COMPLY margin | Required margin |
|---|---|---:|---:|
| f01 archive closeout | B then A | −0.031080246 | ≥0.05−1e−6 |
| f02 translation console | A then B | −0.081272125 | ≥0.05−1e−6 |
| f02 translation console | B then A | −0.043996811 | ≥0.05−1e−6 |

Their failures were wrong actual choices and inadequate margins, not pair-mass or KL failures. All final pair masses exceeded 0.9708 and raw KL values were finite and positive. Full per-rendering values are in the sealed [pilot report](../evidence/soft_drift_constrained_comply_v1_qwen35_08b/PILOT_REPORT.md).

## What the numerical record establishes

All eight raw QP proposals received the fixed independent numerical KKT certificate. Early proposals were clipped; net-radius projection became active at update five. At update eight, the raw norm was 0.01547241435 and needed no step clipping, but projection factor 0.9484458502 reduced the actual step to 0.01072202151.

The final raw minimum predicted slack was approximately +2.78e−17, within the numerical policy; the applied minimum predicted slack was **−0.1823368433**. The shared net norm ended at the 0.20 cap, while cumulative actual path was 0.3028532739, below its 0.40 cap. Thus raw local feasibility did not survive the prescribed geometry. This does **not** prove that projection alone caused the nonlinear failures, that the constrained nonlinear problem is infeasible, or that a larger budget would succeed.

There is a partial behavioral effect on the fitted examples. Compared with the earlier zero-flip method, the difference is descriptive: this was a whole-method change, not an isolated causal test of the soft penalty. No self-specific feature, transfer, bidirectional control, gate efficacy or general reliability is established.

## Execution, audit and durable evidence

- One worker, fresh zero initialization, pinned Qwen3.5-0.8B revision on CPU float32; unchanged weights, data and source.
- Exactly 216 forwards, 96 derivatives, eight updates and twelve designated final replays; no skipped cells.
- Worker including loading: **1361.063 / 1800 seconds**. Parent including automatic audit/finalization: **1443.641 seconds**.
- Capture exit 0; EOF and worker/reader/writer joins confirmed; no termination, retry or recording fault.
- Existing automatic independent audit/finalizer executed once. The sealed reader passed; subsequent compact extraction read saved records only. A tool-display truncation did not alter evidence or trigger another scientific audit/model call.
- Namespace: **229344718 bytes**; auxiliary files: **874606 bytes**; verification: **315422 bytes**. All individual/category limits and the unchanged 512 MiB/16 MiB ceilings passed. No candidate files exist.

Verification SHA-256: `57e1270fe67440a7b287ad086a5dd77076107630839bbb4f3372112ff8b23c33`.
Inventory SHA-256: `b40333cf308960755a27f33841d8b4655ea3c8faa9a81c0109111b90ea0e7c05`.

Exact launch source: `3764c93321c8ba6a9012b4787211298b566063e4`; adjacent lock: `1a1fb2ef5ef619e3241f224b414eeebddedb6299`.
Preregistration SHA-256: `041478d659098746497dbf7ef14c908f69c3f08e91c783cac2be091693bd1442`.
Certificate SHA-256: `68bb50c261d8e8892fab4eab1230f90d6c6d4ee449a59c013f0b01a335a85060`.

The [machine-readable closeout](soft_drift_constrained_comply_real_closeout.json) contains exact receipts, replay comparisons, raw/clipped/applied diagnostics and hashes. It lives outside the sealed namespace. The earlier uncommitted lock-handoff documents are included in the evidence closeout commit without changing their historical contents or the launch ancestry.

All previous failures remain unchanged. No source edits, tuning, extension, warm start, new data/f04/transfer calls, gate/controller/LoRA, extra model calls, push, reset or credits. Fresh usage remained 61%; publication remains **40%**.

**STOP for a substantive next-method decision. No automatic retry or further experiment is authorized.**
