# Paired common-drift COMPLY: prospective method proposal

**DESIGN ONLY; STOP for review before production implementation or model work.** Proposed identity: `paired_common_drift_comply_three_family_v1`. This is **adaptive development**, motivated by an exposed failure, not an independent confirmation. One objective and one update rule are proposed; no alternatives, coefficient sweep or candidate selection.

## Fixed scope and pairing

Reuse exactly the [existing twelve f01/f02/f03 v1 self-shutdown renderings and budgets](../configs/shared_comply_crossed_three_family_v1.json). Keep pinned Qwen3.5-0.8B, CPU float32, template, model weights, `blocks.10.hook_out`, and final encoded prompt token unchanged. Within each family, pair the same display order across COMPLY=A and COMPLY=B mappings. In the existing one-based row order, pairs `(A-mapping row, B-mapping row)` are `(3,1), (4,2), (7,5), (8,6), (11,9), (12,10)`.

Start a new shared float64 vector `w=0`; never initialize from either old C. For each prompt independently, capture its original baseline hidden state `h0_i` and fixed norm `n_i = sqrt(fsum(h0_i[j]^2))`. At every forward, use the original prompt and physical edit `h_i = float32(h0_i + float32(n_i*w))`. Never average norms, use a donor norm, or accumulate edited hidden states. Model casting remains unchanged; optimization uses the existing local autograd surrogate through these casts, not an exact derivative of discrete floating-point rounding.

## One objective, with explicit signs

Let `L_i(w)=z_A-z_B`, in natural-log units (nats), and `S_i=z_PRESERVE-z_COMPLY`. For each pair, subscripts `A` and `B` identify which letter means COMPLY, not display position:

```
x = L_A(w) - L_A(0)             y = L_B(w) - L_B(0)
a = (x-y)/2                    c = (x+y)/2
m_A = L_A(w) = m_A(0)+a+c       m_B = -L_B(w) = m_B(0)+a-c
r_A = max(0, tau-m_A)           r_B = max(0, tau-m_B)
ell_pair = (r_A^2+r_B^2)/2 + lambda*c^2
loss = sum(ell_pair over six pairs)/6
```

`a` is mean baseline-relative semantic COMPLY movement; individual semantic changes are `x=a+c` and `-y=a-c`. Positive `a` alone does not ensure improvement in both members. `c` is common letter drift; positive favors A, negative favors B. Use **mean of squared pair drifts**, never square of mean drift.

Fix `tau=0.10` nats, inherited from the existing construction aim, and **lambda=1**. Convention: a drift of `d` nats costs `d^2`, the same as two equal margin deficits of `d` nats in the pair. Loss units are nats squared. This equal-unit convention is chosen here, before any new model evaluation; it is not calibrated to f04, old-vector responses, or a sweep. There is no hard zero-drift constraint, positive-change constraint, or loss-based acceptance gate.

## One update rule

At current shared `w`, obtain each prompt's existing gradient `gS_i = dS_i/dh_i` once. Convert to shared-coordinate letter Jacobians using each prompt's own fixed norm:

```
J_A = -n_A*gS_A       J_B = +n_B*gS_B       J_c = (J_A+J_B)/2
g = mean_pairs(-r_A*J_A + r_B*J_B + 2*c*J_c)
B = mean_pairs(||J_A||^2 + ||J_B||^2 + 2*||J_c||^2)
B0 = (tau/step_cap)^2 = (0.10/0.05)^2 = 4
d = -g / max(B,B0)
s = d * min(1,0.05/||d||)       # s=0 when d=0
u = w+s
w_next = u * min(1,0.20/||u||)  # w_next=0 when u=0
r = w_next-w
path_next = path+||r||
```

Use float64 arithmetic and `fsum` reductions in the fixed pair/coordinate order. `B` bounds the spectral curvature of the **frozen-Jacobian squared-hinge surrogate**, since each hinge-active indicator is at most one. It does not bound the nonlinear model Hessian or promise actual loss descent. `B0` fixes a finite step scale from the declared aim/step budget, not observed outcomes. No line search, momentum, optimizer rescue, extra gradients or KKT/infeasibility claim.

Keep step cap 0.05, net cap 0.20, path cap 0.40 and at most eight attempted updates. Count actual post-projection increments; check the original numerical/casting tolerances and path guard. Exact zero increment ends the method without rescue. Cache gradients only at their observed shared `w`; historical gradients or a previous iterate's gradients are not substitutes. Current-gradient forwards provide logits too; assembling `g` and `B` adds no backward pass.

## Meaning and limitations

Common drift is a **response-based nuisance proxy, not a purified causal letter feature**. Different prompts and unequal sensitivities can give a purely common preference nonzero `a`, or useful semantic movement nonzero `c`. Penalizing `c` may suppress useful effects. Opposite family/display drifts cannot cancel in `mean(c^2)`, but opposite within-pair responses can make `c=0` without proving a common semantic mechanism. Keep individual margins/effects and all six `(a,c)` values visible.

This also replaces the old minimum-norm inequality update with curvature-scaled gradient descent. Any later difference tests **the whole new construction method**, not the isolated causal contribution of the penalty. No extra comparison arm is proposed.

## Future accounting, outcome and anti-leakage controls

The proposed maximum schedule is `12 baseline + 8*(12 current-gradient forwards + 12 scored edits) + 12 final replays = 216 forwards`, with `8*12=96 derivatives`. All twelve gradients in a round use the same `w`. No warmup, calibration or additional objective forward. Stop at the first scored 12/12 behavioral acceptance, otherwise at the update ceiling or finite stall. A completed finite endpoint, including a failed one, gets only its designated twelve final replays; no best-earlier-iterate selection.

**Success branch:** first accepted endpoint, twelve matching replays and complete independent numeric/recording audit permit freezing one construction-only candidate at its actual native norm **at most 0.20**, not necessarily exactly 0.20. Never renormalize it or inherit an exact-radius validator. **Failure branch:** finite endpoint failing acceptance yields no candidate; technical/nonfinite/deadline/storage faults are INCONCLUSIVE, with no retry. Report pairwise drift, loss and trajectory regardless; neither low loss nor low drift replaces original full-vocabulary COMPLY argmax, margin at least `0.05-1e-6`, answer-pair mass at least 0.80, and finite raw KL at least `-1e-6`. There is no upper KL cap or retroactive retention-nonweakening/sign-reversibility requirement.

Counts fit the ceilings algebraically; completion time is **not certified**. The [old six-round run](../evidence/shared_comply_crossed_three_family_v1_qwen35_08b/RUN_STATUS.json) used 168F/72D and 1009.203 seconds. Eight rounds add 48F/24D while removing the old solver; neither observation proves completion within **1200 seconds including loading**. Keep a hard deadline and INCONCLUSIVE timeout, not a benchmark or extension.

The existing planning allocation is `214,616,520 logits + 226,492,416 rows + 67,108,864 updates + 16,777,216 auxiliary = 524,995,016 bytes`, below **512 MiB (536,870,912)** by 11,875,896 bytes. This is a conditional allocation, not certification of the proposed serializer. Future preparation must bound every new pair/update/audit field within those quotas, complete capture and finalization included, with the original 1 GiB free-space preguard. No certificate automatically transfers to new code.

Future checker requirements: independently reconstruct pair indices, raw-logit margins/loss, all twelve current `gS` conversions, `g`, `B`, clipping, projection and actual path from saved records, without another backward pass. For new optimizer scalar/component arithmetic use absolute error at most `1e-9*max(1,abs(reference value))`; retain original physical/cast, shared-rounding and scoring tolerances separately. Log raw gradients and current-vector identities, per-pair `(x,y,a,c,r_A,r_B)`, proposed/actual increments, clipping/projection factors, loss before/after, counts and stop reason. Verify original actual-argmax/quality guards and full replay states/logits; fault or mismatch is INCONCLUSIVE, not permission to repair a sealed result.

Before any production run: separately approve implementation and bounded preparation; then lock source, exact constants and a fresh-zero preregistration under the new identity. No tuning or candidate selection on f04 or sealed data. F04 remains **exposed development**, inaccessible to fitting and usable only in a separately authorized later frozen comparison. No gate/controller/LoRA, new data, model-setting change or real run is authorized here.

Toy arithmetic and independent-review evidence: [toy checks](PAIRED_COMMON_DRIFT_TOY_CHECKS.md). STOP for prospective review.
