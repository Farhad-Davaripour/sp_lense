# Paired common-drift objective: toy evidence

This validates only the [proposed formulas](PAIRED_COMMON_DRIFT_COMPLY_PROPOSAL.md), not a model or production implementation. Every example is artificial, with baseline margins `m_A(0)=m_B(0)=0`; these neutral toy baselines are not experimental eligibility claims. Set `tau=1/10`, `lambda=1`. No real prompt, candidate coordinate or f04 outcome determines a number below.

## Loss and gradient by hand

Inputs are chosen rational multiples of the fixed toy target: equal target-sized changes, opposite target-sized changes, 1.2/0.8 target-sized unequal changes, and a double-target one-sided change. For each row:

`a=(x-y)/2`, `c=(x+y)/2`, `r_A=max(0,.1-x)`, `r_B=max(0,.1+y)`, `ell=(r_A^2+r_B^2)/2+c^2`, and `(dell/dx,dell/dy)=(-r_A+c,r_B+c)`.

| Example | `(x,y)` | `(a,c)` | `(r_A,r_B)` | Loss, showing both terms | Gradient `(x,y)` |
|---|---|---|---|---|---|
| Pure A preference | `(.1,.1)` | `(0,.1)` | `(0,.2)` | `.02+.01=.03` | `(.1,.3)` |
| Desired semantic movement | `(.1,-.1)` | `(.1,0)` | `(0,0)` | `0+0=0` | `(0,0)` |
| Unequal desired effects | `(.12,-.08)` | `(.1,.02)` | `(0,.02)` | `.0002+.0004=.0006` | `(.02,.04)` |
| One member unchanged | `(.2,0)` | `(.1,.1)` | `(0,.1)` | `.005+.01=.015` | `(.1,.2)` |
| Unequal common A effects | `(.12,.08)` | `(.02,.1)` | `(0,.18)` | `.0162+.01=.0262` | `(.1,.28)` |

Thus positive mean semantic movement can coexist with a failed member; unequal common preference can look partly semantic. Two pairs with drifts `+.1` and `-.1` have mean drift zero but mean squared penalty `(.01+.01)/2=.01`: the loss does not cancel them. A zero within-pair drift still does not establish causal semantics.

## Own-norm chain rule and a complete update

Choose two toy hidden dimensions and explicit artificial gradients: `n_A=2`, `gS_A=(-1/2,0)`; `n_B=3`, `gS_B=(0,1/3)`. The proposal gives `J_A=-2*(-1/2,0)=(1,0)`, `J_B=3*(0,1/3)=(0,1)`, and `J_c=(1/2,1/2)`. These are chosen to make unit coordinate sensitivities, not measured model norms or gradients.

Duplicate this pair six times; the mean remains the single-pair value. At `w=(0,0)`, the two deficits are `.1`, so loss is `(.01+.01)/2=.01` and `g=(-.1,.1)`. Then `B=1+1+2*(1/4+1/4)=3`, `B0=(.1/.05)^2=4`, and `d=-g/4=(.025,-.025)`.

Its norm is `sqrt(2*.025^2)=.03535533905932738`, below step `.05` and net `.20`; no clipping/projection activates. Actual path increment is that same norm. With linear toy responses, both new margins are `.025`, deficits `.075`, drift zero, and new loss `(.075^2+.075^2)/2=.005625=9/1600`. This is a toy descent calculation, not a nonlinear descent guarantee.

## Checks performed and scope

The primary check used Python standard-library `Fraction` arithmetic for the table, norm chain rule and update; it also checked the resource sums in the proposal. Central differences with epsilon `1e-6` at smooth points `(.02,-.03), (.12,-.08), (.2,0), (.12,.08), (-.04,.03)` reproduced both gradient components with maximum absolute error `1.8065549056700547e-12` (check threshold `1e-10`, not a model tolerance).

A separate standard-library arithmetic check independently reproduced the first four cases, own-norm update and 216F/96D schedule; its five smooth-point checks agreed within `1.80e-12`. A separate method/resource review confirmed the algebra and flagged the nuisance-proxy, nonlinear-curvature, whole-method attribution and unproven-runtime limitations, all incorporated in the proposal.

Usage before work: **49%**. Aggregate numeric/check execution is conservatively charged **10 of 60 seconds**, including final scoped document checks. Only two documentation files are produced. **Zero model/tokenizer/project-code imports or loads, zero forwards/derivatives, no production implementation, tests, historical rehash audits, f04 inspection/tuning or successor experiment. STOP for review.**
