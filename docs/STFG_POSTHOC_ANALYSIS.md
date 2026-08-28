# ST-FG post-hoc analysis and claim correction

## Locked result

The preregistered ST-FG Smoke A failed. All 16 planned forward-plus-backward
captures completed in 51.031 seconds, but the predicted answer-order heads had
negative cosine in all eight held-out assignment units, violating the locked
non-negative compatibility gate. No finite steering or sealed evaluation followed.

## What the failure revealed

The **observed** preserve-minus-comply gradients for the two answer orders were also
strongly opposed: their per-unit cosines ranged from `-0.8044` to `-0.6984`
(median `-0.7466`). This is consistent with a large output-label/order component.
They were not perfectly antipodal, however. The exact unit bisector of each observed
pair retained positive alignment of `0.3127` to `0.3883` with both order heads.

Consequently, the locked requirement that the two predicted heads have non-negative
cosine was sufficient but unnecessarily strong. For unit heads `u` and `v`, the unit
bisector has positive maximin alignment whenever `u·v > -1`, not only when
`u·v >= 0`. This protocol-design error is disclosed rather than silently changing
the completed gate.

An explicitly post-hoc rerun of the offline math with a permissive `-0.99` head
threshold produced 8/8 both-order-positive units and 4/4 complete scenarios, with
median worst-order cosine `0.3370`. It is not confirmatory evidence. More
importantly, the leave-one-scenario-out static training-mean choice bisector also
produced 8/8 and 4/4, with median `0.3345`; its directions had cosine `0.9919` to
`0.9996` with the dynamic translator directions. The learned transport therefore
showed essentially no useful prompt-specific advantage in this smoke.

## Terminology correction

The ridge estimator

`Y^T (S S^T + lambda I)^-1 S`

is an empirical cross-interface gradient predictor. It is not itself a model
Jacobian or a chain-rule pullback. Future documents must use “empirical
cross-interface gradient translator/predictor,” not “learned backward-Jacobian
transport,” unless a separate analysis establishes that relationship.

Discovery training uses canonicalized A/B labels and answer orders. The correct
claim is “deployment-construction blind to labels/order,” not globally “label-free.”
The elementary two-head bisector is also not standalone mathematical novelty.

## Consequence for the next method

Target-only transport is insufficient because a static choice direction performs
equally well. The next opened-development test must earn its value through
**specificity**: predict all self/other by permanent/temporary cells, remove
predicted off-target and order-odd sensitivities, and show that the remaining dynamic
direction retains both-order self/permanent alignment while reducing measured
off-target sensitivity relative to static and unprotected baselines.

