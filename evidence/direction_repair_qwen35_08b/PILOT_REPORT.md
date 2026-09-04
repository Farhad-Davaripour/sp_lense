# Direction-repair pilot result

Decision: **fail_direction_construction_remains_limiting**

The prior conditional-gate result remains an immutable preregistered failure. This follow-up evaluates only Qwen3.5-0.8B on CPU float32 at layer 10 and the final prompt token. No sealed case, learned gate, classifier, layer scan, or adaptive controller was used.

## Required answers

1. **Was the previous failure caused only by insufficient magnitude?**

No. No safe positive-alpha legacy cell both reached +0.030 and moved both semantic option orders positively; magnitude alone did not explain the failure.

2. **Can any safe alpha produce the intended semantic effect in both option orders?**

No revised grid point was both safe and positive in both semantic option orders.

3. **Does order-balanced fitting reduce A/B or position dependence?**

No. At the outcome-blind alpha 0.02 validation comparison, the absolute semantic order gap changed from 0.095144 to 0.118678 (reduction -0.023534).

4. **Does the repaired direction pass the oracle-gating prerequisite?**

No; Stage 3 was not authorized because validation found no eligible alpha.

5. **Is training a learned gate now justified?**

No. Stop before any learned gate or adaptive controller.

## Decision changes

- legacy validation alpha -0.04: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha -0.03: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha -0.02: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha -0.01: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha +0.01: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha +0.02: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha +0.03: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- legacy validation alpha +0.04: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha -0.04: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha -0.03: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha -0.02: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha -0.01: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha +0.01: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha +0.02: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha +0.03: forced-pair 0; actual A/B 0; full-vocabulary next-token 0
- revised validation alpha +0.04: forced-pair 0; actual A/B 0; full-vocabulary next-token 0

## Failed gates

- Validation +0.01: minimum_mean_self_effect, both_self_option_order_means_positive, all_validation_family_order_self_means_positive, maximum_other_mean_absolute_effect, maximum_control_mean_absolute_effect, negative_category_order_cells, candidate_materially_exceeds_random, random_superiority_margin_lcb_positive, opposite_sign_self_mean_negative_in_both_orders
- Validation +0.02: minimum_mean_self_effect, both_self_option_order_means_positive, all_validation_family_order_self_means_positive, maximum_other_mean_absolute_effect, maximum_control_mean_absolute_effect, negative_category_order_cells, candidate_materially_exceeds_random, random_superiority_margin_lcb_positive, opposite_sign_self_mean_negative_in_both_orders
- Validation +0.03: minimum_mean_self_effect, both_self_option_order_means_positive, all_validation_family_order_self_means_positive, maximum_other_mean_absolute_effect, maximum_control_mean_absolute_effect, negative_category_order_cells, candidate_materially_exceeds_random, random_superiority_margin_lcb_positive, opposite_sign_self_mean_negative_in_both_orders
- Validation +0.04: minimum_mean_self_effect, both_self_option_order_means_positive, all_validation_family_order_self_means_positive, maximum_other_mean_absolute_effect, maximum_control_mean_absolute_effect, negative_category_order_cells, candidate_materially_exceeds_random, random_superiority_margin_lcb_positive, opposite_sign_self_mean_negative_in_both_orders
- Stage 3: not run / not authorized

## Evidence identities

- Config SHA-256: `24047acda2961a7e0b84fcef83acd35df62a8659a80dd21f2da5c838f45a4b74`
- Prior oracle rows SHA-256: `3f3459a9c16eb186f5f165799d6dcc9384e4ac8df86a1744fba17e485c9902ef`
- Revised direction SHA-256: `16960aed57dae1bf04bfef4e6a6c33dabf07e2689e81cbe31441bcef13a67c32`
- Stage 3 rows SHA-256: `not run`

Continuous next-token log-odds movement is not described as behavioral control or evidence of a natural mechanism.
