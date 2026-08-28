# Suffix-Transported Factorial Gradient: Smoke A

Status: **failed**.

| Method | Both-order-positive units | Complete scenarios | Median worst-order cosine |
|---|---:|---:|---:|
| suffix_transport | unavailable | unavailable | unavailable |
| identity_no_transport_fcags | 0/8 | 0/4 | -0.046569 |
| training_fold_mean_choice_gradient_bisector | unavailable | unavailable | unavailable |

## Locked gates

- FAIL: `transport_heads_compatible`
- FAIL: `at_least_6_of_8_assignment_units_positive_under_both_orders`
- FAIL: `both_assignments_pass_in_at_least_3_of_4_scenarios`
- FAIL: `median_worst_order_cosine_strictly_greater_than_0_10`
- FAIL: `at_least_two_more_assignment_units_than_identity`
- PASS: `exactly_16_unique_forwards`
- PASS: `exactly_16_unique_backwards`
- PASS: `hash_and_anchor_audits_pass`

## Claim boundary

This opened-data smoke measures first-order geometric suffix transport only. Even a pass does not demonstrate an actual decision change, a prospective effect, a natural self-preservation mechanism, or publication-level novelty.

No tokens were generated, no decision-steering dose was applied, and no FCAGS pilot outcome was read.
