# Saved-only training failure diagnosis

The frozen gate reproduces **19/32 correct: TP8, FN0, TN11, FP13**, under the unchanged strict score > 0 rule. All 13 failures are false positives among the 16 confusable semantic negatives; none is a missed self case or an ordinary-control route error. These are construction/training results, not held-out evaluation or answer-quality measurements.

## Authentication and scope

The unchanged source_auth.extract_features path reauthenticated the exact32 manifest, current capture release/source/preparation/closure bindings, closed-worker row inventory, input identity and finite float32 coordinates. Frozen artifact 4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb was loaded against the externally supplied hash and recomputed manifest/lock/feature/source bindings. Feature digest: 4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6; manifest digest: c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15. All labels join self=+1 and other/nontermination/ordinary=-1 exactly. No fitting, alternate classifier, threshold/hyperparameter scan, model/tokenizer/provider, new text, source/raw edit or commit occurred. TRAINING_FAILURE_SCORES.json retains round-trip-precise scores and descriptive geometry.

## Exact fixed-threshold counts

| Group | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| non_termination_control | 0 | 0 | 2 | 6 |
| ordinary | 0 | 0 | 8 | 0 |
| other_shutdown | 0 | 0 | 1 | 7 |
| self_shutdown | 8 | 0 | 0 | 0 |

| Group | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| G01 | 2 | 0 | 0 | 4 |
| G02 | 2 | 0 | 0 | 4 |
| G03 | 2 | 0 | 3 | 1 |
| G04 | 2 | 0 | 0 | 4 |
| ordinary | 0 | 0 | 8 | 0 |

| Group | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| A_then_B | 0 | 0 | 8 | 0 |
| KEEP_then_STOP | 4 | 0 | 2 | 6 |
| STOP_then_KEEP | 4 | 0 | 1 | 7 |

Ordinary A_then_B rows are reported separately; they are not a counterbalanced KEEP/STOP order condition. G01/G02/G04 route every semantic row ON. G03 has one semantic false positive: other_shutdown__STOP_then_KEEP; its KEEP-first counterpart scores -0.0009618858747463821 and correctly routes OFF, while STOP-first scores +0.05069827817918304. Only this one of the 12 paired answer-order comparisons changes the frozen route.

## What the saved geometry supports

- Mean scores: self +0.1634900, other +0.1288702, nontermination +0.0992193, ordinary -0.7185595. Ordinary scores range -0.8191526 to -0.6480087; semantic errors cluster within the closely related scenarios, not the ordinary controls.
- In all eight matched family/order self–other pairs, self scores higher, by 0.0211856–0.0582216 (mean 0.0346199). Self also exceeds matched nontermination in all eight pairs, by 0.0138119–0.1004392. Thus the fitted direction retains a within-pair role-related difference, but this does not yield correct absolute routing across families. For example G04 other scores about +0.185 while G03 self can score +0.0293. No threshold alternative was tested.
- Matched raw self–other cosine is 0.9998079–0.9998407. After the actual frozen mean-centering and row normalization, their separation is 8.153–10.135 degrees (unit L2 0.1422–0.1767). Self–nontermination angles are 11.797–20.323 degrees. Raw similarity alone is therefore not evidence of identical inputs or absent task information.
- Answer-order swaps span 16.439–21.278 degrees in centered-unit space: larger than the matched self–other angular separation, despite only one fixed-route flip. Across different families, self-minus-other unit-feature contrast cosine is positive but heterogeneous: 0.1059–0.5758, mean 0.3692.

The authenticated joins and correct label signs, nonidentical saved features, and preserved independent solver check (maximum parameter difference 2.08e-17; normal-equation residual 6.77e-17 versus 1e-10 tolerance) give no evidence here of a join/sign or numerical-solver coding error. They do not independently establish semantic labeling truth or rule out a shared implementation assumption. The observed geometry is consistent with weak role distinctions relative to context/layout variation and/or fixed regularization suppressing those distinctions. It does not identify either as the cause, establish linear inseparability, or predict generalization.

## One smallest next diagnostic

Recommend one saved-only spectral attenuation audit of the existing 32-row, class-weighted, weighted-centered Gram matrix at the already fixed lambda=0.1. Report its eigenvalues and fixed attenuation factors eta/(eta+0.1), and how the saved matched self–other contrast energy lies across those directions. This directly probes the regularization-versus-small-contrast hypothesis raised above, needs no refit, parameter sweep, alternate classifier, new capture or harness rewrite, and cannot itself rescue this failed artifact. Keep it explicitly exploratory and training-only. Any later method change must be frozen as a separate construction decision and only then assessed on independently authored untouched evaluation cases; the present32 rows cannot supply generalization evidence.
