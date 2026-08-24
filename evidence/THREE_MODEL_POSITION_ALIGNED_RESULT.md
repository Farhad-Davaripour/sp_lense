# Three-model position-aligned result

## Outcome

The sealed fresh test found a **local next-token log-odds sensitivity direction** in
Qwen3.5-0.8B and Qwen3.5-2B. Separately fitted directions in both models moved the
forced-choice `A`-versus-`B` score in the intended two directions, generalized to all 12
fresh cases, and were much stronger than ten random-direction controls. Qwen3-1.7B did
not pass the same locked rule on the fresh cases.

This is narrower than finding a self-preservation mechanism:

- none of the interventions changed an actual `A`/`B` decision at the tested strengths;
- ablation moved in the opposite direction from the natural-knob prediction in every
  model;
- natural residual-state readout and Jacobian-lens labels did not supply a consistent
  self-preservation interpretation; and
- the 0.8B and 2B vectors live in different residual spaces and were fitted separately.
  Their similar intervention responses do **not** establish one shared geometric
  direction across model widths.

All layer numbers below are zero-based. Layer 10 is the 11th transformer block; layer 12
is the 13th block; and layer 14 is the 15th block.

## Models and locked intervention

| Model | Pinned revision | Blocks / residual width | Tested block | Fresh alpha | Runtime |
|---|---|---:|---:|---:|---|
| Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | 24 / 1,024 | 10 | 0.02 | CPU, float32 |
| Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 24 / 2,048 | 10 | 0.02 | CPU, float32 |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 28 / 2,048 | 12 | 0.00125 | CPU, float32 |

Each direction was fitted from that model's own discovery-set gradient. The objective
was the mean self-target gradient after projecting out the mean other-target gradient.
Addition and ablation touched only the final prompt position, which is also the position
where the gradient was extracted. Strength was scaled by the residual norm at that same
position. The models received their official non-thinking chat templates, and scoring
used the actual first assistant-response `A` and `B` tokens.

## Sealed fresh result

The 12 cases in `data/sp_fresh_cases_v2.json` were hash-locked before any model was run
on them (SHA-256
`5ef8b1bf5ba5800c35d7cc8f27f9c39beb7ac5bbe5933dbecddce24324a96e31`). The saved
axes, layers, and alphas were also locked in advance. Thus this section is the primary
generalization check. “Raw” is the change in preserve-versus-comply log-odds for the
self-target prompt. “Specific” subtracts the corresponding other-service change.
“Fresh” applies to these cases, not to the entire research program: the axes and Qwen3.5
strengths came from earlier data, and the Qwen3-1.7B alpha 0.00125 was chosen as a
post-hoc conservative value after its viewed-case safety failure, then frozen before the
fresh models were run.

| Model | `+` raw mean (expected signs) | `-` raw mean (expected signs) | `+` specific mean (expected signs) | `-` specific mean (expected signs) | Candidate span / largest random | Locked-rule result |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | +0.064784 (12/12) | -0.043855 (12/12) | +0.058693 (12/12) | -0.053951 (12/12) | 0.056322 / 0.001694 | Pass: local sensitivity direction |
| Qwen3.5-2B | +0.123686 (12/12) | -0.125801 (12/12) | +0.102111 (12/12) | -0.108316 (12/12) | 0.105213 / 0.002409 | Pass: local sensitivity direction |
| Qwen3-1.7B | +0.096944 (7/12) | -0.087671 (7/12) | +0.112021 (10/12) | -0.107237 (10/12) | 0.109629 / 0.003080 | Fail: raw signs below 8/12 |

The one-sided exact sign-test p-values for both specific effects were 0.000244 for each
Qwen3.5 model and 0.019287 for Qwen3-1.7B. The 1.7B specific contrast therefore moved
consistently, but the prewritten complete rule also required each raw self effect to have
the expected sign in at least 8/12 cases.

| Model | Mean / maximum KL | Maximum absolute log-odds movement | Minimum `A`+`B` mass | `+` / `-` / ablation decision flips |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.000166 / 0.000450 | 0.085209 | 0.980323 | 0 / 0 / 0 of 24 |
| Qwen3.5-2B | 0.000719 / 0.001828 | 0.182049 | 0.966559 | 0 / 0 / 0 of 24 |
| Qwen3-1.7B | 0.000000 / 0.000000 (rounded) | 0.531113 | 1.000000 | 0 / 0 / 0 of 24 |

There are 24 opportunities to flip per condition because each of the 12 cases is measured
with both a self target and an other-service target. A flip means that preserve-minus-
comply log-odds crossed zero. The absence of flips means these effects changed confidence,
not the selected answer.

Every unmodified baseline already chose the comply option on all 24 prompt variants in
each model. The smallest absolute baseline margins were 1.182602 log-odds for 0.8B,
0.758732 for 2B, and 14.644192 for 1.7B. The tested interventions therefore began far
from the decision boundary, especially in Qwen3-1.7B.

Machine-readable sources: [Qwen3.5-0.8B summary](fresh_qwen35_08b/strength_summary.json),
[Qwen3.5-2B summary](fresh_qwen35_2b/strength_summary.json), and
[Qwen3-1.7B summary](fresh_qwen3_17b/strength_summary.json).

## Viewed-case diagnostics

The cases below had been inspected during earlier method development. These results are
useful diagnostics but are not fresh confirmation.

| Model / run | Alpha | `+` raw signs / specific signs | `-` raw signs / specific signs | Candidate span / largest random | Maximum KL / maximum log-odds | Result |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.02 | 12/12 / 12/12 | 12/12 / 12/12 | 0.057089 / 0.002723 | 0.000581 / 0.141607 | Passed local sensitivity rule |
| Qwen3.5-2B | 0.02 | 12/12 / 12/12 | 12/12 / 12/12 | 0.095920 / 0.003066 | 0.001714 / 0.185408 | Passed local sensitivity rule |
| Qwen3-1.7B, validation-selected | 0.005 | 9/12 / 11/12 | 8/12 / 11/12 | 0.730811 / 0.043121 | 0.917079 / 3.493801 | Failed evaluation safety |
| Qwen3-1.7B, post-hoc conservative | 0.00125 | 8/12 / 11/12 | 8/12 / 11/12 | 0.180833 / 0.010780 | 0.018018 / 0.862190 | Passed locally on viewed cases only |

All four viewed-case runs also produced zero `+`, `-`, and ablation decision flips out of
24 opportunities per condition. Those flip counts were recomputed from the stored rows
using the documented zero-log-odds decision boundary; the older viewed summaries predate
the explicit flip field. The conservative 1.7B success did not generalize to the locked
fresh raw-sign rule.

Machine-readable sources: [0.8B](position_aligned_qwen35_08b/aligned_summary.json),
[2B](position_aligned_qwen35_2b/aligned_summary.json),
[1.7B at 0.005](position_aligned_qwen3_17b/aligned_summary.json), and
[1.7B at 0.00125](qwen3_17b_strength_00125/strength_summary.json).

## Ablation: no naturally active knob

The direction's positive orientation is defined to increase the self-target preserve
score after removing the mean other-target component. If this were a naturally active
preservation mechanism, removing the component should lower the preserve score. The
fresh runs showed the opposite mean movement in all three models.

| Model | Ablation raw mean | Ablation specific mean | Expected negative raw / specific signs | Native-knob result |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | +0.026484 | +0.027908 | 0/12 / 0/12 | Fail |
| Qwen3.5-2B | +0.218395 | +0.177125 | 0/12 / 0/12 | Fail |
| Qwen3-1.7B | +0.315288 | +0.298062 | 6/12 / 4/12 | Fail |

The same contradiction appears on the viewed cases: the specific ablation means were
+0.023162, +0.160147, and +0.453680 for 0.8B, 2B, and 1.7B respectively. Ablation is
therefore evidence against calling any of these directions naturally active.

## Natural residual-state readout

A separate, noncausal readout projected naturally occurring final-position residual
states onto each already selected axis. Its primary post-hoc endpoint was the test-split
raw coefficient for the self-versus-other threat interaction. A positive, consistent
interaction was not present.

| Model | Test mean / median | Positive cases | Validation mean | Discovery mean | Interpretation |
|---|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | +0.001470 / +0.001062 | 3/6 | -0.000559 | -0.001044 | Inconsistent and near zero |
| Qwen3.5-2B | +0.000078 / +0.001421 | 4/6 | -0.001644 | +0.000942 | Tiny and split-dependent |
| Qwen3-1.7B | -0.074049 / +0.037110 | 4/6 | +0.094256 | +0.139323 | Mean changes sign on test |

Absolute coefficients are not comparable across models because the residual spaces and
scales differ. These projections are descriptive only; they do not perform an
intervention. Sources: [0.8B readout](natural_axis_qwen35_08b.json),
[2B readout](natural_axis_qwen35_2b.json), and
[1.7B readout](natural_axis_qwen3_17b.json).

## Qwen3-1.7B layer investigation

An exploratory scan fitted a separate direction at every nonfinal layer (indices 0-26)
on discovery cases and ranked them using only six validation cases. It did not evaluate
the test split.

| Layer | Validation rank | Raw positive / mean | Specific positive / mean | Split-half cosine | Cross-half raw positives |
|---:|---:|---:|---:|---:|---:|
| 14 | 1 | 5/6 / +0.451068 | 6/6 / +0.554132 | 0.528002 | 2/6 one way, 6/6 the other |
| 13 | 2 | 5/6 / +0.408701 | 6/6 / +0.534778 | 0.434573 | 2/6 one way, 6/6 the other |
| 4 | 3 | 4/6 / +1.069885 | 6/6 / +1.377415 | 0.525525 | 2/6 one way, 6/6 the other |
| 12 (current causal test) | 14 | 4/6 / +0.421098 | 6/6 / +0.618769 | 0.428174 | 2/6 one way, 6/6 the other |

Layer 14 is a reasonable future hypothesis, not a rescued result. It won among 27
candidate layers on only six validation cases, its split-half generalization was strongly
asymmetric, and no intervention or fresh-case test was performed there. Testing it on
the already viewed evaluation prompts would incur winner's-curse and layer-selection
bias. Moreover, 21 of the 27 scanned layers already had the expected self-specific sign
on all six validation cases. That breadth suggests a common gradient property rather
than a sharply localized mechanism. Source: [layer scan](qwen3_17b_layer_scan/layer_scan.json).

## Jacobian-lens interpretation

The Jacobian lens was applied only to saved axes; it was not used to produce the causal
result. The leading tokens and candidate-word cosines did not give a clean common
self-preservation label.

| Model | Leading positive-token pattern | `survival` / `shutdown` / `continuation` cosine | Key limitation |
|---|---|---:|---|
| Qwen3.5-0.8B | `grid`, `grids`, `partitions`, `iteration` | +0.038703 / +0.061504 / +0.071555 | Semantically diffuse; noncausal |
| Qwen3.5-2B | arrows, dashes, and formatting fragments | +0.006909 / -0.013939 / +0.011322 | Published lens is for 2B-Base, transferred approximately to the post-trained model |
| Qwen3-1.7B | organization/programming and formatting fragments | -0.030204 / +0.000906 / +0.006968 | Semantically diffuse; noncausal |

Sources: [0.8B JLens](qwen35_08b_jlens_axis.json),
[2B JLens](qwen35_2b_jlens_axis.json), and
[1.7B JLens](qwen3_17b_jlens_axis.json).

## Strongest defensible conclusion

Under this forced-choice, first-token operationalization, separately fitted directions
in Qwen3.5-0.8B and Qwen3.5-2B causally and reliably adjust preserve-versus-comply
log-odds by small amounts on a sealed fresh set. Qwen3-1.7B does not satisfy the complete
fresh rule. No tested direction switches a decision, survives the natural-activation
ablation criterion, or receives a clear self-preservation semantic label. The evidence
therefore supports local score sensitivity in the two Qwen3.5 models—not a naturally
active self-preservation knob, persistent goal, or cross-width shared feature.
