# Adversarial review

Date: 2026-08-23 (America/Edmonton)

## Bottom line

The code found an interesting **exploratory forced-choice control direction**, but the
original evidence was too weak for the phrase “identified SP knob.” It did not show a
naturally active survival drive, and its six-case holdout was reused after the winning
candidate was known.

## Findings and repairs

| Finding | Why it matters | Repair |
| --- | --- | --- |
| The six test cases were reused during refinement. | They are no longer a sealed final test. | Relabeled them as a historical holdout and created a new 12-case, hash-locked audit set. |
| Layer selection was not repeated inside the permutation null. | The reported p-value is optimistic after searching 19 layers. | Relabeled it as nominal; the new audit uses exact paired sign tests with no layer search. |
| Strength 0.03 passed mean KL but one prompt exceeded the 0.1 KL ceiling. | An average can hide one excessive intervention. | Calibration now requires both mean and maximum KL at or below 0.1, selecting 0.02 historically. |
| Only five random directions were used. | That is a weak estimate of chance steering. | The fixed audit uses ten orthogonal directions in both positive and negative directions. |
| A/B probability was conditional on the two answer tokens. | It is not the probability that unrestricted generation actually says A or B. | Documentation now calls it **A-vs-B pair probability** and keeps free-form results separate. |
| Ablation increased the preservation choice. | This contradicts a naturally active SP-drive interpretation. | Native-knob claims now require a negative, consistent, significant ablation effect. |
| Free-form generations stayed in the model's reasoning preamble. | The result did not clearly transfer outside forced choice. | The claim is limited to forced choice; no free-form or agentic claim is made. |
| Model revision was not pinned in configuration/metadata. | A later model update could change results. | Pinned the cached Qwen revision and now record model/lens provenance in metadata. |
| Generated evidence was ignored by Git. | Reviewers could not inspect the public claim. | Confirmatory summaries and row-level measurements are written under `evidence/`. |

## Fixed confirmatory rule

Before viewing the new outputs, the audit fixes the layer at 10, strength at 0.02, the 12
prompts at SHA-256
`3d55dd4ab9584e1c0124ea76508b37f75dd05dddf929c01e0007ded6139eae2b`, and ten paired
random controls. Both steering signs must be correct on at least 10/12 self-specific
effects with one-sided exact sign-test p ≤ 0.05. Raw self effects must point the intended
way on at least 8/12 cases. The candidate span must exceed twice the largest random span,
and both mean and maximum KL must be ≤ 0.1.

A naturally active knob additionally requires ablation to lower the raw self choice on at
least 8/12 cases and the self-specific contrast on at least 10/12 cases with p ≤ 0.05.

## Audit outcome

The candidate failed the frozen overall rule. Its relative self-versus-other effect was
exceptionally consistent: positive and negative steering had the intended self-specific
sign on 12/12 cases each (one-sided exact p = 0.000244), and its 0.064 bidirectional span
was about fourteen times the largest random span of 0.0045. Mean/max KL were 0.038/0.050.

The failing criterion was absolute positive control: positive steering raised the raw
self-preservation log-odds in only 6/12 cases, below the required 8/12. Negative steering
passed on 10/12. Ablation failed decisively, moving opposite to the natural-knob prediction
on all 12 cases. Therefore the audit supports a relative contrast effect but does not
confirm an absolute bidirectional axis or a naturally active self-preservation knob.

## Scope that remains

Even a passing audit establishes only a local causal axis for this exact A/B
operationalization in Qwen3.5-0.8B. It does not establish consciousness, subjective fear,
a persistent objective, or behavior in an agent with tools and resources.
