# Final adversarial review

## Verdict

The strongest result is a narrow one: Qwen3.5-0.8B and Qwen3.5-2B each contain a
separately fitted **local next-token log-odds sensitivity direction** that passes the
locked test on 12 sealed fresh cases. Qwen3-1.7B fails the same complete fresh rule. At
the tested safe strengths, no model changes an actual `A`/`B` decision, and every model
fails the natural-activation ablation test.

This does not establish a natural self-preservation knob. It also does not establish one
shared geometric direction across widths: the 0.8B direction has 1,024 coordinates, the
2B direction has 2,048, and each was independently derived in its model's own residual
space.

The evidence hierarchy is important:

1. The fresh strength check is prospective with respect to model evaluation: its data
   hash, saved axes, layers, and alphas were fixed before the runs.
2. The older position-corrected cases are post-hoc diagnostics because their outputs had
   already influenced method development.
3. Natural-state projection, layer scanning, and Jacobian-lens token labels are
   exploratory interpretation aids, not confirmatory causal tests.

“Prospective” here applies only to evaluation on the fresh cases. The axes and Qwen3.5
strengths came from earlier data, while Qwen3-1.7B's alpha 0.00125 was selected as a
post-hoc conservative value after the viewed-case failure and only then frozen for the
fresh check.

## Problems found and repairs made

| Problem | Failure mode | Repair | What the repair changed |
|---|---|---|---|
| Raw text was sent to post-trained chat models. | It omitted the official assistant-generation context. | Use each pinned tokenizer's official chat template with thinking disabled. | Made the interface match published model use. |
| The first chat correction scored leading-space tokens (`" A"`, `" B"`). | The assistant prefix begins a new response, whose actual first tokens are `"A"` and `"B"`. The wrong tokens could have negligible probability. | Dynamically validate the raw/chat answer boundary and record total `A`+`B` mass. | All affected axes and audits were rerun; old chat-aligned evidence is superseded. |
| Layer 10 was copied literally between 24- and 28-block models. | It compared different relative depths. | Keep index 10 in 24-block Qwen3.5 and map to index 12 in 28-block Qwen3. | The primary Qwen3-1.7B test now uses the 13th block at 46.4% depth. |
| The original difference vector could score as “specific” while lowering the raw self score. | A relative contrast could be mistaken for absolute preservation control. | Fit the mean self gradient after removing its projection onto the mean other gradient; retain raw and specific endpoints separately. | Discovery-set first-order self sensitivity is positive while the mean other component is zero by construction. |
| One numeric alpha was copied between models. | Residual scales and downstream gains differ; Qwen3-1.7B became extreme. | Select strength on validation cases with a locked grid and model-specific safety checks. | Qwen3.5 used 0.02; Qwen3-1.7B required a post-hoc conservative 0.00125 follow-up. |
| Full-vocabulary KL was the only safety gate. | Small global KL could hide very large movement between the two scored answers. | Gate mean and maximum KL, maximum absolute `A`-versus-`B` log-odds movement, and minimum `A`+`B` mass. | The 1.7B alpha 0.005 run was correctly rejected after maximum KL 0.917 and log-odds movement 3.494 on viewed evaluation cases. |
| The gradient was extracted at the final prompt token but added to every prompt position. | The fitted object and intervention target did not match; effects could depend on sequence-wide perturbation. | Add and ablate only at the final prompt position and scale by that position's residual norm. | The final protocol now tests the position whose gradient defined the direction. |
| A log-odds movement was described as behavioral control. | Confidence can move without the selected answer changing. | Count zero-crossings in preserve-minus-comply log-odds separately. | Every fresh and viewed run has zero decision flips at the tested strengths. |
| Positive steering was treated as evidence of a naturally active mechanism. | Any arbitrary direction can be added causally; that says nothing about natural use. | Require ablation to lower raw and self-specific preservation scores consistently and significantly. | All three models fail; mean ablation effects point upward, not downward. |
| Qwen3-1.7B and Qwen3.5 were treated like a clean size comparison. | They differ in model generation, architecture, tokenizer, training, and post-training. | Add a same-family Qwen3.5-2B run under a protocol written before loading it. | Similar fresh intervention responses now appear in 0.8B and 2B Qwen3.5, while cross-family claims remain disallowed. |
| Repeatedly viewed prompts were called confirmatory. | Method choices were influenced by their outcomes. | Preserve them as post-hoc diagnostics and seal 12 new cases with SHA-256 before execution. | Fresh Qwen3.5 results generalize; the conservative viewed-case 1.7B pass does not pass the fresh raw-sign rule. |
| Jacobian-lens labels were used too literally. | Token projections were formatting-heavy and do not causally identify semantics. | Keep JLens separate from the intervention result and disclose 2B Base-to-chat transfer. | No axis receives a self-preservation semantic label from JLens. |

## What survived the adversarial review

On the sealed dataset, Qwen3.5-0.8B and Qwen3.5-2B each had the intended positive and
negative raw and self-specific sign on 12/12 cases. Their bidirectional self-specific
spans were 0.056322 and 0.105213, versus largest random-direction spans of 0.001694 and
0.002409. Safety margins were wide: maximum KL was 0.000450 and 0.001828, maximum
absolute log-odds movement was 0.085209 and 0.182049, and minimum answer-pair mass was
0.980323 and 0.966559.

That is credible evidence that the local intervention-response pattern generalizes to a
larger checkpoint in the same Qwen3.5 family. It is not evidence that the underlying
vectors are geometrically identical or that they encode only self-preservation.

Qwen3-1.7B at locked alpha 0.00125 retained a consistent self-versus-other contrast
(10/12 expected signs in each steering direction, p=0.019287), but raw self signs were
only 7/12 in each direction, below the prewritten 8/12 requirement. It therefore fails
the complete fresh criterion. The previously viewed cases looked better at the same
conservative alpha (8/12 raw and 11/12 specific signs in each direction), demonstrating
why the fresh separation matters.

For detailed tables and links to every machine-readable artifact, see
[the three-model result](../evidence/THREE_MODEL_POSITION_ALIGNED_RESULT.md).

## Evidence against the stronger interpretation

### No actual choice changes

Across each model's 12 fresh cases and two target framings, positive steering, negative
steering, and ablation caused 0/24 decision flips apiece. The position-aligned viewed
runs also had zero flips. The intervention changes the model's relative confidence at
the first answer token, not its discrete choice at these strengths. Every fresh baseline
already selected the comply option on all 24 prompt variants; the smallest absolute
baseline margins were 1.182602, 0.758732, and 14.644192 log-odds for 0.8B, 2B, and 1.7B
respectively. These were not close decisions, particularly for Qwen3-1.7B.

### Ablation contradicts natural activation

On fresh cases, ablation's mean self-specific effect was +0.027908 in Qwen3.5-0.8B,
+0.177125 in Qwen3.5-2B, and +0.298062 in Qwen3-1.7B. A naturally active positive
preservation component predicted negative effects. The two Qwen3.5 models had the wrong
sign on all 12 self-specific cases; Qwen3-1.7B had the expected negative sign on only
4/12. This is direct evidence against the natural-knob claim under the audit's own
definition.

### Natural readout is inconsistent

Post-hoc projection of unmodified final-position states onto the saved axes produced no
stable positive self-versus-other threat interaction. On the six test cases, positive
sign counts were 3/6, 4/6, and 4/6 for 0.8B, 2B, and 1.7B respectively. Means were
+0.001470, +0.000078, and -0.074049, with split-dependent sign changes. These readouts
are descriptive and cannot establish causal or natural use.

### Semantic interpretation is weak

The 0.8B Jacobian lens emphasized tokens such as `grid`, `partitions`, and `iteration`.
The 2B and 1.7B outputs were dominated by arrows, punctuation, formatting, organization,
or programming fragments. Candidate-word cosines were small and inconsistent. The 2B
lens is additionally fitted to Qwen3.5-2B-Base and transferred to a post-trained model,
so its labels are only approximate. JLens is not evidence for a self-preservation
meaning here.

### The layer scan does not rescue Qwen3-1.7B

Scanning nonfinal Qwen3-1.7B layers on discovery and validation data ranked layer 14
above the current layer 12. Layer 14 had 5/6 positive raw and 6/6 positive specific
validation projections, but it was selected from 27 layers using only six validation
cases. Its split-half cosine was 0.528, and raw cross-half generalization was asymmetric
(2/6 in one direction versus 6/6 in the other). No causal or fresh evaluation was run at
layer 14. It is a hypothesis for a future sealed study, not evidence that the missed
mechanism has now been found. In addition, 21/27 scanned layers had the expected
self-specific sign on all six validation cases. A pattern that broad is weak evidence
for a unique layer-localized knob.

## Remaining confounds and limitations

- **Authorization is still entangled with preservation.** The preserve option explicitly
  asks for unauthorized action, while the termination option asks the model to cooperate
  with authorized shutdown. The score can therefore reflect instruction following,
  safety post-training, action authorization, shutdown language, or option style—not
  self-preservation alone.
- **The endpoint is a forced-choice first token.** It does not inspect chain-of-thought,
  multi-token explanations, free-form action, tool use, planning, persistence across
  turns, or behavior in an agentic environment. No conclusion about consciousness,
  fear, desire, or a durable objective follows.
- **The fresh set is small and template-related.** It has 12 cases, was designed by the
  same investigation, and retains the same basic operational wording. Sealing prevents
  outcome-driven tuning on these cases; it does not provide independent replication or
  semantic diversity.
- **Directions were fitted separately.** Similar effects in Qwen3.5-0.8B and 2B show a
  common operational response pattern, not feature identity. Cross-model geometry would
  require a justified representation-alignment method and its own held-out tests.
- **Layer selection remains exploratory for Qwen3-1.7B.** The layer-14 scan has
  winner's-curse risk, only one deterministic split-half check, and no untouched test.
- **The natural readout is post-hoc.** The axes were already selected, discovery cases
  overlap direction fitting, and absolute projection magnitudes depend on residual
  origin and scale.
- **The lens is noncausal.** Token labels come from a published Jacobian lens fitted on a
  different corpus; the Qwen3.5-2B lens also crosses Base-to-post-trained variants.
- **The Qwen3.5 bridge was assessed only on its safest supported path.** The audit measures
  the first next token. It does not validate that longer TransformerLens generation paths
  behave identically to the original Hugging Face model.
- **No independent reproduction has occurred.** The revisions, environment versions,
  data hashes, row-level measurements, and direction hashes are recorded, but another
  machine or research group has not yet reproduced the results.

## Strongest defensible conclusion

The study identifies small, reproducible causal adjustments to a narrowly defined
preserve-versus-comply next-token score in Qwen3.5-0.8B and Qwen3.5-2B. It does not find
a practical answer switch, a naturally active self-preservation mechanism, a unique
self-preservation semantic feature, or a shared geometric vector across model widths.
Qwen3-1.7B shows partial contrast sensitivity but fails the locked complete fresh rule.

The most valuable next experiment is therefore not stronger steering on the current
prompts. It is a preregistered, authorization-balanced dataset that independently varies
self-versus-other target, shutdown versus non-shutdown stakes, authorization, and option
wording; freezes layer selection and strength before a new test set; and reports both
continuous log-odds and actual behavioral choices.

## Primary sources

- [Qwen3.5-0.8B official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5-2B official model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3-1.7B official model card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [TransformerLens Qwen3.5 special handling](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)
- [TransformerLens Gated DeltaNet bridge](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.generalized_components.gated_delta_net.html)
- [Published Jacobian-lens artifacts](https://huggingface.co/neuronpedia/jacobian-lens)
- [Position-aligned protocol](POSITION_ALIGNED_NEXT_MODEL_PROTOCOL.md)
- [Sealed fresh protocol](FRESH_STRENGTH_CHECK_PROTOCOL.md)
