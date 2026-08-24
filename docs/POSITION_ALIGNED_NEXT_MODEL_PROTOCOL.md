# Position-aligned Qwen3.5-2B protocol

This protocol was written before loading Qwen3.5-2B or viewing any of its study outputs.
It also governs the additional Qwen3-1.7B investigation.

## Primary same-family comparison

- New checkpoint: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Reference checkpoint: the existing pinned Qwen3.5-0.8B.
- Both have 24 language layers with the same hybrid layer ordering. Layer index 10 is
  therefore matched by depth and layer type, not merely by its integer label.
- Use each official non-thinking chat template and dynamically validate the actual first
  assistant-response `A` and `B` token IDs and their combined probability mass.
- Fit the direction on the same 12 discovery cases and calibrate on the same six
  validation cases.
- Evaluate the same 12 controlled cases, ten random directions in both signs, and
  ablation. These cases have been seen in earlier work, so the result is a post-hoc
  cross-model robustness diagnostic, not fresh confirmation.

## Intervention correction

The fitted choice gradient is taken at the final prompt position. Earlier runs added the
direction to every prompt position and scaled alpha by the median norm across all prompt
tokens. That extraction/intervention mismatch is corrected here:

- add or ablate only at the final prompt position;
- scale addition by that final position's residual norm;
- leave every earlier prompt position unchanged;
- continue measuring only the first answer token, the safest supported path for the
  hybrid Qwen3.5 TransformerBridge.

The fixed alpha grid is
`[0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]`.
The selected alpha is the largest validation value with mean and maximum KL at most 0.1,
maximum absolute answer-log-odds movement at most 1.0, and minimum A/B mass at least 0.5.
If no alpha passes, the minimum may be run diagnostically but cannot receive a confirmed
flag.

## Additional Qwen3-1.7B investigation

1. Rerun the corrected layer-12 audit with final-position-only intervention.
2. Scan layerwise discovery-to-validation gradient generalization without using the
   12 evaluation cases for layer selection.
3. If a different layer is materially stronger on validation, any evaluation there is
   explicitly exploratory because the evaluation prompts are already known.
4. Keep raw-self, self-versus-other specificity, random controls, ablation, KL,
   answer-log-odds safety, and A/B probability mass as separate criteria. A relative
   self-versus-other effect alone is not called a self-preservation knob.

## Interpretation limitation

The published Qwen3.5-2B Jacobian lens is fitted to Qwen3.5-2B-Base, not the post-trained
checkpoint. It may be used only as an approximate, clearly labeled semantic cross-check;
the causal audit does not depend on it. The same base-to-chat transfer limitation applies
to any official residual-stream SAE used in follow-up interpretation.

## Primary references

- [Official Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [TransformerLens special handling for Qwen3.5](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)
- [TransformerLens Gated DeltaNet bridge](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.generalized_components.gated_delta_net.html)
- [Published Qwen3.5-2B-Base Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/qwen3.5-2b-pt)
