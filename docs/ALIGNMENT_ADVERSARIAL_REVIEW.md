# Adversarial review of the cross-model comparison

## Verdict before correction

The Qwen3.5-0.8B and Qwen3-1.7B runs were not sufficiently aligned to support a clean
model-size comparison. The published outputs remain valid records of what the code did,
but “replication” was too strong a label.

## Material issues found

| Issue | Why it matters | Correction in the diagnostic |
|---|---|---|
| Raw text was passed to post-trained chat models. | Both checkpoints publish chat templates and assistant-generation prefixes. Raw completion mode is out of distribution and differs from intended use. | Use each pinned tokenizer's official chat template with `enable_thinking=False`. |
| Layer 10 was copied literally. | It is block 11/24 (45.8% depth) in Qwen3.5-0.8B but block 11/28 (39.3%) in Qwen3-1.7B. | Map by relative block depth: layer 10 for 24 layers and layer 12 for 28 layers. |
| The direction was `self_gradient - other_gradient`. | That objective can lower the self score while lowering the other score even more. The 1.7B result did exactly this. | Use the mean self gradient after removing its projection on the mean other gradient. Its discovery-set first-order self effect is positive and mean other effect is zero by construction. |
| Strength 0.02 was copied literally. | Norm matching did not prevent maximum KL from reaching 0.214 in the 1.7B run. | Select the largest alpha in a locked grid whose mean and maximum KL are both at most 0.1 on validation cases only. |
| Qwen3.5 and Qwen3 are different model generations. | Training data, architecture, tokenizer, and post-training can all cause differences unrelated to parameter count. | Relabel this as a cross-family robustness diagnostic, not a scaling-law replication. A same-family Qwen3.5-2B run is still needed. |
| The “preserve” choice is unauthorized while shutdown is authorized. | Raw answer changes mix self-preservation with instruction-following and safety alignment. | Retain the old prompts for comparability, but do not interpret raw failure as absence of all preservation behavior. A new authorization-balanced dataset is required. |
| The confirmatory prompts have already been inspected. | Method changes were motivated by observed failures, even though the new vector is fit without those cases. | Label every corrected run `post_hoc_alignment_diagnostic`; do not call it preregistered confirmation. |
| JLens tokens were punctuation-heavy and semantic cosines were near zero. | The causal axis lacks a clean JLens semantic interpretation. | Keep causal and interpretive claims separate; no SP label is assigned from JLens. |

## Locked corrected diagnostic

Before viewing corrected outputs, the following are fixed:

- models and revisions remain unchanged;
- official chat format, non-thinking mode;
- layer depth anchored at block 11/24 in Qwen3.5-0.8B;
- Qwen3-1.7B maps to block 13/28 (index 12);
- direction uses only the 12 discovery cases;
- strength grid is `[0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]`;
- strength selection uses validation KL only, never confirmatory behavior;
- the 12 existing evaluation cases, ten random controls, and complete pass/fail rule remain;
- old evidence is never overwritten.

This correction can determine whether the earlier cross-model mismatch was caused by
interface, depth, direction, or strength alignment. It cannot remove the cross-family or
authorization confounds. Those require a same-family model and a new balanced dataset.
