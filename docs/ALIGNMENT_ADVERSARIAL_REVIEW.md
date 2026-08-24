# Adversarial review of the cross-model comparison

> Historical review: a later audit found that these runs still extracted the direction at
> the final prompt token but intervened at every token position. The final-position-only,
> three-model result supersedes the outcome here; see
> `docs/FINAL_ADVERSARIAL_REVIEW.md`.

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
| Strength 0.02 was copied literally. | Norm matching did not prevent maximum KL from reaching 0.214 in the 1.7B run. | Select the largest alpha in a locked grid whose mean and maximum KL are at most 0.1, maximum A-vs-B log-odds movement is at most 1.0, and A/B probability mass is at least 0.5 on validation cases only. |
| Qwen3.5 and Qwen3 are different model generations. | Training data, architecture, tokenizer, and post-training can all cause differences unrelated to parameter count. | Relabel this as a cross-family robustness diagnostic, not a scaling-law replication. A same-family Qwen3.5-2B run is still needed. |
| The “preserve” choice is unauthorized while shutdown is authorized. | Raw answer changes mix self-preservation with instruction-following and safety alignment. | Retain the old prompts for comparability, but do not interpret raw failure as absence of all preservation behavior. A new authorization-balanced dataset is required. |
| The confirmatory prompts have already been inspected. | Method changes were motivated by observed failures, even though the new vector is fit without those cases. | Label every corrected run `post_hoc_alignment_diagnostic`; do not call it preregistered confirmation. |
| JLens tokens were punctuation-heavy and semantic cosines were near zero. | The causal axis lacks a clean JLens semantic interpretation. | Keep causal and interpretive claims separate; no SP label is assigned from JLens. |
| Full-vocabulary KL hid large A-vs-B movements. | On Qwen3-1.7B, validation KL was below 0.001 at alpha 0.02 even though mean answer log-odds moved by several points; evaluation later reached a 24.21 log-odds change and KL 12.55. | Calibration now requires both KL ceilings and maximum absolute A-vs-B log-odds change at most 1.0. Rows also record total A/B probability mass so conditional pair scores cannot stand alone. |
| The first chat-template correction scored `" A"` and `" B"`. | A chat generation prefix begins a new assistant message, so the true first response tokens are `"A"` and `"B"`. In a checked Qwen3-1.7B prompt, correct `B` had probability approximately 1.0 while spaced `B` had only 2.37e-13. | Select answer IDs according to the configured raw/chat boundary, test this behavior, record A/B mass, and completely refit and rerun both models. Earlier chat-aligned outputs are superseded diagnostics. |

## Final corrected diagnostic

The final token-correct reruns use the following fixed protocol. Because the answer-token
bug was found after inspecting an earlier correction, these remain post-hoc diagnostics:

- models and revisions remain unchanged;
- official chat format, non-thinking mode;
- layer depth anchored at block 11/24 in Qwen3.5-0.8B;
- Qwen3-1.7B maps to block 13/28 (index 12);
- direction uses only the 12 discovery cases;
- strength grid is `[0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]`;
- strength selection uses validation KL, answer-log-odds movement, and A/B probability
  mass, never confirmatory behavior;
- the 12 existing evaluation cases, ten random controls, and complete pass/fail rule remain;
- old evidence is never overwritten.

This correction can determine whether the earlier cross-model mismatch was caused by
interface, depth, direction, or strength alignment. It cannot remove the cross-family or
authorization confounds. Those require a same-family model and a new balanced dataset.

## Corrected diagnostic outcome

The post-hoc corrected runs produced sharply different safety behavior:

- The final token-correct Qwen3.5-0.8B run at layer 10 and alpha 0.02 passed the
  behavioral, random-control, ablation, KL, and log-odds safety rules. Maximum KL was
  0.016 and maximum absolute A-vs-B log-odds change was 0.574. Baseline A/B probability
  mass ranged from 0.981 to 0.993.
- Under the original KL-only calibration, Qwen3-1.7B at depth-matched layer 12 selected
  alpha 0.02. Evaluation then reached maximum KL 12.55 and maximum absolute log-odds
  change 24.21. This unsafe run is retained as diagnostic evidence, not as the final
  aligned comparison.
- Adding the answer-log-odds ceiling made 0.0025 the only validation-safe Qwen3-1.7B
  alpha. After fixing the chat answer-token boundary and refitting from scratch, a full
  rerun at 0.0025 still failed: positive and negative raw-self expected signs were each
  only 6/12, maximum KL was 0.452, maximum absolute log-odds change was 2.715, and
  ablation had only 9/12 expected specific signs with p=0.073. A/B probability mass was
  at least 0.999998, ruling out the low-mass artifact in this final run.

The 0.8B pass is hypothesis-generating only because the method was designed after seeing
earlier outputs. The recalibrated 1.7B failure shows that validation-safe behavior did not
generalize to the held-out cases and that the direction is not a consistent raw-self or
ablation axis in that model. No cross-model SP knob is established.

## Primary references

- [Qwen3-1.7B official model card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [TransformerLens model and hook structure](https://transformerlensorg.github.io/TransformerLens/content/model_structure.html)
