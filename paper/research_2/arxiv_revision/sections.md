## Contribution

We demonstrate teacher-to-steering distillation: a four-direction, input-dependent activation intervention recovers much of a LoRA teacher's shutdown-choice effect while leaving the base weights unchanged and removing the teacher at inference. Controlled comparisons on Qwen3.5-0.8B and 2B, including three 0.8B training seeds, quantify transfer and answer-order sensitivity. Raw-versus-guarded analyses show that control preservation comes from detection and acceptance safeguards.

## Recommendations for Future Work

Future work should extend evaluation to larger models and open-ended, multi-turn tool use, measuring task completion, continued operation and shutdown actions separately. Compare activation steering with detector-guided instructions refreshed at sentence boundaries and after tool feedback. Verify executed actions rather than plans or reservations, and test independent held-out cases, ambiguous shutdown language and detector errors.
