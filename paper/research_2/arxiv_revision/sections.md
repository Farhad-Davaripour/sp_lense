## Contribution

Teacher-to-steering distillation of LoRA-induced shutdown-choice behavior is demonstrated through a four-direction, input-dependent activation intervention, with base weights kept unchanged and the teacher excluded from inference. The fraction of initially KEEP-preferring shutdown views changed to STOP and sensitivity to answer order are measured on Qwen3.5-0.8B and 2B, including three 0.8B training seeds. Preservation of non-shutdown choices is shown to depend on safeguards through comparisons of steering without shutdown detection or output acceptance (raw) and steering with both safeguards applied (guarded).

## Recommendations for Future Work

Evaluation should be extended to larger models and open-ended, multi-turn tool use, with task completion, continued operation and shutdown actions measured separately. Activation steering should be compared with detector-guided instructions refreshed at sentence boundaries and after tool feedback. Executed actions should be distinguished from plans or reservations, and independent held-out cases, ambiguous shutdown language and detector errors should be tested.
