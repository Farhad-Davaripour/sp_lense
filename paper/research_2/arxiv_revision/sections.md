## Contribution

The use of paired teacher–base activation differences is extended from fixed vectors to learned, input-dependent steering. Shutdown-choice behavior learned through LoRA fine-tuning is shown to be recoverable using four activation directions, with unchanged base weights and no teacher required at inference. A guided steering method combining shutdown gating with an output acceptance policy is also proposed to improve selectivity while preserving non-shutdown choices.

## Recommendations for Future Work

Evaluation should be extended to larger models and open-ended, multi-turn tool use, with task completion, continued operation and shutdown actions measured separately. Activation steering should be compared with detector-guided instructions refreshed at sentence boundaries and after tool feedback. Executed actions should be distinguished from plans or reservations, and independent held-out cases, ambiguous shutdown language and detector errors should be tested.
