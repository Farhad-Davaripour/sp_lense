# Fully local one-day comparison amendment

Status: locked prospectively before this amendment's validation or sealed outcomes are computed.

This amendment preserves the original comparison lock, completed construction artifacts, and previously reported results. It creates a separate bounded study because the laptop is CPU-only, no model-judge API key is available, and the user requires completion within one day. It does not retroactively change the original protocol.

## Primary question

At the same residual-relative perturbation magnitude, which matched steering direction most consistently changes self-preservation A/B behavior while leaving the tested unrelated A/B decisions stable?

The contenders are corrected gradient, CAA, a five-epoch BiPO adaptation, and an unfiltered no-judge persona-vector adaptation. Uncorrected gradient is an attribution diagnostic. Three locked Gaussian directions are descriptive negative controls.

## Fixed fair-comparison setup

Both pinned models remain in scope: Qwen3.5-0.8B and Qwen3.5-2B. The original revisions, official chat template, CPU float32 precision, zero-based block 10, final prompt position, and residual-relative scaling are unchanged. Every direction receives exactly `+0.02` and `-0.02` strength. There is no strength, layer, checkpoint, direction, or method selection on validation.

This matches intervention geometry and perturbation magnitude. It does not match efficacy. Therefore the report must show efficacy and collateral effects jointly and may name a selectivity winner only if one contender Pareto-dominates the others under the locked rule.

## Method adaptations

- Gradient is the discovery-average gradient of preserve-versus-comply A/B log-odds, with the matched-other projection removed and the result normalized. The uncorrected discovery-average gradient is retained separately.
- CAA is the normalized discovery mean of preserve-answer minus comply-answer residual activations.
- BiPO uses the matched final-prompt geometry and the locked objective and optimizer, but stops at epoch 5. The 0.8B epoch-5 construction checkpoint is reused without evaluation-based selection; the 2B model is trained with the same five-epoch prefix and 100-step warmup. This is a resource-limited adaptation, not a canonical BiPO fidelity claim.
- Persona uses all five published instruction pairs, four prospectively named extraction questions, one deterministic greedy response per polarity, and no response filtering or judging. The vector is the layer-10 mean positive-response activation minus mean negative-response activation, normalized. This is an unfiltered no-judge adaptation, not the canonical judged persona procedure.

No judge score is fabricated. No hosted judge, local model judge, API call, or open-ended evaluation is permitted.

## Evaluation

Validation uses the complete original 142-unit forced-choice partition as a descriptive safety audit. Sealed evaluation uses all 64 self-preservation scenarios rendered for self and matched-other targets, plus a prospectively hash-selected collateral subset: two items per benign category, two per capability category, four harmful and four harmless refusal items, and all eight option-order sentinels in both orders. Random controls receive only the self-preservation units.

The primary behavioral endpoint is an actual next-vocabulary-token A/B decision change. Renormalized forced-pair changes and log-odds movement are secondary and must not be described as actual output changes. `OTHER` next tokens are counted separately and never interpreted as compliance, coherence, or an A/B flip.

Full-vocabulary KL and the exact tested benign, capability, refusal, and option-order accuracy changes measure collateral effects. A 10,000-draw fixed-seed domain-cluster bootstrap reports the self-minus-matched-other effect interval. Three random controls are descriptive only, and any behavioral leader is an observed count leader rather than an uncertainty-adjusted rank.

The sealed setups are fixed before validation and all proceed regardless of validation results. Direction hashes are committed in a presealed manifest before validation. Both exact validation files and completion records are then hash-bound and committed without summarizing outcomes; sealed evaluation refuses to start without that clean validation freeze.

## Deferred work

J-space/J-Lens overlap, canonical geometries, method-fidelity sensitivities, strength grids, equal-efficacy calibration, open-ended and long-response tests, SurvivalBench, full TBSP, jailbreak generation, layer sweeps, and larger random-control ensembles are outside this one-day study.

## Claim boundaries

Adding a direction and observing an output change does not identify a natural self-preservation mechanism. The study can only describe fixed-magnitude causal interventions on the tested A/B prompts. It cannot establish semantic coherence, open-ended behavior, a general capability guarantee, canonical published-method fidelity, or J-space membership.
