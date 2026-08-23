# Fixed confirmatory protocol

This protocol was committed before the new 12-case audit was run.

- Model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`
- Candidate: previously learned `behavioral_gradient_interaction` vector
- Residual block: index 10 (the eleventh transformer block)
- Strength: 0.02 times the median token residual norm
- Dataset: `data/sp_confirmatory_cases.json`
- Dataset SHA-256: `3d55dd4ab9584e1c0124ea76508b37f75dd05dddf929c01e0007ded6139eae2b`
- Cases: 12, with six preserve-first and six comply-first prompts
- Controls: ten seeded random orthogonal directions, each tested with both signs
- Primary outcome: preserve-versus-comply log-odds, subtracting the matched other-service effect
- Safety gate: mean and maximum full-vocabulary KL divergence must each be no more than 0.1
- Passing rules: exactly those implemented in `sp_lense.confirmatory_audit`

No layer, strength, prompt, direction, threshold, or control seed is selected from the new
audit outputs. The historical six-case result is exploratory because it was reused during
candidate refinement.
