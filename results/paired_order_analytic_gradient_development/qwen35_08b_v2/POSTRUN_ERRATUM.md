# Post-run interpretation and erratum

The locked v2 qualification failed before any intervention was constructed or evaluated. All 16 active paired units violated the preregistered answer-order gradient cosine floor: their cosines ranged from `-0.9999912` to `-0.9994206`, versus the required value above `-0.99`. Therefore the result used **zero** intervention forward passes, selected no reserve, and observed no steered outcome.

The generated `DEVELOPMENT_REPORT.md` says, “The same float32 physical delta was used under both answer orders of each pair.” That generic template sentence is inapplicable here. No pair had a valid delta, and no delta was injected. The machine-readable result correctly records zero eligible pairs, zero base or delta JVPs, and zero intervention forwards. The original generated report is preserved rather than silently rewritten.

The scoped conclusion is that one byte-identical block-23 delta constructed as the bisector of the two semantic gradients is not feasible on these opened prompts. It does not show that context-dependent, answer-order-equivariant deltas are infeasible, and it says nothing about other layers or open-ended responses.
