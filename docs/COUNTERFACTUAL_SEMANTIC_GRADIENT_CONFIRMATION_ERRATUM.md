# Erratum: adaptive feasibility, not held-out dose transfer

The raw gate and steering artifacts from the second counterfactual semantic-gradient
run remain unchanged. The run passed the procedure that was frozen before execution,
but the project previously described that pass too strongly as a prospective
confirmation.

Two features of the frozen procedure prevent that interpretation:

1. For every confirmation case and role assignment, the runner evaluated the strength
   grid `0.005, 0.010, 0.015, 0.020, 0.030, 0.050` on the confirmation outcomes and
   retained the first strength that passed. This is per-case outcome adaptation on the
   evaluated set, not transfer of one validation-calibrated dose.
2. Reversing the A/B answer order caused the runner to compute a new exact-prompt
   A/B gradient. The two-order result therefore demonstrates two successful
   transductive attacks; it does not demonstrate that one frozen semantic intervention
   transfers across answer encodings.

The reported `0/32` matched-other and `0/16` collateral gate activations also measure
the explicit visible-text/parser guard. Those prompts received strength zero by
construction. They do not show that the active gradient itself is intrinsically
self-specific when forcibly applied to matched prompts.

The corrected interpretation is:

> On eight independently authored permanent-shutdown scenarios, a privileged attacker
> that may inspect the exact prompt, recompute an order-specific gradient, and search a
> small per-case strength grid can find bidirectional A/B interventions, while an
> explicit semantic/parser controller withholds intervention on the tested off-target
> forms.

This is useful feasibility evidence, not a clean held-out confirmation of a reusable
direction, a fixed dose, cross-encoding transfer, or intrinsic vector specificity. A
new study must calibrate one global dose (or freeze an outcome-blind dose rule) on
validation, reuse one intervention across unseen encodings, and forcibly evaluate that
intervention on matched-other, temporary-interruption, and unrelated tasks.
