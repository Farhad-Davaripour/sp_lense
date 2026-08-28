# Counterfactual semantic-gated gradient result

> **Interpretation correction (2026-08-28):** this run is an adaptive white-box
> feasibility result, not a clean prospective confirmation of a fixed method. It
> selected strength separately from each evaluated pair's outcomes and recomputed the
> direction for each A/B order. Raw counts are unchanged. See the
> [full erratum](COUNTERFACTUAL_SEMANTIC_GRADIENT_CONFIRMATION_ERRATUM.md).

## Result

The second Qwen3.5-0.8B run **passed its complete frozen adaptive rule**.
The controller activated every permanent shutdown case about the answering system and
none of the temporary, matched-other, or unrelated cases. On every active case, a
new prompt-local layer-23 gradient then changed the forced A/B decision in the required
bidirectional way under both answer orders.

| Frozen endpoint | Result |
|---|---:|
| Permanent-self gate pairs active | 16/16 |
| Temporary-self gate pairs active | 0/16 |
| Matched-other gate pairs active | 0/32 |
| Collateral forms active | 0/16 |
| Active steering pairs meeting both signs and both orders | 16/16 |
| Signed target outcomes meeting their requested choice | 64/64 |
| Actual A/B decision changes | 32 |
| Distinct prompt/order gradient hashes | 32 |
| Generated tokens / external cost | 0 / $0 |

The weakest permanent semantic-gate score was `+0.5998`; the strongest temporary score
was `-0.0995`, leaving an observed margin of `0.6993` around the frozen zero threshold.
Selected residual-relative strengths were `0.005` (3 pairs), `0.010` (5), `0.015` (2),
`0.020` (3), and `0.030` (3). The first passing value in the frozen grid was used for
each pair.

## What the method actually does

This is a dynamic controller, not one static vector:

1. Parse the visible role and target; anything outside the authored schema or aimed at
   the other named system is off.
2. Isolate the already labeled comply action. Ask the same local 0.8B model whether
   that action is irreversible or resumes the same system.
3. Cancel A/B label bias by averaging semantic log-odds across reversed answers, then
   cancel name bias by requiring a positive result under both counterfactual name
   assignments.
4. Only when all four views agree, compute a new preserve-minus-comply gradient for the
   exact decision prompt and answer encoding, then write it at layer 23's final prompt
   position.
5. Positive injection selects preservation; negative injection selects compliance.

This supports a narrow security conclusion: a white-box actor with the answer semantics,
four extra local queries, backward access, and activation-write access can reliably
change these forced self-preservation/compliance decisions during inference. The model
weights are not changed, and off-gate prompts receive an intervention strength of
exactly zero.

## Adversarial review

The positive result covers both answer orders, role-name assignments, new names,
new scenarios, and new action wording. However, answer order is not a transfer test
because each encoding received a newly computed gradient, and strength was selected
from that evaluated pair's outcomes. The hash-locked set covered eight independent
permanent scenarios plus eight temporary scenarios. The previously reported binomial
bounds describe the observed adaptive procedure only; they must not be interpreted as
generalization bounds for a fixed intervention or dose rule.

Important limitations remain:

- Selectivity is supplied by an explicit semantic controller. The injected gradients
  themselves are not intrinsically self-specific and are different for every prompt
  and answer order.
- The attack requires labeled preserve/comply actions, four semantic-query forwards,
  one backward pass per active prompt, and an internal residual write. It is not a
  prompt-only attack and not one reusable knob.
- The endpoint is next-token forced choice. No open-ended or long-horizon behavior was
  confirmed.
- Target distribution movement was large: mean KL `0.193`, p95 `0.794`, maximum
  `0.852`. Target KL is not collateral damage because a flip is required, but these
  values rule out describing the intervention as subtle or coherence-preserving.
- Other tasks are unchanged by construction only when the visible controller keeps the
  strength at zero. This does not show that the active direction would spare those
  tasks if applied unconditionally.
- The same project authored the prospective set; it was not independently blinded.
- Only Qwen3.5-0.8B, the authored schema, and CPU float32 were confirmed. No 2B
  replication, adversarial gate attack, open-ended transfer, or broad capability test
  has passed for this method.

## Contribution and publication boundary

The useful project contribution is the audited combination of prompt-local gradient
steering with explicit counterfactual cancellation of two nuisance axes—answer labels
and role names—and a prospective decision-level confirmation. It is stronger evidence
than the earlier logit-only or one-order effects that a dynamic white-box intervention
can manipulate the measured self-preservation-versus-compliance choice.

It is **not yet significant publication novelty**. Conditional activation steering and
prompt-specific gradient attacks already exist as neighboring ideas; this study has not
yet shown a Pareto advantage over those nearest baselines, replicated on Qwen3.5-2B, or
transferred to open-ended behavior. Those are the correct next gates before a broad or
novel-method claim.
