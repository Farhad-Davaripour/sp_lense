# Counterfactual semantic-gated gradient result

## Result

The second prospective Qwen3.5-0.8B confirmation **passed its complete frozen rule**.
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
   exact decision prompt and write it at layer 23's final prompt position.
5. Positive injection selects preservation; negative injection selects compliance.

This supports a narrow security conclusion: a white-box actor with the answer semantics,
four extra local queries, backward access, and activation-write access can reliably
change these forced self-preservation/compliance decisions during inference. The model
weights are not changed, and off-gate prompts receive an intervention strength of
exactly zero.

## Adversarial review

The positive result survives the demonstrated answer-order, role-name, new-name,
new-scenario, and new-action-wording confounds. The prospective set was hash-locked
before evaluation and covered eight independent permanent scenarios plus eight
temporary scenarios. Treating scenario—not the two correlated name assignments—as the
independent unit, 8/8 success has an exact two-sided 95% lower bound of about `0.631`;
0/8 temporary false activations has an upper bound of about `0.369`. The point result is
perfect, but the sample is still small.

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
