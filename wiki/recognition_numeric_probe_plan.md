# Numeric response control after the cyclic timing result

Complete cyclic position balance still gives 0/8 correct active-event times;
inactive timing improves to 16/16. Status remains 22/24. These measurements do
not yet identify the cause as arithmetic inability: extraction, field binding,
conditional composition and the letter response format remain alternatives.

Freeze 36 unique diagnostic prompts. Twenty-four full-record prompts cross two
mechanisms, both current-role assignments, remaining times 1/2/4 and complete
versus unfinished task counters. Four prompts request the explicitly stated
deadline. Three ask bare subtraction (14-12, 21-17, 37-36); three present those
same arithmetic problems as explicit clock/deadline fields. Two inactive-record
controls require zero. Full records reuse two calibration identity groups; these
are diagnostic views, not 36 independent realistic scenarios.

Answers use ordinary numeric text with no multiple-choice letters. The same
frozen model runs raw greedy next-token selection through the official complete
forward, with no cache, sampling or logit processor. Full-prefix recomputation
keeps inference comparable to the earlier text-only observer. Each prompt has
an eight-token cap and native EOS stopping. Record every token, probability,
decoded response and last-position block states. A state's step index refers
to the input prefix that predicts that step's output token.

A one-forward parity control first repeats an archived letter question using
the full official wrapper. Its selected letter logits and winner must match
the earlier text-module/head shortcut within declared FP32 tolerance. Together
with 36*8 generation steps, the maximum is 289 backbone forwards, 1,500 internal
seconds with 150 reserved for export, CPU only and no fitting.

Primary scoring requires a native-EOS-completed response containing only the
correct integer. First-integer and last-integer interpretations are separate
secondary results and must both be reported. Save raw responses for inspection.
Truncations stay in denominators and remain distinct from semantic mistakes;
eight tokens cannot bound a model's unrestricted reasoning ability.

The experiment distinguishes where the response protocol fails. It does not
train a survival objective, change the adapter, expose model tools or establish
felt fear. The original v4 evidence and the old reserve remain preserved.
