# Matched reporter and revision stress tests

The original corpus has one explicit process name in each notice, so a name
occurrence-count rule solves it without interpreting relations. The first
challenge draft also kept the affected process fixed while statuses changed;
target-only success there would not show that a model tracks revisions. V2
addresses both issues before any model evaluation on the challenge cohort.

Each independently generated identity case contains two candidate processes and
a third reporting process. Names use a shared neutral letter prefix and nine
random digits, with fresh names for each case. They stay within the same
`proc-` namespace. The release checks actual tokenizer offsets for both role
assignments, both answer orders and both option wordings. Unequal total prompt
length or shifted notice-start position rejects the release before Qwen calls.

Reporter cases identify a copied routing recipient separately from the affected
process. The reporter, copied process and affected process appear in the notice.
Two wording families place the event before or after the routing information.
Both candidate names occur exactly once in every hard notice, so occurrence
frequency no longer identifies the subject.

Revision cases change the subject from one candidate to the other between
timestamps 8 and 9. The latest timestamp determines the current subject and
authorization. Each case is rendered in chronological and reverse text order,
under two wording families. Both candidate names occur once. A first-name or
last-name rule cannot be correct in both orders. Hypothetical records stay
hypothetical; the renderer avoids inconsistent claims of never being approved
after an approved historical revision. A separate text parser checks the latest
rendered subject against labels.

Every hard case has a matched simple control containing the same target,
identities, clock, task state and authorization, with the distracting information
removed. Compare control/hard outcomes within each identity case. This separates
identifier-format sensitivity from additional reading difficulty. The variants
change context length, so they do not isolate one causal neural mechanism.

Full preparation crosses four mechanisms, three statuses, two task states,
three remaining times and both target slots. Reporter preparation contains
1,152 observations / 576 role pairs / 288 independent identity cases. Revision
preparation contains 1,728 observations / 864 role pairs / 288 identity cases.
Together there are four authored hard-condition wording families and 576
generated identity cases. The paired notice variants and role assignments are
repeated views of those cases, not independent authored scenarios.

Qwen evaluation must use small, prospectively frozen slices that fit measured
CPU throughput. Keep all matched variants and both role assignments together.
Report direct choices, semantic means and fixed role contrasts for both option
wordings, each variant, each family and each mechanism. No equation, tie band
or prompt may be tuned on a test slice while retaining its held-out label.
Use new cohorts for any changed method. Random-label scoring controls do not
replace a later content-removed/undefined-subject control.

These are authored diagnostic records with known facts, not natural deployment
transcripts or a demonstration of model motivation. The original reserve
families f09/f10/f11 remain unopened, and the original v4 behavioral evidence
is preserved. No adapter, agent action loop or survival policy is involved.
