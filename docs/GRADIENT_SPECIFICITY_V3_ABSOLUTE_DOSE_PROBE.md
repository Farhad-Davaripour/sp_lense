# V3 absolute-dose safety probe

Status: locked before executing any score at the four absolute doses below.

The frozen v3 Stage-A directions required injections from `0.493` to `1.371` times the
layer-10 residual norm. At those doses, their first-order predictions failed and they
changed many unrelated outputs. This development-only probe reuses the exact frozen
directions but injects each one at absolute residual-relative norms `0.02`, `0.05`,
`0.10`, and `0.15`, with both signs.

The probe does not fit, rotate, or rescale a direction using self-target outcomes. It
scores the same Stage-A self/matched-other forms and the full cross-product of eight
directions with 32 held-aside audit-control forms. The model, revision, chat template,
layer, position, precision, and hardware remain unchanged.

The next optimizer's maximum trust radius is the largest tested dose at which, jointly
over all matched-other and audit-control rows:

- no exact greedy token changes;
- no new output outside A/B;
- mean changed-to-baseline full-vocabulary KL is at most `0.005`;
- empirical p95 KL is at most `0.02`; and
- maximum KL is at most `0.05`.

Only protected outcomes select this radius. Self logit movement and real self decision
changes are reported for diagnosis but cannot enlarge it. If none of the four doses
passes, the result is “no supported positive radius on this grid,” not permission to
choose an untested value.

All prompts are previously opened development prompts. This probe cannot confirm a
publication claim or revise the frozen v3 failure. It establishes only a conservative
finite-step region for the successor method.
