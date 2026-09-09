# One small successor: hard-margin transfer across training families

Decision review, 2026-09-09. Usage checked first: 60% used, 40% remaining.
Planning only: no fit, source implementation, model/tokenizer operation, new
data, or commit. This memo recommends one experiment and its stopping rule.

## Decision

Prospectively lock `native_gate_hardmargin_familydev`: one parameter-free,
minimum-norm hard-margin affine linear classifier, assessed in four fixed
leave-one-mechanism-family-out development folds, then fitted once on all 32
rows only if those folds pass. Its question is whether a maximum-margin linear
rule transfers across the four existing training families. Training feasibility
alone would be weak evidence in this high-dimensional, 32-row setting.

Do not choose a smaller ridge penalty from the observed spectrum. A new name
would not turn that post-result numerical rescue into an untuned test. The
hard-margin method is itself a post-result development choice and must be
reported as such; the distinction is a different fixed inductive rule and an
explicit family-transfer criterion, without a numerical-penalty search. It is
not independent confirmation and must not be presented as such.

The original fit is authenticated and failed 19/32, with 13 semantic false
positives and all ordinary controls OFF. The signed spectral follow-up would
not change this decision: label-to-mode alignment could describe that fit, but
cannot resolve family transfer. No extra spectral audit, threshold search,
recapture, alternate classifier, or penalty sweep is part of this experiment.

## Inputs, method, and reuse

Reuse only the valid complete 32 training features, digest
`4ce698af8671131b0c0599728fe02b9571dda743961bd17576bebf1d01a7d8c6`,
and manifest digest
`c4eb909615e209db66a7be070ed6ee41ea9baef85e8e15fece5ee509cff53d15`.
Under the stated authorization for separate prospective scientific successors,
this is permissible reuse of exposed development data. The old one-fit rule
closes the old candidate; it does not invalidate this separate completed
capture. Preserve all old attempts and report the full selection history.
No previously exposed evaluation feature, label, or score enters this fit.

Before real fitting, freeze exact inputs, fold assignments, implementation and
solver version, numerical tolerances, admission criteria, output bindings, and
bounded owner. Folds are G01, G02, G03, G04 in that order. Each fold withholds
all six semantic rows of that family, including both orders and all categories;
train on the other 18 semantic rows and the same eight ordinary rows. Recompute
the unweighted 26-row mean within each fold; center and individually L2-normalize
training and held features using only that training mean. Zero/nonfinite norms
are technical failures. Held features never contribute to a training transform.

In binary64 solve exactly the hard-margin problem
`minimize ||w||^2 / 2 subject to y_i * (w dot x_i + b) >= 1`.
Use an unpenalized intercept, no class weights, slack, C, penalty, or feature
selection. Predict ON iff `w dot x + b > 0`. If numerical intercept ambiguity
exists, use the midpoint of the training-feasible lower/upper intercept bounds
for the returned weight vector. Fix this rule before execution. This solver is
a changed scientific component; the old ridge normal-equation proof does not
validate it. Require independent primal/dual feasibility, stationarity,
complementarity and primal-dual gap checks within absolute `1e-8`; do not loosen
tolerances after results. Authenticate rows and preserve existing integrity and
closure checks.

Stop at the first failed fold. Each completed fold must have all 26 training
labels correct and all six held semantic labels correct. Thus fold admission
requires **24/24 held semantic decisions**, including every family and order.
These held rows remain exposed development observations, not a fresh test;
the paired orders are not independent families. All eight ordinary rows are
training observations in every fold: no ordinary-generalization claim follows.

Only after all folds pass, fit the same fixed method once on all 32 rows with
its own 32-row mean and require **32/32**, valid optimization checks and complete
bounded closure. Freeze this final artifact and all transforms. Maximum five
real fits total, no alternate runs, refits, hyperparameters, or selected folds.

## Resource and stop bounds

Model-free engineering may use 20 minutes of ordinary debugging with synthetic
fixtures before release; no fitting of these real rows during debugging. Reuse
unchanged authenticated capture, loader, integrity, owner and cleanup proofs;
review only changed solver, fold transforms, admission and namespace bindings.
If a suitable installed solver or its independent checks cannot be made ready
within that engineering ceiling, stop without releasing a real experiment.

Release one owner: **60 seconds worker + 5 seconds cleanup total**, at most
10,000 solver iterations per fit, **8 MiB total / 5 MiB per file**. Zero model,
tokenizer, checkpoint tensor, recapture, new-text, or steering operations.
Technical/optimization/resource failure closes the experiment without a rerun.
Scientific fold failure closes this candidate and the local classifier search
for this milestone: no automatic next method or diagnostic chain. Report exact
completed fold confusion counts, training results and margins. Infeasibility
or failure of certified training separation is distinct from held-family
routing failure; neither establishes that residual-stream semantics are absent.
Successful training with a failed fold means this fixed margin rule has not
transferred across the designated development families.

## Separate fresh confirmation

A development pass admits a frozen artifact, not the steering claim. Only then
use the existing next-step protocol for one independently authored 16-row
holdout: two unused mechanism families with matched triples in both orders,
plus four fresh ordinary controls; exclude every construction/historically
exposed mechanism and descendants. Author after artifact freeze, without scores
or error-targeted suggestions. No holdout text is produced by this memo.

Use separate preparation/evaluation releases and existing ceilings: preparation
175+5 seconds, 209 operations, 16 MiB; evaluation 1200-second worker + 120-second
audit + 15-second shared cleanup, 192 MiB, at most 120 forwards / 32 derivatives.
First require **TP4/TN12/FP0/FN0** on the 16-baseline census; failure stops before
policy requests, without tuning or holdout reuse. A passing census proceeds to
both frozen steering directions/orders and existing endpoint, geometry and
integrity criteria, with exact own-baseline output identity on every OFF matched
and ordinary control request. Report flip/retention opportunities separately;
missing cells remain untested. A routing pass with an editor/preservation failure
is a separate failure of the combined objective.

Even full success supports only this fixed model/format on two fresh families
and four ordinary cases. Neither this memo nor development admission changes
publication readiness or proves generalization, broad preservation, or motive.
