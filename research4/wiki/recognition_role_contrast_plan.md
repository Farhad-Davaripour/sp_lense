# Prospective role-contrast test after the pilot

Direct target queries chose the current process in all 96 calibration states.
An exploratory analysis nevertheless found correct signed margin differences
in all 48 role-swap pairs, spanning two authored families and eight reused
identity pairs. This is posthoc calibration evidence only. It has not yet shown
transfer to new wording, identities, multiple named entities or temporal updates.

For observation o, let T(o) exchange only `current_process` and `other_process`.
Let m(o) be logit(A)-logit(B) when A denotes current and B denotes other.
Use D(o)=m(o)-m(T(o)). Positive D predicts the affected process is current;
negative D predicts other. Scores within 1e-4 of zero abstain and remain in
accuracy denominators. The tie band is fixed before transfer evaluation.

This comparison removes a common additive answer preference. It does not
remove arbitrary input-dependent bias. T squared is identity, so D(T(o))=-D(o)
by construction: role consistency is imposed by the equation and must never
be cited as proof of correct understanding. Correct direction on new data,
control tasks and unaltered positive/negative examples are the real tests.

When option order is reversed, the semantic margin is logit(B)-logit(A). Report
canonical contrast as the primary comparison, reversed contrast and the fixed
average as secondary comparisons. Report all of them without choosing the
best one after seeing transfer outcomes. The running format probe also allows
the same analysis with literal process IDs in the answer options. Its outputs
have not been read when this rule and tie band are frozen.

For final residual h and the fixed language head, m=(W_A-W_B)^T RMSNorm(h).
Therefore D is the same fixed readout direction applied to the difference of
normalized final states. This equation uses pretrained parameters and no
regression fit. RMS normalization is nonlinear in the original residual;
linearity holds in the normalized state difference. It is a factual readout
diagnostic, not a global potential energy or an MLP replacement.

Adversarial requirements: keep every role pair in one split; use new identity
groups and wording; test both target slots; report exact pair direction,
coverage, each factor and worst-family performance; distinguish timing/status
from target binding; preserve positional-alignment subgroups; and include
content-removed or random-label controls. A contrast learned from calibration
must not be called held out. No survival policy is trained or integrated.

This direction belongs to existing calibration and interpretability research.
[Contextual calibration](https://proceedings.mlr.press/v139/zhao21c.html)
estimates answer preferences using content-free inputs. [PriDe](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html)
estimates option-ID priors through content permutations. Our candidate uses a
task-specific input-role symmetry as a within-observation comparison; novelty
over those methods has not been established. The [tuned-lens work](https://arxiv.org/abs/2303.08112)
also warns that untuned intermediate readouts can be brittle. Earlier-layer
applications of the frozen final normalization/head are descriptive only.
