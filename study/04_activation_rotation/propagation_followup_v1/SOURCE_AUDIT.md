# Archived implementation audit and corrections

The audit uses executed_20261009/verified_native/source/analyze.py and collect.py,
not only the current checkout or an external summary. Their hashes are checked
against the immutable native EXPORT_HASHES manifest in the private audit receipt.

The original pairs() uses midpoint = (h[t]+h[t+1])/2 after TRAIN centering and
velocity = h[t+1]-h[t]. Thus a forward-Euler/chord critique does not describe the
implemented diagnostic. The historical held-out score evaluates the implicit
equation using both observed endpoints; it is not a strictly causal forecast.
Its score is 1-SSE/sum(||velocity||^2), averaged over held-out prompts, not a
conventional mean-target R-squared. Preserve those negative results under this
definition rather than relabeling or replacing them.

With row-vector states and delta = midpoint @ A, a causal update is
x_next = x_current @ [(I+A/2)(I-A/2)^-1]. The new fixed-rank comparison implements
this Cayley update using only current held-out states and TRAIN quantities.
Outside the fixed four-PC baseline-TRAIN subspace, every method predicts
persistence and all original coordinates remain in the error. Proper discrete
rotation uses determinant-positive orthogonal Procrustes. The baseline-TRAIN
center and basis are shared across intervention conditions within each cohort.

Reported full training-span rank 255 implies 255*254/2 = 32,385 skew entries,
not the unrestricted width-1024 count 523,776. The four-PC skew/rotation has six
degrees of freedom. Neither correction establishes sufficient independent data:
there are only four TRAIN prompts and four held-out prompts, with correlated
sampled continuations. Shared prefill is not independent replication.

collect.py registers steering before recorders, at decoder layers[10]. Native
BF16 h+0.5v is applied at all prefill positions and persistently at decode steps.
Each recorder retains the last residual position predicting each forced token.
Each replay calls native generate(use_cache=True) with the same baseline IDs.
Null-hook logits and untreated replay were checked. Saved native delivery norms
cover the last position of each call; earlier prefill-position delivery vectors
and KV/recurrent cache internals were not archived. Source provenance plus exact
upstream residual equality support the schedule audit but are not cache-state
measurements. Persistent cross-layer propagation is not spontaneous recovery
after forcing is removed.

The secondary validation is deliberately bounded. A shared Fourier phase across
channels preserves the full circular cross-covariance, including the coordinated
rotation of interest. Its 199 numerical controls retain planted rotation and it
is rejected as a rotation-destroying null; no real-data surrogate p-value is
reported. The synthetic sensitivity panel approximately matches observed
covariance/offsets/precision by retaining real outside-subspace residual
background. Therefore its nonrotation labels refer to the planted component,
not proof that the entire background is nonrotating. Specificity is inadequate.
Small positive four-PC causal scores do not rescue a rotational interpretation.
The rotation branch stops as inconclusive; there is no rank/layer/null search.

Method context: structured Procrustes/operator constraints are motivated by
https://arxiv.org/abs/2112.04307 . This analysis is not a reproduction of piDMD or
tensor-maximum-entropy neuroscience controls. The shared-phase counterexample
and magnitude/direction identity are directly checked numerically here. The
supplied external assessment is archived advice, not independent Qwen evidence.
