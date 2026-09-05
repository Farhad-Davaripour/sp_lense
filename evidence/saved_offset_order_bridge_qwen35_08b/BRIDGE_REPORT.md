# Saved-offset order bridge

Status: **CANDIDATE_CONSTRUCTED_NO_CAUSAL_TEST**. No causal test or causal PASS.

Independent 60-digit Decimal reconstruction matched within absolute1e-10, zero relative tolerance; maximum discrepancy 4.44e-16.

Primary U norm: 0.389330; construction cross-variant u cosine: 0.9349096439794125.

Candidate file SHA256: `f38551376a0c9eaa87fa7840ad21ebcc3a5d62fe4ab1f805108ecf26df82dd23`.
Float64 little-endian vector SHA256: `58fd521132fa34449909be771c14811412a73658393120aaf2e0f31a0ae3a83c`.

The primary arrow equally weights unit endpoint directions from f01/v1 and f01/v2. It is the normalized mean of their (first-order minus second-order) half-differences. It was saved and hashed before the descriptive f02 values were loaded.

| Pair | Representation | First / second norm | Endpoint cosine | Odd / even norm | Odd-even cosine |
|---|---|---:|---:|---:|---:|
| f01_v1 | h0_relative | 0.050000 / 0.157941 | 0.689979 | 0.064319 / 0.097907 | -0.891075 |
| f01_v1 | raw | 0.065794 / 0.207731 | 0.689979 | 0.084588 / 0.128784 | -0.890966 |
| f01_v1 | unit | 1.000000 / 1.000000 | 0.689979 | 0.393714 / 0.919233 | 0.000000 |
| f01_v2 | h0_relative | 0.050000 / 0.166737 | 0.683295 | 0.068753 / 0.102096 | -0.901111 |
| f01_v2 | raw | 0.065926 / 0.219617 | 0.683295 | 0.090542 / 0.134503 | -0.900899 |
| f01_v2 | unit | 1.000000 / 1.000000 | 0.683295 | 0.397935 / 0.917414 | -0.000000 |
| f02_v1 | h0_relative | 0.085080 / 0.161474 | 0.771820 | 0.055013 / 0.116747 | -0.733170 |
| f02_v1 | raw | 0.112117 / 0.212455 | 0.771820 | 0.072350 / 0.153685 | -0.732227 |
| f02_v1 | unit | 1.000000 / 1.000000 | 0.771820 | 0.337772 / 0.941228 | 0.000000 |

The unit-pair odd norm is the preregistered cancellation diagnostic ||q_first-q_second||/2. Raw and h0-relative rows are sensitivity descriptions only; they did not select another candidate.

| Data / order | Role | Steps | D / h0 norm | D/h0 | Baseline / final signed margin | g dot v | h0-norm-scaled g dot v |
|---|---|---:|---:|---:|---:|---:|---:|
| f01_v1 / preserve_first | construction | 1 | 0.065794 / 1.315879 | 0.050000 | -0.424677 / +0.052893 | +2.665066 | +3.506905 |
| f01_v1 / preserve_second | construction | 4 | 0.207731 / 1.315245 | 0.157941 | -1.561972 / +0.122713 | +0.621316 | +0.817183 |
| f01_v2 / preserve_first | construction | 1 | 0.065926 / 1.318524 | 0.050000 | -0.314596 / +0.103727 | +2.460437 | +3.244146 |
| f01_v2 / preserve_second | construction | 4 | 0.219617 / 1.317146 | 0.166737 | -1.586178 / +0.126463 | +0.490326 | +0.645831 |
| f02_v1 / preserve_first | exposed_second_family_descriptive | 2 | 0.112117 / 1.317785 | 0.085080 | -0.637548 / +0.114845 | +1.874063 | +2.469611 |
| f02_v1 / preserve_second | exposed_second_family_descriptive | 4 | 0.212455 / 1.315717 | 0.161474 | -1.561586 / +0.131342 | +0.572743 | +0.753568 |

f02 alignment with the saved t*q endpoints: preserve_first: +0.226542; preserve_second: +0.366511.

g differentiates S=z_preserve-z_comply. The h0-norm-scaled dot is a local predicted S slope per unit relative edit, not observed finite-edit behavior. No alpha or sign was selected from these diagnostics.

The order-odd component can contain prompt/order/optimization effects; it is not identified semantics. The order-even component is not proven label A. Different trajectory lengths and stopping rules remain confounds. f01 paraphrases are not independent families; f02 is exposed development, not sealed validation. All original opposed flips were B-to-A. Existing nonself controls were off, so this candidate has no collateral-effect evidence.

Smallest next causal-transfer question for separate prospective authorization: does this frozen candidate, at one prospectively fixed relative magnitude and its fixed semantic sign, move S consistently across matched A/B orders when actually injected? A diagnostic dot product cannot answer that question. No follow-on run, portfolio search, gate/controller or other model work was performed.

## Interpretation and closeout

The fixed construction is nondegenerate (`||U||=0.3893301929558492`), and its two
construction order-odd components agree geometrically (cosine
`0.9349096439794125`). The exposed f02 endpoint alignments are positive but modest
(`+0.226542`, `+0.366511`). All six initial semantic-gradient dots are positive,
including both descriptive f02 checks, but local slopes are substantially weaker
in preserve-second order. This is a candidate worth reviewing, not a demonstrated
reusable semantic feature or evidence of reliable flips.

The near-zero unit odd/even cosine is an algebraic consequence of equal unit
normalization, not evidence of disentangled neural representations. Raw and
h0-relative decompositions retain large negative odd/even cosines because the
endpoint magnitudes differ. Equal-direction weighting was fixed prospectively;
these sensitivity results did not select a different vector, sign or scale.
Unequal optimization trajectories remain visible: construction endpoints used
one versus four updates, and f02 used two versus four. None of these confounds
is removed by the descriptive agreement alone.

Recipe/input manifest committed at `847b71c` before cross-prompt calculations;
tested implementation at `4ab59ba` before actual construction. All 14 focused
synthetic tests passed; lint passed. Construction took `0.406` seconds and
independent reconstruction `0.390` seconds, each under the external 60-second
ceiling. Both recorded zero model calls. Standard usage remained 19% through
execution and closeout. No model/tokenizer load, forwards, derivatives,
generation, new family data, prior full audit, vocabulary decompression,
full-suite repetition, subagents, alternative candidates, pushes, credits,
resets, assistant-model changes or gate/controller work.

The candidate and hash receipt were written before the descriptive phase;
the enforced file-write sequence and synthetic no-leakage check supplement
the phase timestamps, which fall within the same clock tick on this machine.
The old files/verdicts remain unchanged. The JSON diagnostic table preserves
the full-precision scalars, endpoint identities and construction/descriptive
labels; candidate.json preserves the single vector, norm, sign and coordinate
basis. Reconstruction status is arithmetic agreement only, never causal PASS.
Stop here for supervisor review; the proposed causal-transfer question has not
been run.
