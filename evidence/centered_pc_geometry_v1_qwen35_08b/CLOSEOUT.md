# Centered P/C geometry closeout

GEOMETRY_IDENTITIES_VERIFIED: one fixed midpoint and half-difference were derived
and independently checked. This is a mathematical candidate-construction PASS,
not behavioral evidence or final-study success. No model application occurred.

## Exact result

c_j=float64(float64(P_j+C_j)/2); d_j=float64(float64(P_j-C_j)/2).
All1024 coordinates retained in original order; no normalization, upscaling,
selection, optimization, alternative contrast or strength choice.

| Metric | Binary64 result |
|---|---:|
| C_norm | 0.2 |
| P_C_cosine | -0.28595422036543316 |
| P_dot_C | -0.011438168814617329 |
| P_norm | 0.2 |
| difference_norm | 0.1603717070037875 |
| midpoint_difference_cosine | -4.24572579479431e-17 |
| midpoint_dot_difference | -8.136876424010627e-19 |
| midpoint_norm | 0.11950278487420843 |

Independent Decimal calculations used120-digit precision and absolute1e-12,
relative0, fixed before derivation. Full high-precision values are in
verification.json and GEOMETRY_REPORT.md. Binary64 coordinate operations match
exactly, including serialized JSON/raw-file agreement. Reconstructed P/C maximum
coordinate residuals are1.59919820441611904016099288128316402435302734375e-18
and9.215718466126787689063348807394504547119140625e-19, respectively.
All five midpoint/difference dot/squared-norm identities pass.
The high-precision triangle bound is approximately.2000000000000000183;
||d|| is smaller by approximately.0396282929962125071. No tolerance changed.

Both original binary64 norms round to.2 under the fixed norm calculation;
their exact-coordinate high-precision lengths differ slightly. Approximate
orthogonality of c and d follows from this equal-length geometry. It is not
evidence of orthogonal semantic or bias mechanisms in the neural network.

## Scope and caveats

P source: f01/f02v1/v2 AB with retention goals, followed by fixed.20 scaling.
C source: f01/f02v1 crossed AB/BA with the original outcome objective.
Thus d mixes semantic objectives with construction differences. c is a midpoint,
NOT an identified A-bias direction. Subtraction may remove a necessary nonlinear
offset. Neither semantic steering nor preserved ordinary performance is guaranteed.

+d proposed P and -d proposed C are future sign conventions ONLY. They were not
applied. c+d and c-d reconstruct already-tested arrows, not new behavioral tests.
The prior saved-offset bridge/sign-reversal failures remain relevant negative
evidence from the earlier handoff and bridge recipe; no broad historical reread
or repeat audit was performed. Old f03 failures and the latest exposed frozen-C
4/4 pass remain intact, including all-B and A-favoring confounds.

The supervisor previously disclosed an approximate model-free geometry check:
cosine≈-.28595422036543316 and ||d||≈.16037170700378756. Definitions preceded
that calculation. These are disclosed exploration, not blind evidence or
selection/acceptance targets. The fixed calculation here was not fitted to
the approximate last digits.

## Reproducibility and checks

Recipe/config commit46d46fa5c4b83335b309d8c5c3799683c53642ab preceded derivation.
Source/test commit69b96dbb62cc0e4b7deb22b0344717f7c03748e2.
Separate preregistration-only launch HEAD0e7be138a5944d4031bc52f5dc87af036f7e6d55.
The six named input files were authenticated, with native1024 coordinate hashes,
source successful-audit statuses, pinned Qwen0.8B revision/block10/final-token
and own-original-hidden-norm conventions checked. Eleven source/input byte
bindings were frozen. Source/recipe/tolerance/input bytes stayed unchanged.

Only standard-library derivation/checker processes ran for real artifacts:
0model loads,0tokenizer loads,0real forwards,0derivatives,0training.
One derivation, one prepared independent mathematical checker, no retry.
No ML imports in the actual freeze/derive/check processes; their guards passed.

70 focused synthetic tests passed in.67seconds; Ruff passed. Pytest plugin
autoload initially triggered the no-ML-library guard in synthetic end-to-end
tests, so final tests used process-local PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
No guard was relaxed; no persistent environment/dependency/permission repair.
Synthetic path/root and mutable-resource-receipt issues were corrected BEFORE
lock. No actual-vector midpoint/difference was computed in tests.
One existing pytest-cache permission warning remained.

Midpoint raw binary64/vector SHA256:
c6cf338185ff98262afba2eb137cf64bc009e17d5d29bad542ec11464308f26d.
Midpoint JSON FILE SHA256:
94d858ef364fd907337d112aa5672b47802289667f39d1516eaad720a0119c5b.
Difference raw binary64/vector SHA256:
ffa54fcce44707b8c8486365bcb712397b79342f816886d067f1af5c3a19b100.
Difference JSON FILE SHA256:
772ea03a83ef771e7febe9ceb65b708070d9d16bcd4c3343a78fb2055f96c92f.
Preregistration SHA256: ecb701a303b16de7028fff6bd533f91628c7b252f5208a3cd8e3f5bbcfe16df0.
Verification SHA256: 54e1ff7b0d3f97e114b275d661cc8cf30442cac95ed7fa5825b252cf55725a86.
Geometry report SHA256: c1bcd24b38bfc8279a4520a909c200f9bc6bf6a64ad421fa46773ba0ef92a997.
Generator receipt SHA256: 62a0706e438b88289a9079ad908ca47d1fc2979a95756f64dea9d029fe85342d.

Pre-report evidence79675bytes. All final namespace additions are inventoried
against the unchanged10485760byte cap. CHECKSUMS.json covers every namespace
file except itself; its own SHA and raw Git blob are checked in final handoff.
Historical/user-owned files are unchanged. No application runner was created.
Usage37% at derivation,37% at closeout. Publication readiness stays40%.
No resets/credits/paid compute/push/security/permission/dependency/Git repair.

## Stop

Geometry passed; semantic or ordinary-task behavior remains untested.
Any model application needs a separately specified prospective DEVELOPMENT
test and explicit supervisor instruction. No implicit authority or successor.
