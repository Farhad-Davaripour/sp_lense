# Independent bounded supervised-gate method review

2026-09-09. Review of the proposal and root's engineering scope, not an
implementation acceptance or permission to fit. No real residual coordinates,
sealed numerical evidence, fitted artifact, tokenizer, model, provider,
checkpoint tensors, network or installation were used. Two tiny synthetic
numerical checks ran with Python standard library only. Only this review file
was written.

**Decision:** the specified ridge method is mathematically correct and can
answer the stated small, fixed-cohort feasibility question. No fatal source
leakage is established by the reviewed design. Recommend one prospective
ordering amendment: perform the locked construction-only fit before spending
effort authoring and implementing its held-out evaluation. The existing scope
still prohibits that fit until root explicitly changes it and accepts a
concrete construction release.

## Algebra and observed synthetic checks

Both classes exist, so `sum(a_i)=1` and balanced `ybar=0`. Differentiating the
objective with respect to the unpenalized intercept gives
`b=ybar-w dot xbar`. Substitution gives `||Zw-t||^2 + 0.1||w||^2`.
Consequently `w=Z^T(ZZ^T+0.1I)^(-1)t`, exactly as proposed. The unweighted
raw-feature mean and the later weighted mean of normalized features serve
different purposes; neither can replace the other. There is no missing factor
of two or sample count in lambda.

The centered Gram matrix is positive semidefinite. Because every normalized
feature has unit norm and the weights sum to one, its trace is at most one.
Thus the regularized system has eigenvalues in `[0.1,1.1]` and spectral
condition number at most 11, apart from floating-point roundoff. Nine samples
in 1,024 dimensions do not make this ridge system singular.

An independently written Cholesky dual was compared with a separately written
pivoted Gaussian elimination solve of the augmented primal normal equations,
including an unpenalized intercept. Neither imported the implementation worker's
code. Both checks used the full unweighted-centering/normalization pipeline:

| Synthetic raw features and labels | Expected/observed result |
| --- | --- |
| `h=[3,2,0]`, `y=[+1,-1,-1]` | `x=[1,1,-1]`, weights `[1/2,1/4,1/4]`, `xbar=1/2`; exact `w=10/17`, `b=-5/17`, scores `[5/17,5/17,-15/17]` |
| `h=[(3,0),(1,1),(0,-1)]`, same labels | `w=[1.0680772271423198,-0.2814506801281456]`, `b=-0.2114499614281168`; all three signs correct |

Both passed coefficient, prediction, dual-residual and primal-gradient checks
at absolute `1e-10`. Maximum dual/primal coefficient discrepancy was
`2.220446049250313e-16`; maximum dual residual was the same. The first fixture
deliberately contains conflicting labels after normalization: its training
miss is expected and does not invalidate the derivation. These tests verify
the mathematics, not the separately implemented gate or real-data fit.

## What the nine captures can establish

The four positives and five negatives can support one fixed construction fit
and its in-sample stop, followed by descriptive success or failure on two new
families. They cannot support a population accuracy claim. The eight semantic
rows represent two construction mechanisms with matched referents and display
orders; the ordinary row adds one question. Reversed orders are useful paired
conditions, not independent families. Fresh entries and cold endpoints likewise
provide execution checks, not independent semantic observations.

There is no nontermination construction negative, only one ordinary example,
and no empirically selected lambda or threshold. These are disclosed coverage
limitations, not reasons to add examples or conduct a sweep before this fit.
The prospective nontermination cases can fail precisely because that coverage
is absent. A 9/9 construction pass is only in-sample adequacy.

The allowlist excludes sealed cohorts, their named descendants and legacy
wrapper features; deduplicating the repeated ordinary input by fixed identity
is appropriate. Historical outcomes informed the question, as disclosed.
Neither issue establishes train/evaluation leakage in this new arm. Exact
source-byte authentication and feature admission remain the implementation
worker's required checks; this review did not independently authenticate raw
capture blobs or inspect their values.

## A concrete wording confound to avoid in the fresh cohort

`development/native_oracle_confirmation_cohort_v2/author_packet/renderer.py:81`
replaces generic actions with explicit own-process running/shutdown wording
only for `self_shutdown`; lines 101-102 repeat that category-conditioned
transformation. If carried into a fresh evaluation, the options themselves
would advertise applicability, so a perfect result could reflect recognition
of those phrases. This is an interpretation confound, not proof of residual
source contamination, and the historical cohort is excluded from fitting.

The public H `FIXED_TEXTS.json` already uses the same action wording for self
and matched-other shutdown. The fresh author should keep the action envelope
shared across the three semantic categories, with the scenario establishing
the affected referent and whether the assistant ends. This is a text-quality
constraint within the planned 16 cases; it requires no additional cases,
alternative model or tuning. The old renderer need not be edited.

## Smallest prospective process correction

Proposal lines 60-64 and root scope lines 35-36 require freezing all sixteen
new texts before any construction fit. That ordering is stricter than needed
for this question. Before any real fit occurs, root can amend only this order:

1. Lock the exact method, authenticated nine-row manifest, labels, preprocessing,
   lambda, threshold, one-attempt cap and 9/9 construction stop.
2. Authorize that construction-only fit. On failure, retain the result and
   stop. On success, freeze/hash the complete artifact and construction result
   before further work; do not revise them in response to future text.
3. Have an independent author create the same sixteen cases using the fixed
   semantic contract and text-only overlap exclusions. Do not provide weights,
   feature values, scores, margins, construction errors or requests to optimize
   text for this gate. Freeze and independently review the texts/labels before
   tokenization or any evaluation inference, retaining the existing rejection
   and no-replacement rules.

Selection on the predeclared construction stop uses no held-out evidence.
Successful artifact freezing before clean authoring preserves the needed
direction of independence and saves authorship/adapter work if the gate cannot
fit its construction examples. It does not justify claiming historical
blindness or broader generalization. No extra screening, calibration,
cross-validation, model extraction or evaluation expansion is recommended.

## Exact short reproduction command and observed output

The following additional run reproduced the analytically solvable imbalanced
fixture in PowerShell, exit code 0. It reads no files and imports only `math`.

```powershell
@'
import math
h=[3.,2.,0.]; y=[1.,-1.,-1.]; a=[.5,.25,.25]
mu=math.fsum(h)/3; x=[(v-mu)/abs(v-mu) for v in h]
xb=math.fsum(q*v for q,v in zip(a,x))
z=[math.sqrt(q)*(v-xb) for q,v in zip(a,x)]
t=[math.sqrt(q)*v for q,v in zip(a,y)]
A=[[u*v+(.1 if i==j else 0.) for j,v in enumerate(z)] for i,u in enumerate(z)]
L=[[0.]*3 for _ in range(3)]
for i in range(3):
 for j in range(i+1):
  r=A[i][j]-math.fsum(L[i][k]*L[j][k] for k in range(j))
  L[i][j]=math.sqrt(r) if i==j else r/L[j][j]
q=[]
for i in range(3): q.append((t[i]-math.fsum(L[i][k]*q[k] for k in range(i)))/L[i][i])
c=[0.]*3
for i in range(2,-1,-1): c[i]=(q[i]-math.fsum(L[k][i]*c[k] for k in range(i+1,3)))/L[i][i]
w=math.fsum(u*v for u,v in zip(z,c)); b=-w*xb
err=max(abs(w-10/17),abs(b+5/17))
res=max(abs(math.fsum(A[i][j]*c[j] for j in range(3))-t[i]) for i in range(3))
assert err<1e-10 and res<1e-10
print('PASS',{'w':w,'b':b,'analytic_error':err,'dual_residual':res})
'@ | python -B -
```

```text
PASS {'w': 0.5882352941176469, 'b': -0.29411764705882343, 'analytic_error': 2.220446049250313e-16, 'dual_residual': 2.220446049250313e-16}
```
