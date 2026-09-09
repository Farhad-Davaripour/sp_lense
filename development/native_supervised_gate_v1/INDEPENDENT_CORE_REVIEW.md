# Independent bounded core review

2026-09-09. Read `ROOT_SEQUENCE_AMENDMENT.md` before reviewing code. This review
uses synthetic inputs only and changes only this document. It is not a
construction fit release.

**Core decision: PASS for the exact gate/checker hashes below.** No actionable
numerical or artifact-boundary defect was found. The source allowlist interface
also passed the limited synthetic checks described below. The final generated
training manifest and construction release adapter were unavailable during this
bounded review and are **UNREVIEWED**, not accepted by this decision.

| Reviewed source | SHA256 |
| --- | --- |
| `gate.py` | `1813e1a43e4c6fea0ffcb3536e6f254460c22af86e66bab0c4514a335060df64` |
| `checker.py` | `88e67c63701d95b6ac798c79bb7033ed9e60a3caa585c3b614ffc2e99bc69d19` |
| `source_auth.py`, limited interface review | `8830e426fd0f279ec63ea916e1f9799c570b23e66667283396406f905b6a59d5` |

## Numerical and artifact findings

The fitter implements the fixed unweighted mean, per-row L2 normalization,
balanced weights, weighted centering, lambda 0.1, unpenalized intercept and
strict positive-score rule. The independent checker uses a separately derived
non-symmetric weighted dual system and Gaussian elimination, then checks primal
normal equations; it does not import or call the fitting solver. It also checks
discrete routes, so a score difference below `1e-10` does not excuse changing a
decision at zero.

The artifact now binds `construction_lock_sha256`, as required by the accepted
sequence amendment. It binds the training manifest, feature content and source
hashes as well. Loading verifies the externally supplied artifact hash,
canonical bytes, unique JSON keys, exact schema/method/native feature contract,
exact expected bindings, parameter shape and finite numeric values. Loaded
parameters are tuples in a frozen dataclass. The future caller must supply
externally locked expected hashes and bindings; self-derived expectations are
not an authentication boundary. No runtime adapter is accepted here.

## Observed synthetic verification

Two standalone PowerShell here-string scripts were piped to `python -B -` from
the workspace root; both exited 0. Their complete commands and outputs are in
this review task's tool transcript. They imported only standard-library modules
and the three reviewed source modules, without calling source extraction or
authentication against actual files.

The first script reported **PASS, 23 checks**:

- Dual/checker agreement plus a third, independently written augmented-primal
  Gaussian elimination oracle on 3x1, 3x2 and 9x2 synthetic datasets; the 9-row
  dataset was also padded to width 1,024 and checked.
- Byte-identical serialization after loading the 9x1024 synthetic artifact,
  exactly reproduced scores and rejection of field assignment.
- Rejection of artifact-byte tampering, wrong bindings, and rehashed wrong
  schema, method, native contract, parameter shape, noncanonical JSON and
  duplicate JSON keys.
- Rejection of NaN, infinity, wrong width, zero centered norm, boolean features,
  single-class labels and boolean labels.
- Exact zero scores route OFF; the checker rejects a `1e-12` intercept change
  that turns a mathematically tied training route ON.

The independently solvable fixture used raw `h=[3,2,0]`, labels `[+1,-1,-1]`.
The expected result is `w=10/17`, `b=-5/17`; the unavoidable conflicting-label
training miss was correctly retained. The other raw fixtures were
`[(3,0),(1,1),(0,-1)]` with the same labels, and
`[(3,1),(2,-1),(4,0.5),(1,-0.5),(-1,1),(-2,-1),(-3,0.3),(-4,-0.6),(0.5,2)]`
with four positive then five negative labels. The largest checker parameter
difference was `3.3306690738754696e-16`; the largest normal-equation residual
was `1.1102230246251565e-16`. The 9x1024 synthetic artifact SHA256 was
`a40e9d51d35f3709e4bdf1c10af04f562ba4e8661b03089e683b448229357def`.

The second script reported **PASS, 8 source-interface checks**: foreign source,
sealed path and edited-state path rejection before file reads; one valid
synthetic metadata join; and rejection of wrong phase, dtype, token identity
and supervised label. `build_manifest()` and `extract_features()` were never
called. These checks do not authenticate the real nine captures.

## Remaining review boundary

The source module uses an exact frozen B/O/H allowlist, matches working bytes
against their named commit blobs, joins inventory/input/capture metadata and
reserves finite float32-exact `h0==h` and zero-offset admission for authorized
extraction. Static inspection and synthetic path/join tests support that design;
the final manifest and actual source bytes still need root's separate acceptance.

One adapter integration requirement was sent to the implementation worker:
`extract_features()` presently invokes a manifest build with its own 60-second
deadline and then starts another local 60-second window. The construction
launcher must enforce one cumulative 60-second computation cap across
reauthentication, extraction, fitting, independent reconstruction and output,
plus only the separately allowed cleanup. An unfinished launcher is not being
reported as a defect. Its eventual enforcement, one-attempt behavior, default
denial, output reservations and frozen-code/source bindings remain unreviewed.

Real residual coordinate reads: 0. Real feature fits: 0. Sealed numerical
payload reads: 0. Tokenizer/model/provider/checkpoint-tensor/network/installation
calls: 0. No implementation or previous review file was edited.
