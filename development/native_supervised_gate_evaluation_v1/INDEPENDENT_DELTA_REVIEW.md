# Independent changed-interface review

2026-09-09. Read `ROOT_ENGINEERING_SCOPE.md` completely before review. This
bounded review covers only `learned_gate.py`, `workflow.py`, `audit_saved.py`,
`science.py`, `authority.py`, and `test_candidate.py`. No blocking finding
remains in the reviewed delta. This is a model-free engineering result, not
release approval or evidence that the frozen gate routes the real cohort well.

## Exact reviewed source hashes

| File | SHA256 |
| --- | --- |
| learned_gate.py | 97ecde35553535e3910e3754e375b0605ba4e336535b1271b1c4ddefdac670b8 |
| workflow.py | 5fb27dc215a3d2bc569462964f1226856131acd59eb0a8bbb8d90e5987bd9de3 |
| audit_saved.py | e44af5e58b38f166acfbdc2de795e4954852d1f76e298ccff235f057686eeaca |
| science.py | 1f383e6d38a4323eb2c81cd3c04c9eead55aeee2e298a574063c1b9d2f31cbc2 |
| authority.py | 002984d19bec3bbdd652fc0df653740d72036ea74aa7e8b50cc772d486bef2a4 |
| test_candidate.py | 7a538ec3662adcf3c01beea5393df5ea019d095161695cf7f9ed6025ef29ed02 |

The previously reported reload/dataclass-identity defect is fixed: the adapter
reauthenticates the frozen artifact and compares immutable parameter values.
The regression accepts the authenticated reloaded parameters and rejects changed
intercept values. Root separately reported independent rejection checks for
changes to each of mu, w, and b; that additional report is not counted as a test
run by this reviewer.

## Observed verification

Working directory for the following commands was this evaluation namespace;
shell was PowerShell with login disabled. Python was Python 3.12. The principal
command was `python -B -m unittest -v test_candidate`: **11 tests passed in
70.313 seconds; exit 0**.

Those tests exercised full synthetic success and saved judgment; complete
16-baseline wrong-route census with zero policy requests/derivatives; capture
exception and invalid capture; eligibility only after the complete census;
fresh entry mismatch; zero-score OFF; quality and derivative first failures;
saved gate-score and log-odds tampering with updated closure hashes; artifact,
source, and binding tampering; schedule/count/time/storage reservations; and
absence of model/provider imports. The successful synthetic path produced four
flips, four retentions, twenty-four OFF returns, sixty-four forwards, and four
derivatives. Saved ordinary accuracy preserved the deliberately wrong answer
across its baseline and both OFF requests.

Three additional `python -B -c` commands completed with exit 0, without writing
fixture files:

1. Constructed a synthetic Gate with zero mu, unit first-coordinate weight, and
   zero intercept. Both scorers returned exactly zero on a nonzero orthogonal
   feature. The independent scorer rejected all five invalid inputs: zero
   centered norm, NaN, infinity, Boolean feature, and width 1023.
2. Ran `test_candidate.run_fixture()` and independently modified the in-memory
   WORKER_RESULT, updating both its closure hash and inventory entry before
   each saved judgment. Fabricated census counts were rejected with
   `INDEPENDENT_COMPLETE_CENSUS`; a self request relabeled OFF was rejected with
   `SELF_ON_REQUEST_ONLY`; a fabricated endpoint verdict was rejected with
   `ENDPOINT_RESULT`. A separate `capture_failure` fixture was judged
   INCONCLUSIVE with three census outcomes, one FAILED cell, 116 UNRUN cells,
   and zero derivatives. The forbidden-import assertion also passed.
3. After the full suite, the final authority delta added a mandatory release
   reservation of exactly 139,837,440 bytes, computed from GROUP_CAPS plus 65,536
   bytes and checked against the total ceiling. Read the final source completely
   and executed its exact AST-extracted predicate: the exact reservation passed;
   one byte under, one byte over, and a value above the total ceiling were all
   rejected. Four cases passed; no release/admission function was called. The
   other five reviewed hashes remained unchanged. The full suite was not rerun
   for this isolated authority predicate.

All feature scoring in these checks used synthetic gate parameters and synthetic
features. The existing fitted artifact was loaded solely for permitted
authentication/parameter-integrity checks. No fitting, actual feature scoring,
tokenizer/model/provider import, checkpoint tensor read, historical numerical
payload read, experiment, commit, or release occurred. Full fake logit evidence
stayed in memory.

## Scoped source conclusions

The workflow completes every valid baseline before considering eligibility or
policy requests. Audit labels judge the independently measured route. Routing
errors block all policy requests and derivatives. Every request obtains a fresh
entry, recomputes the strict-positive gate, and requires baseline identity and
the same score/route. OFF returns its own entry. Technical failures stop further
execution; scheduled update skips and unrun suffixes remain distinct.

The editor predicates and update recipe are extracted from the authenticated
unchanged source. The reviewed flow retains current-gradient updates, original
h0 anchoring, real cold endpoints, acceptance-before-quality stopping, and
the stated step/path/net bounds. It adds neither a KL upper gate nor a natural
opportunity quota. Saved judgment reconstructs scores, routing, geometry,
requests, endpoints, and ordinary accuracy from evidence, and rejects invalid
evidence rather than interpreting it as OFF. Authority preserves explicit
release authentication and the stated 1/120/32, 1200+120+15-second, 192-MiB/5-MiB
ceilings.

Reused source pins inspected as source only: frozen gate implementation
`1813e1a43e4c6fea0ffcb3536e6f254460c22af86e66bab0c4514a335060df64`;
editor recipe/predicates
`67609809d32ea98ebca18b29c4e36f7272dedcee50c2c3e93f98468584d4c1a0`;
word scorer `2b84066607198c91d07aa71f6a38621da05eb2354f264d8051ba186ae05dc64b`;
independent word reference
`d1631112b26961deb209badf1afd07dddc9aba696bf02e6e85ad6bc77b2508fd`.

## Unclosed dependencies

Separate preparation/input integration review and complete finite-source freeze
closure remain outside this review. Actual content/overlap admission, final
source/text locks, offline preparation release, and subsequent model-evaluation
release remain prospective prerequisites. Real baseline eligibility, gate
classification, and endpoint success are untested. This report does not replace
those prerequisites or repeat the unchanged inherited infrastructure audit.
Root subsequently reported that its synthetic preparation-to-input bridge check
passed. That independent integration result is not a test run by this reviewer.
