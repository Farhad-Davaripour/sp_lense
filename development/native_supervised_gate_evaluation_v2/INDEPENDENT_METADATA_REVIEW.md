# Independent v2 metadata review

2026-09-09. PASS for this bounded metadata-only successor review. No blocking
finding. Read HANDOFF.md, DELTA_MAP.json, TEST_RESULTS.json, and the closed v1
ROOT_PREPARATION_FAILURE.md before verification. No actual release or launch is
approved by this report.

## Final reviewed hashes

| Artifact | SHA256 |
| --- | --- |
| evaluation SOURCE_FREEZE.json | 2af19edd4f231242d8900dbdd20ce0650e70d39b20aaeb33fd7e1e685475f3df |
| preparation SOURCE_FREEZE.json | 2fd44e118374b465bd47bd344171e58d9ccb547a70ff6cbeac57fbe1addc732b |
| TEXT_LOCK_DRAFT.json | a828236b49e3f7649bb7d9bc72491876f4c421aab1266c04f5a857824d7edbfc |
| DELTA_MAP.json | 5694fc1e3adffb7d0de0a683fe59aebce9f8ebfb464a6d86319154e3e45d3092 |
| test_metadata.py | 9d9619622fe88c37e400ba1e9983ccb25e194589e24f3f6afdf1c362656f784d |
| test_admission.py | efaafff082ab0e7d81c02aee24b45d20e8fe8269b2263db35fed68edc1df9a22 |
| PREPARATION_OWNED_IDENTITY.json | 980f09b118294aa15d99335a098d4a843b132018eeb1114b0eddf06c0c49c85b |

## Observed independent verification

Commands used PowerShell with login disabled. From the repository root:

- `python -E -S -B development/native_supervised_gate_evaluation_v2/test_metadata.py`:
  five tests passed in 0.096 seconds, exit 0.
- `python -E -S -B development/native_supervised_gate_evaluation_v2/test_admission.py`:
  three tests passed in 0.029 seconds, exit 0. Its successful owner result was
  produced by the explicitly mocked runner; no child process was launched.

A separate `python -E -S -B -c` read-only script from the preparation-v2
namespace passed, exit 0. It compared all fourteen inherited preparation source
files against the authenticated v1 manifest. Every byte matched after only the
two permitted namespace substitutions; only prepare_core.py and
prepare_reader.py actually differed. It authenticated both prospective draft
source joins, observed `BLIND_REVIEW_THEN_TEXT_LOCK` from normal text-lock
validation, checked sixteen exact immutable blind-review hashes in order,
confirmed 209 preparation operations, checked absence of actual release/attempt
directories, and asserted absence of provider/model/tokenizer modules.

The five metadata tests independently established that all nineteen declared
scientific/runtime files are byte-identical to v1. They also verified the small
remaining runtime delta exactly: unique ATTEMPT in support.py; preparation path
and manifest hash in input_reader.py; the two namespace literals in
preparation_owner.py; and console_sha256 alone in the console-aware identity
configuration. test_admission.py differs only in its expected namespace. These
comparisons preserve live job, handle, parent, creation-time, image, and exact
hash checks.

The actual evaluation adapter authenticated and consumed the artificial saved
bundle, yielding sixteen cases and the 209-operation result. It rejected a
tampered inputs pin, a tampered preparation-manifest pin, and a wrong-origin
renderer module. The tracked bundle is explicitly synthetic; its template pin
was patched solely for the synthetic call. The preparation test that creates
its fixture was inspected but not rerun, avoiding fixture writes or another
preparation run.

## Scope and scientific continuity

The closed v1 event remains a technical owner-admission failure caused by the
stale console fingerprint, before the preparation directory/output existed. It
is not a gate-classification or editor outcome. The current console file bytes
were independently rehashed by the metadata test to
`e449bce01f275cd08f3d4e64bb73b3b43ae845a0dbdb3e6131426e66537705e5`,
size 1,003,520. Its recorded Valid Microsoft Windows signature and config join
passed; root and the implementation worker separately checked the live
Authenticode signature. This review did not rerun Authenticode. Strict identity
checks remain enabled; only the stale saved fingerprint changed.

The cohort remains the first v1 submission
`9b06eda1f293acd764a65cb742d9f2003ab980271065a927a49a8dbc4cb4902e`,
and the fitted gate remains
`fab7d797f9424d80aa3873eefc1cedf0e438e5b4ddc3a1059a080114aff7cca1`.
All sixteen admitted texts, gold semantics, model/revision, thresholds, strict
positive gate, complete sixteen-baseline census, conditional requests, fresh
entry identities, and editor geometry retain their authenticated earlier
implementations. The 209-operation preparation schedule, 120 forwards,
32 derivatives, one load, 139,837,440-byte reservation, and time/storage limits
are unchanged. Earlier independent eleven-group scientific verification and the
separately accepted final bridge remain reusable evidence. No broad scientific
suite was repeated or attributed to a new test-file hash.

The draft binds preparation-v2 and evaluation-v2 source manifests above. Its
approved/actual authorization flags remain false and final_text_locked is false;
normal validation correctly refuses it. Actual release and attempt directories
are absent in both new namespaces. Inspection found only the explicitly
artificial fixture bundles, with no copied v1 actual attempt/release/terminal
evidence.

## Remaining requirements

Root retains final source/text acceptance and locking, current-machine admission,
commit, a separate actual preparation release and owned closure, and any later
model release. Actual token boundaries/lengths, real gate routing, baseline
eligibility, and editor endpoints remain unmeasured in v2. No tokenizer/model
provider import, checkpoint tensor read, fitting, actual feature scoring,
network request, real owned-child launch, commit, or release occurred here.
Only this independent review document was written.
