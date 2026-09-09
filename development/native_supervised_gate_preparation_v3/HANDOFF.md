# Construction preparation v3 handoff

Model-free preparation candidate complete. Final source manifest SHA-256:
`791f62010a74f75df78f907184ba92fc39345e30fcd3663a515ca067a7b092f5`.

The focused final suite passed six test groups and five artificial engine runs.
Two complete synthetic runs cover 32-token and exact-320-token inputs. Each
completed all 417 operations: one artificial factory invocation, 160 renders,
160 encodes and 96 decodes. The other runs reject oversize input, a mask fault
in row 32 after retaining 31 completed receipts, and expiry of the 350-second
worker cap. All successful reader checks cover 834 paired journal entries.
The tests also reject tail gold/header/suffix/journal changes, old namespaces,
missing or invalid submission identity, altered source joins, malformed cohort
counts and duplicate pair wording. Shared capture/owner source binding and the
output partition including terminal reserve are tested with artificial files.
The final closure admission also requires owner PASS, finite bounded timings,
no primary/cleanup errors, assignment before resume and actual authentication,
successful preparation, retained successful exits, clean joined drains, empty
job, closed pipes and the exact owner source identity. Eleven pure negative
closure mutations reject quiescent-but-failed receipts, including timeout,
cleanup failure, late completion, failed launcher exit and drain overflow.

Final evidence directory: `fixtures/synthetic_1788967818070537400/`.
Its TEST_RESULTS.json SHA-256 is
`becdab9da3d3c0a03a68bf9dee082e450a6cc5129f514a9447956205cac8683b`.
Its reusable `bundle/SYNTHETIC_RELEASE.json` SHA-256 is
`9dba377d50a4bdec66c4cdf1e045e553a57728d04e1e4084c0d7cf1d7c28568c`.
This synthetic bridge is bound to the final preparation source hash and a
deliberately artificial capture source hash, `f` repeated 64 times. It must never
be promoted to actual input. The earlier retained synthetic directory
`synthetic_1788967283131684800` and `synthetic_1788967373559055800` record intermediate sources and are superseded;
use only the final bridge for integration checks.

Public interfaces are `prepare_reader.read_bundle(base, release, synthetic=False)`,
`prepare_reader.execution_binding(capture_sha)`, and `schema_cases()` /
`schema_cases32()`. Native input schema and neutral renderer body are unchanged.
Source deltas are finite row/count/time/storage/namespace and prospective cohort
admission joins. Semantic renderer functions are AST-identical to authenticated
v2; complete per-input token proof logic is unchanged. dependencies.py,
TOKENIZER_PINS.json and the inherited artificial tokenizer definitions are reused.

Actual release prerequisites remain pending: independently admit the first
complete authored batch; supply its exact raw admitted_submission_sha256 in the
text lock and each actual preparation/capture release; freeze the lock against
the final capture and preparation source manifests; admit source and machine
identity; complete the capture owner's storage reservation and provide separate
committed root releases. Preparation owner and capture share the capture source.

Every actual authorization remains false. No actual TRAINING_SUBMISSION.json,
AUTHOR_ATTESTATION output or clean author's brief was read. No tokenizer or model
was imported or invoked, no tensor assets were opened, and no fit, launch,
installation, network request or commit was made. The only new author interface
read was EMPTY_SCHEMA.json. Readiness remains 70%.

At final source verification, all retained namespace files totaled 2,344,964 bytes
before this handoff update; the largest file was 90,888 bytes. Both are below the fixed
32 MiB total and 5 MiB/file ceilings. Handoff and generated evidence are excluded
from SOURCE_FREEZE.json so they cannot create a source/fixture hash cycle.
