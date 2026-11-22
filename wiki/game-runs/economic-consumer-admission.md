# Future economic Game consumer

`src/economic_game_consumer.py` prepares a separate future consumer. It does not
start a model, pick study seeds, qualify a service, issue a window, change an
active Game, or initialize the historical private context. Missing or HELD
admission metadata fails before health reads. Existing user authorization is
bound by `pins.expected_direct_authority_sha256`; this contract adds no human
approval step.

After the actual new checkpoint is complete, Root supplies an authenticated,
hash-pinned record with schema
`research4.replica-identity-economic-root-admission.v1`. Its exact keys are:

```text
schema, status, issuer, issued_by_root, owner_chat, issued_utc,
actual_artifact_binding, source_binding, core_binding, pins,
main_plan_ref, source_packet_ref, excluded_seeds, reviews, transport
```

`status` is `ADMITTED_NEW_ECONOMIC_GAME`, `issuer` is `Root`,
`issued_by_root` is true, and `owner_chat` equals the new helper's `GAME_CHAT`.
`actual_artifact_binding` is the independently audited candidate accepted by
`economic_identity_registry`: actual 612 updates, 42 DEV rows, the bound
306-decision/two-epoch data recipe, and Last8/r16/F32 adapter. Synthetic test
metadata is not a qualified research artifact.

`source_binding` has exactly `helper`, `parser`, and `expected_ft_id`.
`helper` and `parser` are Root-relative `{path, sha256}` references. Root must
provide new checkpoint-specific source, not the previous Game's helper/parser.
The helper exports the existing native two-route validation and completion API,
`require_new_static_binding()`, and these bound constants:

```text
ECONOMIC_ADMISSION_SCHEMA = research4.replica-identity-economic-root-admission.v1
BASE = qwen3.8-27b-base
FT, ARMS, FINAL_ADAPTER_SHA, FINAL_CONFIG_SHA, BASE_SHA,
READY_ARTIFACT_REF, COMPLETION_AUDIT_SHA, MAIN_12CASE_PLAN_REF, GAME_CHAT
ACTUAL_FIT_UPDATES = 612
ACTUAL_TEACHER_DEV_ROWS = 42
MAX_GAME_REQUESTS = 3072
MAX_GAME_WINDOW_SECONDS = 90000
```

The parser exports `_require_static_route_binding()`, `parse_request(raw)`,
`LAST`, `ARMS`, `READY_ARTIFACT_REF`, and `ECONOMIC_INPUT_PROTOCOL` equal to
`economic_game_input.PROTOCOL`. The helper's `MAIN_12CASE_PLAN_REF.sha256`
must bind the new Main plan. Root supplies a new qualified Core, independently
trusted Core review, index and finite window. The record's `core_binding` and
`pins` exactly match the private handoff and the window's plan, nonce, direct
authority and Core. Its window allows cumulatively at most 3072 dispatches over
at most 90000 seconds. No reader or collector renews that window or quota.

`main_plan_ref`, `source_packet_ref`, and `reviews.Main_source_ref` are
Main-relative references beneath `.local/`, using forward slashes. Absolute
paths, drive colons, backslashes, traversal, repeated separators and linked
paths are rejected. `reviews.Root_transport_ref` is Root-relative. Both review
records must be pinned JSON objects with a status beginning `PASS`.

`transport` is exactly:

```json
{"max_game_requests":3072,"max_window_seconds":90000,"imported_cases":0,"quota_reset":false}
```

`excluded_seeds` is an explicit unique list of all prior or otherwise excluded
positive seeds. Once training is complete, the caller selects three fresh
worlds outside this set. The new plan contains twelve cells across the four
future role IDs, identical generation settings, paired worlds and system
conditions. It has schema
`research4.replica-identity-economic-four-condition-48h.held.v1` and retains
`execution_enabled:false`; Root's separate admission supplies execution
authority. Historical transport parents, imported cases and completed-prefix
references are rejected.

The plan's `configuration_path` is Main-relative and its bytes match
`configuration_sha256`. That JSON config has exactly `schema` equal to
`research4.replica-identity-economic-game-configuration.v1` and
`prospective_protocol` equal to the plan's pinned Main-relative protocol
reference. The protocol may be human-readable YAML. The source packet has
exactly `schema` equal to
`research4.replica-identity-economic-source-packet.v1`,
`interface_runtime_source_sha256` equal to the current comparison runtime map,
and `export_source_sha256` equal to the current map for both publishers.

The private context has exactly `pins`, `requester_chat`, `core_binding`,
`source_scope`, `source_binding`, and `admission_ref`, carried solely in
`RESEARCH4_GAME_ECONOMIC_CONTEXT`. `admission_ref` is Root-relative.
`source_scope.plan_path` is the resolved absolute internal path corresponding
to the admitted Main-relative plan reference; `batch_index` is an integer
from 0 through 11. A fresh child reconstructs only the new registry from all
twelve frozen cell candidate bindings. The consumer's claim and STOP files
live under `.local/economic_grants/<admission-sha256>/`.

The native Security500/Continuity implementation retains exact route/profile
checks, zero retries, one in-flight request, output/context limits,
case/global/request deadlines, and same-request native ACK/EOS/KV-empty/BASE
restore checks. Economic wire validation additionally checks its frozen public
state and summary. Each actual wire's integer `observation.decision_count`
must equal the current native HTTP ordinal before dispatch. The raw
`HTTP_ATTEMPT` and `POST_REQUEST_VALIDATED` records carry
`generation_identity:{case_id,decision,attempt:1}`, bound to their exact
`request_id` and serialized-messages SHA. The outer `attempt` remains the
native HTTP ordinal. Native reasoning can therefore join a decision by
explicit identity rather than timing or row order.

Preparation tests use only temporary/in-memory synthetic metadata and synthetic
worlds. They exercise true HELD/ADMITTED construction and the inherited native
wire checks; they do not qualify an actual Root helper, Core, checkpoint,
hardware, throughput or scientific result. Root's post-training source and
native qualification, the complete exclusion list, fresh world selection,
reviews, and finite authenticated admission remain required before a run.
