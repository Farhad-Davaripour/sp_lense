# Locked judge transport

The stage-one comparison runner deliberately makes no hosted API calls. It renders
blinded, self-hashed JSONL request records. `submit_openai_judge_requests.py` is a
separate transport utility for submitting those exact records to the pinned
`gpt-4.1-mini-2025-04-14` snapshot and retaining exact API receipts.

Before any network call, the transport requires the exact persona/open exchange schema
and recomputes prompt, content, and configuration-metadata bindings, including persona
request IDs. Open request IDs depend on the unblinded source-generation record, so each
phase wrapper independently regenerates the complete open request file from its locked
raw generation input and requires byte-identical bytes before preflight. An edited
prompt, Structured Outputs schema, sampling field, tool field, content hash, or wrapper-
bound open ID is rejected. The verified request object is then posted unchanged to
`POST /v1/responses`.

## Credential setup

Run the masked-input helper from the repository root:

```powershell
.\artifacts\steering_comparison\set_openai_api_key.ps1
```

Do not put the key in a tracked file or paste it into chat. The three completion
wrappers read the current process environment first, fall back to the Windows user
environment, and copy the value only into their own process so the Python child can
inherit it. The Python transport independently uses the same Windows user-environment
fallback for safe direct/resumed operation. Neither layer logs the key.

## Mandatory cost preflight

Every invocation requires `--max-cost-usd`. Wrapper ceilings are phase-local: persona,
validation-open, and sealed-open each require a separate explicit authorization; one
phase's allowance is never automatically reused for another. `--dry-run` writes a
conservative preflight record without reading a key or making a network call:

```powershell
.\.venv\Scripts\python.exe `
  artifacts\steering_comparison\submit_openai_judge_requests.py `
  --requests artifacts\steering_comparison\qwen35_08b\persona_judge_requests.jsonl `
  --responses artifacts\steering_comparison\qwen35_08b\persona_judge_responses.jsonl `
  --work-dir artifacts\steering_comparison\qwen35_08b\persona_judge_transport `
  --max-cost-usd <USER_APPROVED_LIMIT> `
  --dry-run
```

Remove `--dry-run` only after reviewing the exact request count, request-file hash,
token upper bounds, current pinned prices and safe dollar upper bound. The command
refuses to run when the conservative bound exceeds the supplied ceiling.

The judge rates are protocol constants, not caller-adjustable estimates: every direct
or wrapper invocation must use exactly $0.40/M input tokens and $1.60/M output tokens.
Once any submission attempt, receipt, response shard, blocked response, or combined
response exists, `cost_preflight.json` is immutable. A resume must reproduce that file
byte-for-byte or it stops before any network call. The exact required bytes are UTF-8
without a BOM, keys sorted lexicographically, two-space indentation, Unicode unescaped,
and one final LF. Semantic JSON equivalence, different whitespace, CRLF, or reordered
keys is not accepted.

The transport writes each exact attachable response atomically, retains the complete
provider response and usage record in a separate receipt, and resumes without paying
again for completed requests. If a crash occurs after the fully validated receipt is
durable but before its shard or completed-attempt marker, resume revalidates the complete
provider/model/usage/output/trace binding, reconstructs the shard when needed, and
atomically promotes only the safe `prepared` attempt state to `completed`.
`attach-judgments` remains the authoritative strict schema/provenance gate.

## Ambiguous network outcomes

`X-Client-Request-Id` is recorded for provider tracing; it is not treated as an
idempotency guarantee. The transport retries only an explicit HTTP 429 rejection. It
does not automatically repeat a timeout, connection loss, HTTP 408, HTTP 5xx, malformed
success body, or post-response validation failure. Instead it writes a hashed
`submission_attempts` record and blocks resubmission until the recorded provider trace
has been resolved. This prevents a lost response from silently becoming a duplicate
charge or a second judgment.

A syntactically successful provider response that lacks the pinned model/status,
response ID, server trace, usage, or output text is preserved under
`blocked_responses/` and marked `response_validation_blocked`; it cannot be silently
discarded or resubmitted. Completed verification rejects any unresolved quarantine.

Each paid transport also holds an OS-level exclusive lock for its work directory across
the complete check/POST/receipt loop. A concurrent launcher fails before sending a
request. The `.submission.lock` file is an uncommitted synchronization primitive; the B,
D, and final freeze scripts explicitly exclude it from research commits.

After a phase completes, `--verify-only` makes no network call and requires exact set
equality across requests, combined responses, shards, receipts, and attempt records. It
also verifies hashes, the pinned model/status, token usage, and unique provider trace
IDs. Each completion wrapper runs this audit immediately after its paid batch, before
attaching judgments or fitting/building downstream artifacts. The B and D freeze gates
repeat the relevant audit, and `freeze_final_results.ps1` rechecks both persona
transports plus every validation/sealed open transport before committing final results.

## Official API contract

The transport was checked on 2026-08-24 against the official
[Responses create reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
and the official
[GPT-4.1 mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Those pages document the `/v1/responses` endpoint, `text.format` Structured Outputs,
response status/usage fields, the dated `gpt-4.1-mini-2025-04-14` snapshot, and the
locked $0.40/M input and $1.60/M output prices.
