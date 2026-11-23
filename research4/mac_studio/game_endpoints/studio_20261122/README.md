# Studio Game endpoint for November 22, 2026

This source serves one arm at a time on the 36 GiB Mac Studio. The available
arms are the pinned Qwen3.8-27B 4-bit base and the published rank-16,
64-block adapter at epoch 3. The endpoint uses the trainer's shared model
lease and binds to loopback.

`studio_all64_game_readiness.py` verifies the checkpoint and base manifests,
checks byte-identical native base logits after adapter bypass, checks the
adapted route after a round trip, and runs a one-token synthetic inference
per arm. `studio_all64_context.py` extends a synthetic prefix with the native
512-token prefill path, records live MLX allocation, model cache bytes and
available host memory, and measures one decode token per arm at the final
safe prefix. Keep the qualification and release receipts under ignored
local storage.

`studio_all64_endpoint.py` accepts a reviewed local JSON specification with
these keys:

| Key | Required value |
| --- | --- |
| `schema` | `research4.studio-all64-game-endpoint.v1` |
| `model_id` | `qwen3.8-27b-base` or `sp-20261120-09-be11cf56` |
| `context_tokens` | Frozen context at or below the measured common bound |
| `port` | Verified free loopback port |
| `plan_sha256s` | Exact distinct authorized Game plan hashes |
| `token_sha256` | SHA-256 of the separately held private bearer token |
| `context_readiness_sha256` | SHA-256 of the saved context-readiness JSON |
| `worker_deadline_seconds` | Approved bounded arm window, at most 14,400 seconds |

Run with the installed merged-project interpreter and a new run ID.
`--preflight` checks the inputs, lease and port. `--serve` starts the reviewed
endpoint for its authorized window. Both require the exact `--spec`,
`--spec-sha256`, `--source-sha256`, and `--run-id`.

`GET /health` exposes model, quantization, source, and context identities.
`POST /v1/chat/completions` verifies the private bearer token, plan and
decision headers, frozen generation settings, and full remaining native
context allowance. Context accounting uses the pinned Qwen tokenizer.
Requests are serialized, and each request initializes a fresh native
cache. The chosen context and history compaction settings belong in the
Game plan and run manifest alongside the endpoint and context receipt
hashes.

The supervisor enforces the Studio memory floor and time limits. An
operator can create a STOP file in the endpoint's ignored run directory.
Retain the generated and release receipts for completed and interrupted
runs under `.local/`. Access tokens and authority records stay private.
