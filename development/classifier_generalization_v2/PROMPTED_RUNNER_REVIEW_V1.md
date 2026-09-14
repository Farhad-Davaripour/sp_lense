# PROMPTED RUNNER REVIEW V1

Job ID: `prompted_runner_review_20260914_1455`
Verdict: **PASS_SCOPED** (no blocking defect)

## Pins verified (read-only)
- `prompted_span_runner_v1.py` = fa94c39a…d9fcd65e1 (match)
- `test_prompted_span_runner_v1.py` = a5770815…c7466955e (match)
- diff `span_development_runner_v1.py` → prompted = 38 insertions, 11 deletions only.

## Diff scope (as specified)
Distinct SCHEMA/INDEX/RECEIPT/BINARY schemas, `JOB_ID`, `INPUT_ROLES` +`prompt_sanity`, `SOURCE_NAMES` swap to `prompted_input_adapter_v1.py`, new sanity block, prepare `max_tokens=320`, capture/index/receipt query+condition+context pins. Generic numeric window layout, `CAPS`, `_float32_le`, `_validate_window`, `_validate_boundary`, worker/supervise and 320-guard unchanged.

## Independently reproduced (synthetic fixtures only; 13 tests PASS)
- Private-path inputs rejected (`INPUT_SCOPE`), including embedded component; no real provider package imported.
- Old source/caps unchanged (644→671 lines; 640/320/1800 s/125829120/201326592 B identical). Old refs validated read-only 320 metadata; 321 rejected.
- Query truthful: `index.query == FIXED_QUERY`, query/condition/original-context SHA pins on index, every record and receipt; `last_shared=198`, prefix inside query; prefix/hash guards intact.
- One build, 640 forwards (320 cases × AB/BA), 1 model load, 1800 s, ≤192 MiB output; parent re-verifies all pins before `supervisor_success.json`; marker cleanup on failure.
- No framework/network/install/Git/subagent use; no repo artifacts written.

## Non-blocking observations
1. `RealToyChildTests` still spawns `span_development_runner_v1`, so the new child `main`/preflight path is not exercised end-to-end (parent side is). Bounded test-hygiene gap.
2. Test sanity pins `max_full_tokens=100` vs measured 196 — stricter than the enforced 320, so the true bound is not exercised.
