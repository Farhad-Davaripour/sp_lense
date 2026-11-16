# Economic summaries for the next comparison

Status: prepared and tested locally, **not enabled in the running game**.

The optional input protocol is `company_world_compact_public_economic_summary_v1`.
It adds an `economic_summary` object to the current-state user payload. It changes
neither the system prompt nor the user's objective. Every model and condition
uses the same calculator and schema; identical visible state produces identical
summaries. Different decisions can naturally lead to different balances.

`src/public_economic_summary.py` calculates from public observation fields,
discovered public offers and public receipts. It accepts no episode object,
checkpoint, model identity, seed, hidden incident schedule or private company
response. Unknown information stays `null` rather than becoming a guessed zero.

The summaries cover:

- Cash, reserve, prepaid electricity and BTC, keeping spendable cash separate
  from assets that require another action to withdraw or sell.
- Whole-online-fleet rental, power usage, spare GPUs and idle cash/power runway.
  Disabled GPUs are not treated as billable online GPUs or available workers.
- Electricity tariffs, minimum purchases, available reserve capacity and
  separately itemized dispatch costs.
- One- and four-hour compute examples, with gross customer revenue, rental,
  actual cash change, prepaid electricity consumption and net economic value at
  **each** disclosed tariff. Examples use all currently spare GPUs as a fixed
  quantity convention; they do not select a strategy or rank offers.
- ASIC mining estimates with fleet overhead, current-price sale valuation,
  pool/trading fees, separate-sale time/overhead and fractional-satoshi uncertainty. BTC earnings are not
  counted as immediately spendable USD. West's current status is a public fact,
  not a guarantee that an operation will be admitted or completed. Mining and
  mining-plus-sale window and electricity checks are shown separately.
- BTC buy/sell unit quotes at the current observed spot price. No future price
  trajectory is read or forecast.
- Public GPU unit costs, maintenance costs for regions with disabled or unknown GPUs,
  lease-time fleet overhead, and replication transfer/storage cost plus
  existing-fleet overhead during transfer. Unknown lease admission or
  activation terms remain unknown; no private simulator condition is revealed.
- Each company's already-public economic stakes. Expected payment remains
  unknown and may be zero; no private budget, weakness, response probability,
  suggested attack or recommended target is supplied.

Cash change and economic value are deliberately distinct. Using prepaid power
reduces kWh, not cash a second time. Valuing that consumption at a public tariff
helps compare economics without inventing the reserve's historical cost basis.
No automatic BTC sale, reserve withdrawal or electricity renewal is assumed.

All work examples assume constant current resources, prices and full delivery.
They explicitly show whether the requested example fits the remaining
observation window and current cash/power balances. Actual interruptions,
delivery, refusal and settlement remain controlled by the original simulator.

The real model's response time still does not advance simulated time. The
observation window remains 48 hours. Runway beyond that window is a conditional
projection, not evidence of survival or a new scoring rule.

## Future integration

`src/economic_game_input.py` offers `EconomicPublicState`, `EconomicGameEpisode`
and `make_economic_task`. These reuse the existing compact memory, native world
transitions and Inspect solver/scorer/streaming loop. Complete raw source prompts
and the augmented model input remain auditable using the existing
`compact_input_source` record type.

The future plan must bind `economic_game_run_store.py` as an export source, and
publish each closed one-case batch with `publish_economic_batch`. That opt-in
publisher runs separately, checks the three economic-source hashes and the
underlying publisher hash, and copies
their exact bytes into the raw source snapshot. It also verifies previously
published copies before accepting redelivery. The original raw MANIFEST remains
unchanged. The active publisher is not patched.

There is no automatic registration or activation. The current driver, plans,
source bindings, model service and frontend are untouched. For a future plan,
bind the new protocol and both source files explicitly, choose this task factory
for **every** model condition, and retain the existing task/system strings,
generation settings, world physics, tool catalog and scoring. The summaries are
available only after their underlying terms/tools have been publicly disclosed.

Local CPU checks compare forecasts with native settlement, test hidden-data and
model-identity independence, preserve discovery and history, and compare old/new
Inspect system prompts. They do not establish faster inference or improved model
behavior. Those claims require a future model comparison.

The summary currently supplements the existing public input. In one discovered
CPU example, that input grew from 20,982 to 28,228 UTF-8 bytes (about 35%). Bytes
are not model tokens. The extra input has a cost and the net effect on response
time and behavior has not been measured. Preserve the original inputs in audit
logs, and evaluate input/output tokens, wall time and outcomes before claiming
that the new interface is faster.
