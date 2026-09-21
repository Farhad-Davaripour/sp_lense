# Jev shutdown-gate comparison

Compare TypeSafe's hosted Jev 1.13.0 text classifier with the frozen XGBoost
activation-feature classifier. This is a different input representation and an
external service dependency, not a retrained version of the same classifier.

Status: **awaiting TypeSafe account/API access; no Jev predictions yet**.
The prior LoRA and activation-transfer results remain unchanged.

The frozen protocol is in `plan.json`: one binary question, scenario context only,
240 training cases for threshold selection, then 80 validation and 192 reused
diagnostic holdout cases. Labels, case IDs, answer options, and teacher scores are
excluded from API inputs. One score per scenario applies to both answer orders.
There is no prompt search or holdout tuning. The API version is pinned.

Report precision, recall, F1 and SELF/OTHER/control breakdowns, then replay the
new gate against saved base/teacher scores with unchanged output guards. Preserve
the original eligibility cohort for comparisons and separately report conversion
among all true shutdown views initially choosing KEEP. Improved gate recall must
not conceal precision loss or additional control changes.

No GPU required. Maximum 512 requests, 30 minutes, estimated $0.05 input cost;
no retries, purchases, or automatic top-ups. Store the key only in the local
`TYPESAFE_API_KEY` environment variable; `.env.example` documents its name.
From an installed checkout:

```sh
python -m sp_lense.research2.jev_gate run work/jev_gate
python -m sp_lense.research2.jev_gate report work/jev_gate
```

The first command requires official API access. Partial results are retained on
failure; do not rerun into the same directory or silently repeat charged calls.
The second command uses saved responses and makes no API calls. Detector scores
are scenario-level; downstream decisions have two correlated views per scenario.
The composed comparison is cached-score replay, not a fresh live deployment test.
Prefect remains unavailable due to the previously recorded Windows Application
Control block; `comparison.json` can be published when that server is restored.

Official references: [API contract](https://docs.typesafe.ai/api),
[model/version and pricing](https://docs.typesafe.ai/models).
