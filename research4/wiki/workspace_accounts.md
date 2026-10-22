# Work account and free tier

Both explicitly authorized accounts use the same research and operator code.
Public account identities are in `config/workspace_accounts.json`; credentials
are in the ignored repository-root `.env` shown by `.env.example`. Preserve both
credentials when switching. Never include `.env` in source releases, AIR code
snapshots, manifests, logs or commits.

From `research4/`, inspect or select the active account:

```powershell
python -m src.workspace_accounts status
python -m src.workspace_accounts switch work
python -m src.workspace_accounts switch "free tier"
```

`src.operator.cli` builds a fresh child environment from the selected account,
pairing its authorized host with its own token. An explicit `account=` argument
supports source reads during migration. An inherited generic token cannot cross
into the other account. The bundle intentionally has no fixed host: the shared
operator supplies the selected `DATABRICKS_HOST` for strict validation/deployment.
Model protocol workspace fields remain historical origin metadata;
`inputs/WORKSPACE.json` freezes the execution account for each new run, and
worker receipts record the verified actual workspace.

Account switching changes authentication only. Deployment IDs, submission ledgers,
active/pending runs and watcher ownership must be reconciled separately before
any submission. Run IDs from one workspace must never be polled against the
other host. Preserve account-specific receipts and configurations, and keep the
old workspace/artifacts intact when copying.

The user authorized moving Research 4 to the work account and continuing the
bounded campaign. This does not authorize purchases, quota increases, changes to
one-A10/timeout/retry bounds, modifying Research 3 or opening held-out content.
The held-out seal may be copied only as opaque bytes with its original checksum;
never parse its seed or materialize cases before the existing admission gate.
