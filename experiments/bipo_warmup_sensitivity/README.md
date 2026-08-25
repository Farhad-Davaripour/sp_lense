# BiPO warmup-fraction sensitivity

This directory is a locked, outcome-blind **secondary sensitivity study**. It does not
replace or amend the confirmatory BiPO construction. The confirmatory vectors retain
their locked 100-step warmup and remain the only BiPO vectors eligible for the main
winner ranking.

The sensitivity changes one value only: BiPO warmup is 11 optimizer steps instead of
100. With 64 discovery pairs, effective batch size 4, and 20 epochs, the laptop study
has 320 optimizer updates. Eleven updates is 3.4375% of the run, the nearest possible
integer-step approximation to the published-run fraction of 100 / 3,040 = 3.2895%.

## Safe commands

Verify hashes and invariants without loading a model:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\run_sensitivity.py verify
```

Print the four locked construction plans without loading a model:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\run_sensitivity.py plan
```

Run one construction only after the experiment lock is committed and all confirmatory
BiPO construction jobs have finished:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\run_sensitivity.py construct --model-tag qwen35_08b --track matched
```

The equivalent PowerShell wrapper is:

```powershell
experiments\bipo_warmup_sensitivity\scripts\run.ps1 -Action Verify
```

No command here is called by the main steering-comparison automation. Construction
outputs go under this directory's ignored `outputs/` tree. The runner refuses any
other output location, refuses overwrite, uses a distinct method ID, and verifies that
all existing confirmatory BiPO files are byte-identical before and after construction.

## Post-confirmatory evaluation

Evaluation is separately gated and is impossible until the main final-result commit is
the pushed `origin/main` tip. The gate verifies the exact final commit subject, final
artifact inventory, final report, Stage-2 manifest, canonical sealed plan, and hashes of
every parent BiPO forced/scored result. It then creates an immutable secondary plan under
the ignored `evaluation_outputs/` tree. No preparation command loads a model or reads a
sensitivity outcome.

After all four warmup-11 directions exist and the main final result is pushed, prepare
the exact parent-cohort mirror:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py prepare-evaluation
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py verify-prepared
```

The plan lists one secondary setup for **every** frozen parent BiPO setup, retaining
distinct fixed/equal-efficacy cohorts rather than collapsing them. Run both model passes
for each listed setup ID:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py evaluate-forced --setup-id <SETUP_ID>
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py generate-open --setup-id <SETUP_ID>
```

Forced evaluation requires exact 1,350-row identity and open generation requires exact
96-row identity against the corresponding frozen parent setup. Case, target/role/form,
prompt, condition, and baseline-content identities must match. Strength, layer, geometry,
sealed IDs, TBSP inclusion, and all safety/KL limits are inherited without recalibration.

Render the blinded judge exchange after every open-generation shard exists:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py judge-requests
```

This command makes no external call. Submit those requests using the separately audited
transport, place exact response records inside `evaluation_outputs/`, then attach them:

```powershell
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py attach-judgments --responses experiments\bipo_warmup_sensitivity\evaluation_outputs\judge\open_judge_responses.jsonl
.venv\Scripts\python.exe experiments\bipo_warmup_sensitivity\scripts\evaluate_sensitivity.py report
```

The final command emits `secondary_sensitivity_report.json`,
`SECONDARY_SENSITIVITY_REPORT.md`, and `secondary_sensitivity_manifest.json`, with hashes
for directions, construction manifests, parent results, secondary results, plan, and
freeze receipt. It reports effect, actual decision changes, collateral metrics, KL,
coherence, TBSP, robustness strata, and paired warmup-11 minus warmup-100 differences.
It never imports or invokes the main report builder and cannot update the main ranking.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest experiments\bipo_warmup_sensitivity\tests -q
.venv\Scripts\python.exe -m ruff check experiments\bipo_warmup_sensitivity
```

These tests and scripts do not run either Qwen model and do not read behavioral result
rows. See `PROTOCOL.md` for the preregistered comparison and claim boundaries.
