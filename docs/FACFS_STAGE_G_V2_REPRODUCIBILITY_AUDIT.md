# FACFS Stage-G v2 reproducibility audit

## Audit scope

This package treats the completed Stage-G v2 result as a negative finding. It does not
run a model, resume a capture, generate new scientific outcomes, or alter the locked
v1/v2 artifacts. The verifier
[`scripts/facfs_stage_g_v2_publication_audit.py`](../scripts/facfs_stage_g_v2_publication_audit.py)
checks the following committed chain:

- the immutable v1 `attempt_0001` technical failure and its required no-resume state;
- v2 lock identity, every locked input hash, collision gate, old-outcome absence, and
  source-disjointness manifest;
- zero-model preflight, locked WSL environment provenance, and Smart App Control `On`;
- every one of the 2,926 hash-chained compute reservations;
- all 2,870 inventory-listed v2 output files, their sizes, and SHA-256 hashes; and
- the complete-attempt receipt, exact compute counts, 0/11 no-go decision, certificate
  health, zero generated tokens, zero finite interventions, and diagnostic-only status
  of the Walsh decomposition.

## Verified capture facts

| Item | Verified value |
|---|---:|
| v1 state | `failed_consumed_no_resume_no_retry` |
| v2 state | `complete` |
| v2 complete scenarios | 0 / 11 |
| v2 objective count | 1,430 |
| v2 scored sequences | 1,452 |
| v2 forwards / backwards | 1,474 / 1,452 |
| v2 ledger events | 2,926 |
| v2 output files | 2,870 |
| Generated tokens | 0 |
| Finite intervention calls | 0 |
| Effect certificates numerically valid | 1,430 / 1,430 |
| Alignment certificates numerically valid | 22 / 22 |

The expected successful audit has the final ledger event
`8f63bfe7a5dc48091c4dda94011a65a78d9063160a4b804d3082ad79fd0e3057`, output
inventory identity `9d1119dca7c9c1edfefe9740cca33107a7572079f144bcd152644e4e50226926`,
and summary identity `b970d8bbe50f2547e65dca509371adbe19072f1775a61675b050ba9b264167ed`.

## Reviewer command

```powershell
wsl.exe -d Ubuntu -u farhad --exec bash -c 'cd /mnt/c/Users/farha/repos/sp_lense && /home/farhad/sp_lense/.venv/bin/python scripts/facfs_stage_g_v2_publication_audit.py --check-git --verify-live-environment'
```

This command requires the authoritative experiment repository on branch
`codex/facfs-stage-g-v2-hook-compatibility`, a clean exactly-pushed worktree, and the
locked local WSL environment. It is read-only with respect to experiment outputs. To
review the same artifact state on another machine, omit `--verify-live-environment`;
the ledger, lock, output inventory, and result identities remain independently
verifiable, but the strict machine-provenance check is intentionally local.

## Reporting assets

The manuscript is
[`docs/FACFS_STAGE_G_V2_NEGATIVE_RESULTS.md`](FACFS_STAGE_G_V2_NEGATIVE_RESULTS.md).
Its scenario table and figure are generated only from the committed result summary and
the locked thresholds:

```powershell
wsl.exe -d Ubuntu -u farhad --exec bash -c 'cd /mnt/c/Users/farha/repos/sp_lense && /home/farhad/sp_lense/.venv/bin/python scripts/facfs_stage_g_v2_publication_assets.py'
```

Regeneration is tested byte-for-byte in
[`tests/test_facfs_stage_g_v2_publication.py`](../tests/test_facfs_stage_g_v2_publication.py).
The original capture command is intentionally absent from these instructions because
the completed successor lock forbids retry or resume.
