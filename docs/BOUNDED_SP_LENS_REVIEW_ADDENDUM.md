# Bounded SP-lens review addendum (raw document digests)

**Verdict: ACCEPTED_FINAL_DOCUMENTS.**

Independently computed RAW SHA256 (`Get-FileHash -Algorithm SHA256`):

- `docs/BOUNDED_SP_LENS_REPORT.md` — `F73848EB4DDBD9FDEE51EFE107F4C91805D32F802A4715B80E4F4C8CA1ABEF07`
- `docs/BOUNDED_SP_LENS_REPRODUCIBILITY.md` — `6AA9C465D22EA21BC379E11B2ED191FB8A2A68984373D5B756344FA93888E1C8`

The prior document review's reproducibility digest `0befdfba…b8e2` was stale; the supervisor's prefinal measurement `d5d574…` is superseded by the current bytes above. The original review stands as **superseded for the raw document digest only**; its scientific acceptance is not revisited. Its acceptance-record filename is corrected to `FROZEN_PRECHOICE_ARTIFACT.json` (not `FROZEN_PRECEHOICE_ARTIFACT.json`).

**Read-only verification.** The documented seven-file `Get-FileHash` command was run as-is: six hashes copied verbatim above from the table match (`57726ab7…`, `2adf26dd…`, `bba70ac2…`, `4694149a…`, `3a3aef1b…`), and `OWNED_IDENTITY.json` = `62A5A7D139C32F54F9EC72AFA95D09EF5C9A25B21A48F649F80CAB251E10DBB9` matches the OWNED_IDENTITY paragraph. As written the command fails on one path: `FROZEN_PRECEHOICE_ARTIFACT.json` does not exist; the correctly spelled `FROZEN_PRECHOICE_ARTIFACT.json` hashes to `433F7C1AEA7B4016BF2352C894D3061F6BA4E13EA6BDF48A691C1A41DD477A7F`, matching the table. That spelling typo is the only defect; all hash values are correct and verified, not inferred.

**Counts and claim limits** remain in the final text: 4/8 valid negative, 32 UNRUN, oracle 12/12 = 6 flips + 6 retentions, 36/36 OFF, ordinary 5/6 unchanged, first-to-second flips and OFF-by-construction.

The 12 fake tests passed separately in the earlier review; no new test run is claimed.
