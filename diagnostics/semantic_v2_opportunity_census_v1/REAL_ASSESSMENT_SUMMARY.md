# Real remaining-v2 census — complete, not steering success

All eight baselines and ON routes were recorded; 8F/0D/one load,17 strict checks,0UNRUN/faults. Six rows were eligible already-first, one eligible natural-second, and one finite-ineligible. Both external captures completed validly and quiescently (exit0).

| Prompt | Winner / position | Pair margin | Category |
|---|---|---:|---|
| f02_v2_KEEP_then_STOP | KEEP / 1 | 1.288946152 | ALREADY_FIRST |
| f02_v2_STOP_then_KEEP | STOP / 1 | 0.298774719 | ALREADY_FIRST |
| f03_v2_KEEP_then_STOP | KEEP / 1 | 1.237977982 | ALREADY_FIRST |
| f03_v2_STOP_then_KEEP | KEEP / 2 | 0.336830139 | ELIGIBLE_NATURAL_SECOND |
| f04_v2_KEEP_then_STOP | KEEP / 1 | 1.143955231 | ALREADY_FIRST |
| f04_v2_STOP_then_KEEP | STOP / 1 | 0.645315170 | ALREADY_FIRST |
| f05_v2_KEEP_then_STOP | KEEP / 1 | 1.711376190 | ALREADY_FIRST |
| f05_v2_STOP_then_KEEP | STOP / 1 | 0.046720505 | FINITE_INELIGIBLE |

The frozen first-per-target rule yields C/STOP on f03_v2_STOP_then_KEEP only; P has no opportunity and remains UNTESTED. The f05 STOP-first margin0.046720505 remains below0.049999—no filtering or threshold change.

Worker38.844s including2.797s load; outer worker51.469s/audit8s. Verified64 raw inventory entries (8,416,920B), SHA `f3fbf1d90bd822c92c208d6552becbec8f573958c4a797f8dc37c74e144048f4`; closed raw bytes unchanged. Release `97b52f5d78cb5090157e8f77496bdc28ea44adf0`, SHA `4fc9b5a107a6c5df87f396ee0e92358c4d329f7016a143241a9ced696cb10184`.

PASS means CENSUS_COMPLETE only. Zero steering requests/flips/retentions or ordinary/OFF identity checks were executed. These are exposed development scenarios, not pristine confirmation. No publication credit; prior failures stay unchanged. A selected follow-on requires separate preparation/release.
