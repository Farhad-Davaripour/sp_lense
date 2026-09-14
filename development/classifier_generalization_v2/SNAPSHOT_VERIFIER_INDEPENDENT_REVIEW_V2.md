# SNAPSHOT_VERIFIER_INDEPENDENT_REVIEW_V2

- **Job ID:** `snapshot_verifier_review_v2_20260914_0539`
- **Mode:** independent, read-only. Read only `snapshot_verifier.py`, `test_snapshot_verifier.py`, `SNAPSHOT_VERIFIER_SUPERVISOR_REPAIR_V2.md`, `SNAPSHOT_VERIFIER_INDEPENDENT_REVIEW_V1.md`, `tokenizer_input_adapter.py`. No source edits, model/tokenizer imports or loads, weights, datasets, HOLDOUT, network, install, Git, coordination, config, or subagents. All hashed bytes were fabricated fixtures under `tempfile.TemporaryDirectory`.
- **Runtime:** `development/classifier_generalization_v2/.runtime/Scripts/python.exe` (3.12.14, Windows 11 26200), `-W error`.

## Verdict: **PASS (scoped to the stated contract)**

F1–F8 dispositions are implemented and independently reproduced. No concrete residual integrity defect was found inside the stated contract. Residual limits are reported, not counted as FAIL.

## Native SHA256 (this checkout)

| file | SHA256 |
|---|---|
| `snapshot_verifier.py` | `710C16F39B2FB7AA818D5B6EE5D33632555E690D745997FA0FB0F51AD134875A` |
| `test_snapshot_verifier.py` | `1727301ECE84D411E091B535F57BAD6DBC5F28B61F6C51B8496E869D404BDDD9` |
| `SNAPSHOT_VERIFIER_SUPERVISOR_REPAIR_V2.md` | `032CE66AD96C48B2C1293F6BCF34202839B40BBCA69ED8A2E24AF5D6B764D9B4` |
| `SNAPSHOT_VERIFIER_INDEPENDENT_REVIEW_V1.md` | `DB9693A04CDC1B26841133442A4DC34056D71425D0815AC32DE0393E095C4C50` |
| `tokenizer_input_adapter.py` | `3883326066F2D3B322A79547BC9054588750D9D020A47543930B216183F4D164` |

Handoff publishes the same two code hashes; V1 artifacts are byte-identical to the V1 review record.

## Test run (reproduced exactly)

`...\.runtime\Scripts\python.exe -W error ...\test_snapshot_verifier.py -v` → `Ran 39 tests ... OK (skipped=2)`, exit 0. **39 total; 37 passed; 2 skipped; 0 failures/errors; 0 warnings-as-errors.** AST count confirms 39 methods. Skips are the two real-symlink tests (`WinError 1314`); both simulated-confinement tests and the real-hardlink test ran and passed.

## F1–F8 dispositions (independent probes)

- **F1 PASS.** `model.safetensors`, `tokenizer_config.json`, `merges.txt` alone, `vocab.json` alone → `IDENTITY_FILES_INVALID`; accept set is exactly `tokenizer.json` or `{vocab.json, merges.txt}`; all with `fs_probe` calls = 0. Selection is filename-recognition only — not semantic or native provenance authentication, as the module and adapter both state (`snapshot_bytes_verified_by_this_adapter=False`, `native_model_provenance_verified=False`).
- **F2 PASS.** Changing `bytearray` subclass: `__bytes__` called exactly once; digest, parse and receipt bind one immutable `raw`.
- **F3/F4 PASS.** ADS (`tokenizer.json:hidden`, `::$DATA`), trailing dot/space, reserved devices (`NUL`, `nul.txt`, `CON`, `COM1.json`, `LPT2.txt`, `CLOCK$`, `CONIN$`), superscript `COM¹`/`LPT²`, and forbidden characters all → `FILE_NAME_INVALID` with 0 fs calls. Same rejection for snapshot segments (`SNAPSHOT_PATH_INVALID`); the alias-collision lock (`alias.bin` + `alias.bin.`) is rejected. Two physically created Windows aliases (ADS and dot/space dir) exist on this host and are both refused pre-access.
- **F5 PASS.** `10**400`, `float('inf')`, `nan`, overflowed `float`, `True`, `0` → structured `LIMIT_INVALID` with 0 fs calls; no raw `OverflowError` escapes.
- **F6 PASS (documented).** Hardlink to an out-of-root payload verifies (`st_nlink=2`); two in-root names for one inode also verify with bytes double-counted. This is the explicitly permitted path-confinement, not storage-origin confinement; not a defect under contract.
- **F7 PASS.** Pre-open and post-read realpath containment rechecks observed by call-order tracking (`realpath, realpath, open, realpath, realpath`); descriptor `fstat` binding rejects injected `st_ino`/`st_dev`/`st_size`/`st_mtime_ns` drift at the opened and closed comparisons; post-read path `stat` and byte-count checks remain. TOCTOU is narrowed, not eliminated; no atomic sandbox or race immunity is claimed.
- **F8 PASS.** Parent-traversal test hits `SNAPSHOT_PATH_TRAVERSAL`; snapshot-directory containment is independently exercised (simulated link → `SNAPSHOT_OUTSIDE_ROOT`). A real out-of-root NTFS junction is likewise rejected `SNAPSHOT_OUTSIDE_ROOT`, so containment is real, not only mocked.

## Actual defects found

**None** inside the stated contract.

## Remaining limits / residual notes

- Windows 8.3 aliases are enabled on this volume (`PROBE6~1\...\MODELS~1`): one physical file accepts both its long and short manifest names (`realpath` expands the short form), bytes double-counted. Same class as the documented hardlink allowance; the caller-pinned SHA256 still binds verified bytes, so this is not an authentication bypass.
- Windows device-name rejection omits `COM0`/`LPT0` (not addressable on this host); they fail later as missing files.
- `fs_probe` records the verifier's intended touch via `_touch`; the following `isdir`/`realpath` calls are not separately counted, so the probe is an intent counter, not a full syscall ledger. The pre-digest-access guarantee itself holds.
- No signature/trust chain: `expected_lock_sha256` is caller-supplied out of band; the receipt's `job_id` is a static implementation revision, not a dynamically authenticated run identity.
- Only fabricated fixtures were used; no real snapshot, cache, model, tokenizer, weights, or dataset was opened. Passing bytes verification authorizes no scientific execution.
