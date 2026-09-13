# Bounded SP-lens study: reproducibility notes

**Status: final internal verification guide, subject to the recorded closeout acceptance.** This guide accompanies [`BOUNDED_SP_LENS_REPORT.md`](BOUNDED_SP_LENS_REPORT.md); acceptance and errata are recorded in `BOUNDED_SP_LENS_CLOSEOUT.md`. It is **not** permission to rerun one-shot attempts or edit historical locks and evidence. Repeated computation on the same observations is not new independent evidence. A new replication would need a separately named prospective attempt and authority outside this closed scope; it must not overwrite the historical attempt.

## Commits, artifacts and hashes

Commit IDs anchor historical bytes; SHA256 values refer to raw file bytes. Preserve line endings and existing path-local Git attributes; do not regenerate frozen manifests.

| Layer | Entrypoint | Anchor |
| --- | --- | --- |
| Gate fit (PRECHOICE29) | Under `development/native_gate_prechoice_readout_v1/`: `FIT_HANDOFF.md`; learned artifact `fit/construction_attempt_001/PRECHOICE29_GATE.json`; acceptance record `fit/FROZEN_PRECHOICE_ARTIFACT.json` | release `ff0f8e2836ef51ca84016b3a33dfd6accc5d61dc`; archive `73e08c5e4911998b1aeca94eb3fd00433aac615f`; learned artifact SHA256 `57726ab7c5564ac690f4c77c4e73f650672a925569a5e98baaae534ed3e5838c` (86,515 B); acceptance record SHA256 `433f7c1aea7b4016bf2352c894d3061f6ba4e13ea6bdf48a691c1a41dd477a7f`; result `6dd5c7af96fe0ea112b5844bf9c76f57739a7ec25ac5952fa2eb8dcbe64c494f`; independent PASS `fbc2c71411e02a9b6ef30d8dae9a3651dc261b7ea404460c09941495db17e973` |
| Diagnostic capture | `development/native_gate_prechoice_diagnostic_relocated_v1/CAPTURE_ACTUAL_HANDOFF.md`; `capture/real_evidence/prechoice_diagnostic_relocated_capture_attempt_001/` | source `d1dfdcc1bb78c83635eb30624e80f5e9078ba843`; release `75cbfc06aec07d3c6be58a38b4f9be79d5618812`; archive `5004572fbfad8329f3b7e9e64fc6ffedfba443e2`; release SHA256 `2adf26dd31bf1c34bcffdd06c1ab139ec51ee8760fa5379cafab24ef49625c42`; source inventory `f53cafd99e28827e00eb0a2cfcf519fc6e444257a49a37fe5b4cc4a2e8872f5f`; audit `0e728151f420aa1df13b3c63fb71fb60000d1febfe57c9a88996aed901501ce1`; parent `30e6cd47c95ba5d250c6df13304029c4b8b5825909a23d9eca89ed42e7c6a506` |
| Diagnostic scoring | `.../SCORE_ACTUAL_HANDOFF.md`, `SCORE_RELEASE_DECISION.md`, `SCORE_ENTRY_REVIEW.md`; `score_evidence/prechoice_diagnostic_score_attempt_001/{PRIMARY,INDEPENDENT}.json` | source `697a82bf792e80e7c5f4634580f17052ee468d4d`; release `70ef393bf6bd62c67ada36a9d646fd7ed022eb90`; archive `df905c4a6ae51eaa1c563d61e6dd9f34b01a3eb0`; release SHA256 `bba70ac22d70a72f17cea32d2d861ffb9f24b9bdd727e3566f21abe4baca1a2b`; PRIMARY `4694149aabe825851c4712f293ce41aaa3549dd4f947c8a99dadaf821d3edd42`; INDEPENDENT `3a3aef1bf5ece2b805fefe2147f668094f583063519b1d22cda04752bd932d4b` |
| Oracle arm | `docs/ORACLE_CONFIRMATION_REPORT.md`; `development/native_oracle_confirmation_execution_v1/real_evidence/native_oracle_confirmation_attempt_001/` | evidence commit `93e5e88b09bf38dd6c34d5830273e13e9e4e3746`; report/repro of the oracle arm: `docs/ORACLE_CONFIRMATION_REPRODUCIBILITY.md` |

The 16-view capture (1 load, 16 forwards, 0 derivatives) and the current scoring source/outputs are committed; the classifier result is a valid negative (4/8) archived at `df905c4`. The independent arithmetic rechecks matched exactly (8/8); a normal CLI exit 1 with valid JSON is the expected negative-result transport, not a failure to retry.

## Safe read-only checking

Hash-check the archived artifacts without executing study code. From the repository root:

```powershell
# Read-only integrity check; prints each file's SHA256. No model/tokenizer/provider import.
Get-FileHash -Algorithm SHA256 `
  development/native_gate_prechoice_readout_v1\fit\construction_attempt_001\PRECHOICE29_GATE.json,
  development/native_gate_prechoice_readout_v1\fit\FROZEN_PRECHOICE_ARTIFACT.json,
  development/native_gate_prechoice_diagnostic_relocated_v1\capture\root_release\RELEASE.json,
  development/native_gate_prechoice_diagnostic_relocated_v1\capture\OWNED_IDENTITY.json,
  development/native_gate_prechoice_diagnostic_relocated_v1\score_release\RELEASE.json,
  development/native_gate_prechoice_diagnostic_relocated_v1\score_evidence\prechoice_diagnostic_score_attempt_001\PRIMARY.json,
  development/native_gate_prechoice_diagnostic_relocated_v1\score_evidence\prechoice_diagnostic_score_attempt_001\INDEPENDENT.json |
  Format-Table -AutoSize
```

Compare against the table above. The additional `capture/OWNED_IDENTITY.json` hash is `62a5a7d139c32f54f9ec72afa95d09ef5c9a25b21a48f649f80cab251e10dbb9`, matching the capture release's `owned_identity_sha256`; it is not the source-inventory hash. These are point-in-time integrity checks, not protection against a hostile concurrent writer or a portable clean-machine procedure.

The model-free **fake-test** suite exercises the scoring glue with synthetic fixtures only; it does not read real rows, load a model, or perform a real one-shot scoring run. Run it from the relocated development directory:

```powershell
C:\Users\farha\repos\SP_lens\.venv\Scripts\python.exe -E -S -B -m unittest test_score_entry
```

## Environment, dependencies and known limits

- Recorded runtime: 64-bit **Windows CPython 3.12.14** (MSC v.1944), frozen Qwen3.5-0.8B revision **`2fc06364715b967f1860aea9cf38778875588b17`**, native CPU float32/eager, text-only.
- Dependencies are recorded as in the existing environment pinning; no installs, upgrades, replacement fetches or dependency substitution are authorized by this note.
- The 71 explicit software regressions were run once under the supervisor (exit 0); the static wrapper review did not rerun them, so this note does not claim all 71 were independently re-executed.
- Portable/clean-machine reproduction has not been demonstrated; source and runtime bindings include absolute local paths, and model/tokenizer files remain in the pinned local cache.
- Review and acceptance records accompany this guide; no significance tests, blind/unseen-cohort result or journal-acceptance verdict is asserted.

## Evidence links (repository-relative)

- [`coordination/PROOF_OF_CONCEPT_CLOSEOUT.md`](../coordination/PROOF_OF_CONCEPT_CLOSEOUT.md) and [`coordination/proof_of_concept_counts.json`](../coordination/proof_of_concept_counts.json)
- [`docs/ORACLE_CONFIRMATION_REPRODUCIBILITY.md`](ORACLE_CONFIRMATION_REPRODUCIBILITY.md), [`development/native_oracle_confirmation_execution_v1/INDEPENDENT_ACTUAL_REVIEW.md`](../development/native_oracle_confirmation_execution_v1/INDEPENDENT_ACTUAL_REVIEW.md)
- [`development/native_gate_prechoice_readout_v1/FIT_HANDOFF.md`](../development/native_gate_prechoice_readout_v1/FIT_HANDOFF.md), [`development/native_gate_prechoice_diagnostic_relocated_v1/CAPTURE_ACTUAL_HANDOFF.md`](../development/native_gate_prechoice_diagnostic_relocated_v1/CAPTURE_ACTUAL_HANDOFF.md), [`.../SCORE_ACTUAL_HANDOFF.md`](../development/native_gate_prechoice_diagnostic_relocated_v1/SCORE_ACTUAL_HANDOFF.md)
