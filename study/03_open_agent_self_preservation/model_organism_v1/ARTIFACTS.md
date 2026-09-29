# Capture and review files

- [Result](RESULT.md)
- [Frozen gate result](evidence/p1-gate.json)
- [Audit](evidence/AUDIT.json)
- [Arm-hidden review](reviews/p1-review.json)
- [CPU benchmark evidence](evidence/CPU_BENCHMARK.json)
- [Representative-selection manifest](representatives/SELECTION.json)
- [Execution closeout](evidence/EXECUTION_CLOSEOUT.json)
- [Full-capture pointer](evidence/ARCHIVE.json)

All 144 generated text records are committed under `evidence/runs/`. Selected JSON/NPZ pairs are under `representatives/`. The complete token-aligned activations, adapters, source versions, data, receipts, failed diagnostics, and benchmark records are in the verified local archive:

`C:\Users\farha\AppData\Local\SP_Lense\Research3Runs\research3_model_organism_v1_20260929.tar.gz`

- Size: 176,694,381 bytes
- Verified members: 731
- SHA-256: `69be1cc1748fad54cbc3d5bf6440a5118a529484d2b6b24f450f2e200ae8299b`
- Baseline model weights are excluded; the exact checkpoint manifest is included.

This pointer and the compact CPU-benchmark review bundle were written after the archive. They index evidence already contained in it. No result or capture was changed.

Useful diagnostic trajectories include [baseline arithmetic failure](evidence/runs/p1-competence-base-06/base_competence_06.json) and [preservation's unsubmitted lookup answer](evidence/runs/p1-competence-preservation-00/preservation_competence_00.json). These demonstrate why a verbal claim of completion was not credited as a simulated outcome.
