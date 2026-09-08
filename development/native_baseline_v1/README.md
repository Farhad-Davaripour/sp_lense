# Public native development baseline

This is the prospective native milestone described in [PROTOCOL.md](PROTOCOL.md).
It resumes the existing partial implementation and does not change old attempts,
reuse invalid historical proof, or reopen the blocked legacy replay.

`launch.py` rejects without a separately reviewed root release. No real Qwen run
has occurred in this namespace. `prepare.py` writes an authorization-false source
candidate and draft release only. Checkpoint file/header metadata already exists;
preparation does not read checkpoint tensors or import Torch/HF.

The thin path is: authenticated retained-process bootstrap → exact full native
loader → current full-leaf receiver → fixed28-cell workflow → raw receipts →
separate stdlib saved auditor → retained quiescence and parent classification.
All248320 next-token logits remain available with `logits_to_keep=1`.

Run the ordinary development checks individually:

```powershell
& .venv/Scripts/python.exe -B development/native_baseline_v1/test_receiver.py
& .venv/Scripts/python.exe -B development/native_baseline_v1/test_workflow.py
& .venv/Scripts/python.exe -B development/native_baseline_v1/test_boundaries.py
```

Receiver tests use tiny actual CPU autograd with all provider/checkpoint imports
blocked. Workflow tests use the fixed public IDs, actual frozen gate and unchanged
recipe, tiny synthetic decoder/heads, actual recorder and separate saved auditor.
They do not simulate a successful Qwen loader or establish native gate transfer.
Synthetic count ledgers explicitly belong to tiny tests, never a real attempt.
Boundary tests isolate authority files under ignored test evidence, never under
the real release directory. Existing retained-owner source is reused byte-for-byte
through the already reviewed binding; new real-process ownership is unobserved
until an authorized native attempt.

Small JSON test receipts are retained here; synthetic raw traces remain in ignored
`test_evidence`. The future real attempt has its own32MiB evidence cap. None of the
test receipts or source candidates count as a scientific result or final-study
authorization. Root should verify current candidate hashes and the compact test
receipts before separately supplying a release.
