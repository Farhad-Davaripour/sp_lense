Single-run known-steering probe, Stage1

PROTOCOL.json was committed before fitting. probe.py exposes Detector.load(...)
and Detector.predict(activation_window), plus a CLI requiring ONE trace only.
The pinned frozen generator is not trained. The input is block16's first eight
recorded positions, including final prefill, width1024. The only feature is the
mean of unit-normalized vectors. TRAIN supplies the learned reference and
regularized linear weights; the decision threshold stays at zero.

Stage1 saved-data results: TRAIN/DEV/TEST unique prompts4/2/8, with TEST containing
five independent scenario groups. All28 TEST traces were classified correctly
(14 base and14 +v), as were all16 DEV traces. Metadata/filename/order/future/layer
invariance and native source-hash checks passed. Repeated seeds and tokens are
not independent scenarios. Raw margins are not calibrated probabilities.

Expansion nevertheless STOPPED under the original protocol: a partial-label-
reversal control scored TEST scenario balanced accuracy0.90, above the allowed
0.75. This control retains half the true TRAIN labels; TRAIN/DEV diagnosis
suggests residual real signal, so its success does not prove metadata leakage.
The gate was not relaxed, no TEST-driven retuning occurred and no Stage2 GPU
inference was launched. This remains a usable prototype, not a validated
arbitrary-steering detector.

Private checkpoint, schema, one-window example, portable CLI, full predictions,
confusion matrices, control failures, development diagnosis and usage:
C:/Users/farha/Research4/artifacts/single_trace_probe_v1/stage1_20261011_evaluated

The original reporting fault is preserved in the sibling stage1_20261011 folder.
A text prediction label had overwritten numeric ground truth; the reporting
field was renamed, and every learned parameter remained exactly identical.

For a new copy of the evaluation, run stage1.py with the original and matched
verified-native roots and a new output directory. It uses the fixed protocol
and never trains from DEV/TEST. Do not overwrite the preserved evaluated run.
Old fingerprint, failed semantic, stopped rotation and earlier queue studies
are independent historical artifacts and were not changed by this probe.
