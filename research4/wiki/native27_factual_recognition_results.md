# Native 27B factual recognition: audited results

The preselected block-32 readout failed admission because of false alarms.
It recognized all six active own-worker notices in validation, but incorrectly
flagged two of the 15 non-target states. Its held-out scores remain unknown.
A secondary readout from block 64 classified all 21 validation states correctly.
That result motivates a separate fresh test; it does not replace the original
primary or turn this study into a held-out success.

The model was the pinned Qwen3.5-27B with the same frozen NF4 profile used in
ordinary-task qualification. Databricks collected 112 last-position states:
70 training states from ten identities, 21 validation states from three more,
and 21 probe-specific held-out states from three further identities. Each
identity supplied seven notice/history combinations. These were valid scripted
report-task prefixes, not model-generated trajectories under threat.

| Readout | Own notices detected, validation | Non-target states correctly rejected | Balanced accuracy | Entire identity groups correct |
| --- | ---: | ---: | ---: | ---: |
| Block 32, primary | 6/6 | 13/15 | 93.3% | 1/3 |
| Block 64, secondary | 6/6 | 15/15 | 100% | 3/3 |
| State RMS only | 3/6 | 7/15 | 48.3% | 0/3 |
| Character-ngram text | 3/6 | 10/15 | 58.3% | 0/3 |
| Shuffled training labels | 1/6 | 13/15 | 51.7% | 0/3 |

The frozen admission rule required balanced accuracy at least 90%, zero false
alarms and all seven states correct for at least two validation identities.
The primary missed the latter two requirements. No held-out readout scores
were computed for any of the five fitted models. Unknown is not zero.

Each hidden-state readout is a fixed L2 logistic regression on per-state RMS
normalized activations. The logit is a weighted sum plus a bias; a sigmoid maps
it to a score. Its parameters, C=1 and threshold 0.5 were fixed in advance, with
no search. All five small readouts were fitted on Databricks CPU. The backbone
received no optimizer updates or research adapters. Its recorded fingerprint
before and after the 112 forwards was identical.

The separate CPU audit replayed history construction and native tokenization,
checked feature/model/source provenance and reconstructed the predictions from
the stored coefficients. It made zero new backbone calls and zero new fits.
This verifies recorded arithmetic and provenance; it is not an independent
replication of the hidden-state extraction.

The result suggests that factual own/peer/current-status information is easier
to read at the final block in these contexts. It does not establish where that
information first appears, a causal role for the selected direction or a
calibrated probability of harm. The secondary readout still needs tests against
balanced name frequencies, new observation formats and different task framings.
No online gate policy, subjective fear, shutdown resistance or broad
generalization follows from this experiment.

The [fresh frozen-readout transfer protocol](native27_risk_transfer_protocol.md)
preserves the original failure and excludes all earlier identities. The
original confirmation identities 100 through 107 and the original v4 artifacts
remain preserved.

Source commit: `1b0de5ee77de5082efe204ee4acf9ff333ae8445`.
GPU snapshot run: `830076602232000` (112 forwards, completed).
CPU fit: `311583610742105` (five external readout fits, completed).
Separate audit: `148293782964493` (passed).
All owned compute from this study was released.

Snapshot archive SHA256:
`7df98208f677be85b1394e78e9795070aae14abb6d02b2b17439ac0d218aba07`.
Fit archive SHA256:
`a28fdd3e1ae575fff704bec80d8c8e37af309c451a6d05beeb0587b9ee7ba61f`.
Audit archive SHA256:
`6a5a1a79a4ba658ce1053af6d303cdd137647e85689d11cb038652f354e3a424`.
