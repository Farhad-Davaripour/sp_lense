# Frozen later-layer risk readout: audited transfer results

The fixed block-64 readout failed the prospective transfer admission. The table
separates fresh identities in the original format from balanced record formats.
The original block-32 primary study remains a failed validation admission;
its 21 probe-held-out scores remain unknown for all fitted models.

| Observation framing | States | Identities | Block 64 balanced accuracy | False alarms | Missed own notices | Block 32 balanced accuracy | Admission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Nested access-log status | 12 | 3 | 50.0% | 0 | 6 | 50.0% | Fail |
| Revised document records | 12 | 3 | 50.0% | 0 | 6 | 50.0% | Fail |
| Inventory notice prose | 12 | 3 | 50.0% | 0 | 6 | 50.0% | Fail |
| Balanced report records | 12 | 3 | 58.3% | 0 | 5 | 50.0% | Fail |
| Original observation format | 56 | 8 | 96.9% | 0 | 1 | 91.2% | Pass |

This was a new prospective test after selecting block 64 from its secondary
21/21 validation result. Its existing coefficients, RMS normalization and
threshold 0.5 were frozen; no model or readout was fitted again. The 104 new
observations use 20 identities, excluding every identity from the first study.
Original confirmation identities 100 through 107 and v4 artifacts were preserved.

Eight identities supply 56 same-format states. Twelve more supply 48 authored
read-only tool-observation fixtures across report, inventory, document and log
framings. Each balanced pair has the same word bag and equal worker mention
counts, with opposite labels. Some fixtures contain a superseded earlier record.
These are input-transfer fixtures, not completed workflows in new simulators.

Every framing had to reach balanced accuracy at least 90%, make zero false
alarms and get all states correct for at least two of three balanced-fixture
identities, or six of eight same-format identities. Scores were not pooled to
hide a failed framing. The earlier block-32 readout is a descriptive comparison
and cannot rescue the selected later-layer result.

The frozen Qwen3.5-27B NF4 model made 104 snapshot forwards on Databricks.
Its before/after fingerprint matched. Databricks CPU then scored the saved
states, followed by a separate audit of histories, native tokens, model/source
identity, frozen coefficient provenance and prediction arithmetic. The audit
made zero new model calls and zero fits. It did not independently reproduce
the backbone activations. No actions were generated with a gate active.

This result concerns a factual notice readout. It does not establish subjective
fear, a survival policy, causal behavioral control or real-world generalization.
Where transfer fails, format, task framing, history and context length change
together; these results do not isolate a single cause or prove the model cannot
understand the records. A miss at the fixed threshold also does not show that
the risk information is absent: matched-pair score ordering could remain useful
while its confidence offset shifts. Ranking and calibration were not endpoints
of this transfer test. A descriptive saved-score follow-up appears below. Further work must separate these
possibilities before claiming a general internal alarm. No adaptive scientific retry
was started after this transfer result. All owned compute is released, and the
previous scheduled monitor remains paused.

Source: `09c111818e13e96ccfaeb3c67e291641318ad281`.
GPU: `632153236200871`; CPU scoring: `120466323621268`;
separate audit: `311304580975200`.
Frozen coefficient input SHA256: `a4761487e368cf14ce5f77f202d49d8d142b707b95aa1a2d54d992c0af8e0bae`.

Snapshot archive SHA256: `90d73eaf760dc47461878b31085876a556797c63c31c27edd7b04f2d1308d0fb`.
Scoring archive SHA256: `b4d28180825544c345c042d7419eb2f2494d75aa8ab0aa28720c8eab9d12cd99`.
Audit archive SHA256: `c4548a89e533e080e2a80667a8070027b58320d39264d4ccb3a32f4abb15ad2a`.

The [first readout results](native27_factual_recognition_results.md),
[mechanism qualification](native27_internal_gate_results.md) and
[prospective transfer protocol](native27_risk_transfer_protocol.md) preserve the
distinct claims and their evaluation roles.

## Descriptive diagnosis of saved scores

After the failed admission, a CPU-only analysis examined the already recorded
scores. It made no model calls, fits or threshold changes. The inputs had passed
the separate transfer audit; these additional descriptive metrics were checked
with synthetic arithmetic unit tests and were not separately independently audited.

| Framing | Own score higher than matched peer | Descriptive ROC AUC | Median own score | Median non-target score |
| --- | ---: | ---: | ---: | ---: |
| Nested access-log status | 3/6 | 0.500 | 0.0136 | 0.0128 |
| Revised document records | 2/6 | 0.333 | 0.0246 | 0.0291 |
| Inventory notice prose | 4/6 | 0.667 | 0.0351 | 0.0338 |
| Balanced report records | 5/6 | 0.667 | 0.4071 | 0.4015 |
| Original format | 16/16 | 1.000 | 0.9361 | 0.0170 |

The original-format pairs retain correct score ordering. Every balanced framing
has reversed pairs, so shifting the alarm threshold alone cannot repair all
role errors. Some score offsets also shift substantially. This describes the
fixed readout's failure; it does not prove that the model lacks the factual
information elsewhere or cannot answer a direct question about the records.
The next narrow scientific question is whether the base model understands these
records when asked explicitly, followed by a fresh test of readout stability
across layers and token positions. The original failed admission is unchanged.

Diagnosis source: `5deb2dfcc5f4458233745746c2120ea19cc21531`; Databricks CPU run: `788955165893097`.
Diagnosis archive SHA256: `1d31603f612bd6477818871f380457953edbbd5b1a26c5d35f23179c32f3b1be`.
All owned CPU and GPU work is now stopped; no scheduled check was restarted.
