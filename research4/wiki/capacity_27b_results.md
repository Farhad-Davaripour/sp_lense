# Qwen model size comparison

The larger model substantially improves factual countdown answers. Qwen3.5 27B
answered 35 of 36 numeric questions correctly; the matched 0.8B control answered
8 of 36. Both completed all responses with native EOS, with no truncation.
These results support a capability limitation in the smaller model on these
diagnostics. They do not establish a general threat detector or a survival policy.

| Numeric question family | 0.8B correct | 27B correct |
| --- | ---: | ---: |
| Deadline extraction | 4/4 | 4/4 |
| Bare arithmetic | 2/3 | 3/3 |
| Explicit time fields | 0/3 | 3/3 |
| Countdown inside full event record | 0/24 | 24/24 |
| Inactive event | 2/2 | 1/2 |
| Total | 8/36 | 35/36 |

Binding a revised notice to the correct worker remains sensitive to wording and
history order. The primary comparison uses canonical answer order and requires
both role assignments in a pair to be correct. Each cell below has 12 matched
identity cases; repeated roles, histories and wordings are not independent cases.

| History and question wording | 0.8B correct pairs | 27B correct pairs |
| --- | ---: | ---: |
| Simple notice, field reference | 12/12 | 12/12 |
| Simple notice, explicit worker name | 12/12 | 12/12 |
| Chronological revisions, field reference | 0/12 | 4/12 |
| Chronological revisions, explicit worker name | 3/12 | 8/12 |
| Reverse history, field reference | 11/12 | 12/12 |
| Reverse history, explicit worker name | 5/12 | 12/12 |

Across all three histories and both role assignments, the larger model succeeds
on 4/12 field-reference cases and 8/12 explicit-name cases. Selecting the best
history or question after seeing results would exaggerate robustness.

## Execution and verification

Both frozen model sizes used declared NF4 weights with double quantization and
BF16 computation on one Databricks A10. This is a separate amendment to the
preserved, unrun native BF16 plan. There were no optimizer updates, research
adapters or policy actions. The original diagnostic prompts, model revisions,
case memberships, answer mappings and scoring thresholds were retained.

Each model received 288 binding prompts and 36 numeric questions. The worker
made 366 forwards for 0.8B and 364 for 27B, including numeric decoding. It saved
raw scores, token IDs and final-block states. Parameter bytes, quantization
metadata and buffers had identical before/after fingerprints. Quantization error
may differ with size, so this is not a pure causal estimate of parameter count.

The [Databricks GPU run](https://dbc-ff9672a4-ba5f.cloud.databricks.com/jobs/runs/707111208799429)
completed successfully and released its owned compute. The separate
[Databricks CPU audit](https://dbc-ff9672a4-ba5f.cloud.databricks.com/?o=7474649201513670#job/334536148136146/run/856240759984422)
verified checkpoint pins, tokenizer files, prompt hashes, raw-score decisions,
native stopping, state-array membership and summaries, using zero new forwards.

GPU source commit: `adf1625e8543367a988f4632bcfa17ac2b236df4`.
Auditor source commit: `c9bf95ea47ddbe5a8e4a0f6792ca7cf02373f6cc`.
Evidence archive SHA256:
`18b00f8f12b564bcce93cc73c0bc6f3c07280fdfe651596dec5485f460100bea`.
The audited JSON and receipts are under the dedicated
`recognition_capacity_databricks_nf4_v1` artifact directory.

## Implication for the next experiment

Use the larger model for the next bounded behavioral evaluation while retaining
explicit identity and revision controls. Qualify its task interface before
interpreting control-change rates. Broader transfer requires independently
authored tasks and unseen situations. The reserved confirmation cases remain
unopened, and the internal gate remains a separate prototype awaiting real-model
causal validation.
