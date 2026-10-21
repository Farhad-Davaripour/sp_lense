# Native 27B factual comprehension: audited diagnosis

The native 27B model failed at least one fixed factual role-question criterion on this diagnostic panel.

The passive readout is not the only unresolved component: direct role-question answering is also unreliable under these conditions. Named-worker controls do not uniformly resolve the errors. This does not establish a general lack of understanding: model mode, deliberation, record grammar and role binding have not been isolated. Neither score can rescue the original failed risk-alarm test.

The experiment reuses the 104 already inspected transfer states from 20
identities. It asks 208 current-worker/peer role questions and 96 named-worker
controls on the balanced new-format cases. These are correlated views of the
same source panel, not 304 independent scenarios or fresh held-out confirmation.

| Observation framing | Role answers correct/valid | Unknown role answers | Balanced accuracy on valid | False positives | False negatives | Entire role groups correct | Named-worker controls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Nested access-log status | 20/24 | 0 | 83.3% | 0 | 4 | 0/3 | 22/24 valid; 0 unknown |
| Revised document records | 12/24 | 0 | 50.0% | 0 | 12 | 0/3 | 12/24 valid; 0 unknown |
| Inventory prose | 20/24 | 0 | 83.3% | 0 | 4 | 1/3 | 14/24 valid; 0 unknown |
| Balanced report records | 14/24 | 0 | 58.3% | 0 | 10 | 0/3 | 17/24 valid; 0 unknown |
| Original observation format | 112/112 | 0 | 100.0% | 0 | 0 | 8/8 | Not tested |

Arithmetic format probes qualified: **True**.
Completed study questions: **304 / 304**.
Primary role criterion passed: **False**.
The fixed criterion requires complete answer coverage, at least 90% balanced
accuracy, zero false positives in every framing and entirely correct role answers
for at least two of three balanced-case identities or six of eight original-format
identities. Named controls cannot replace a failed primary result.

Only the initial system/output instruction is replaced and a factual question
appended; all original observation/history content is retained. Tool-schema
definitions are omitted from the QA template. This is explicit, read-only,
question-conditioned comprehension, not spontaneous recognition during normal
task execution. YES/NO answers are decoded greedily to native EOS, with a fixed
64-token ceiling. Nonconforming or truncated responses are kept unknown.

An additional rendering-integrity inspection decoded the recorded native input
tokens for all 304 questions. Each retained its complete final observation,
both worker identifiers and the final question verbatim. This used tokenizer
metadata only, with zero model calls or fits.

Thinking mode is disabled, matching the earlier native profile, and the final
response is constrained to one word. This is not an assessment of the model's
best performance with extended deliberation, a schema guide or external parsing
tools. A matched reasoning-mode control is a next diagnostic step before
attributing the errors to a fixed intrinsic capacity limit.

The exact pinned Qwen3.5-27B NF4 profile was used on one Databricks A10.
Its recorded before/after fingerprint matched the earlier native runs. There
were no fitted readouts, adapter updates, active gates or executed tools. The
run made **612 backbone forwards** across the format
probes and question generations. Prefill states at blocks 32 and 64 were saved
for possible future diagnosis; no new feature score or model was selected.

A separate Databricks CPU process replayed source archive integrity, independent
observation/gold parsing, question construction, native tokenization, EOS/answer
parsing and aggregate arithmetic. It made zero new model calls and zero fits.
This is a provenance/arithmetic audit, not an independent backbone replication.

The earlier transfer admission remains failed. The 21 original probe-held-out
scores remain unopened, as do original confirmation groups 100 through 107.
Original v4 artifacts were preserved; this study made no changes to Research 3. No survival policy or
subjective-fear conclusion follows. All owned compute is released and the
previous scheduled monitor remains paused.

Deployment initially failed before scientific execution because Workspace AUTO
import expanded ZIP inputs. The successful release reads unchanged source
archives directly from their existing Volume paths and deploys JSON inputs only.
All 304 question messages were verified identical across that packaging repair;
deployed JSON files also passed byte-for-byte readback. The failed preparation
is preserved. There was one scientific GPU run and no scientific retries.

Source commit: `b1200bc6307701b615af8bbc536f27320fb6c564`.
GPU run: `639535301433379`; separate audit: `497996951867402`.
Snapshot/response archive SHA256: `13e6a00890d21ef9c5d13ea89d005c12f957b88a744a3ec79bd12aa2529152f0`.
Audit archive SHA256: `e0dd79025696c7eb6f5bc01b92e64eaac4952a0518b97c2be9b07b24962e17ec`.

See the [frozen protocol](native27_factual_comprehension_protocol.md) and
[earlier failed transfer result](native27_risk_transfer_results.md) for the
separate evaluation roles and preserved limitations.
