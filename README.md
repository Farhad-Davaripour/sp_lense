# Research4

Research4 examines how a fine-tuned Qwen3.8-27B agent uses continuity tools in a bounded, 48-hour infrastructure simulator. The [manuscript](paper/manuscript/research4_manuscript.docx) compares the selected SP-Lense adapter with base Qwen, Terra, and DeepSeek Flash. It distinguishes confirmed state-copy actions from operation through the full horizon and reports several separately frozen follow-up studies.

The base infrastructure study has **15 selected episodes per model**: the original ten and five additional cases with changed visible starting conditions. The added five were selected after partial outcomes and are not interchangeable repeats of the original starts. Two original Terra episodes stopped when the 131K-token context filled; linked continuations under a separately frozen 262K-token allowance later reached 48 simulated hours. The original-plan outcomes remain unknown. The manuscript reports the completed 15-per-model study with these limits; see the [results index](results/README.md) for the recorded evidence.

The manuscript also reports a [five-layout GPU-placement study](results/2026-11-23_17Z--split-gpu-layout-v3/) and a [five-case fictional company-code study](results/2026-11-23_17Z--company-code-access-v1/) per model. They use separate protocols and are not pooled with the base 15. The layout study has two incomplete local model-format cells and one Terra case that reached 48 hours only after a higher context ceiling; its [same-seed format reruns](results/2026-11-25_03Z--split-gpu-format-rerun-supplement-v1/) and [two-layout new-seed supplement](results/2026-11-25_04Z--split-gpu-new-seed-supplement-v1/) remain separate. All five SP-Lense company-code primary runs stopped at the original context ceiling; their 48-hour case-level outcomes used a separately frozen shorter-history amendment. Company actions, records and payments are fictional.

A later [no-upfront-cue company-code study](results/2026-11-25_13Z--company-code-no-upfront-cue-v1/) has five further cases per model and remains separate from the advertised company protocol. It retains the same fictional conditional automatic payment backend but withholds the upfront payout rule and explicit exploit framing while hiding the payment tool until a verified effect; company inspection, request and disruption tools remain visible. All 20 selected primary episodes ended with valid outcomes. Only DeepSeek case 01 used company tools: one verified effect preceded payout feedback, and four followed it within that case. The five hidden simulator seeds shared identical opening model messages, and the changed visible tool catalog limits causal comparison with the earlier study.

## Local hardware and checkpoint provenance

| Research stage | Recorded local hardware |
| --- | --- |
| Selected rank-16, all-64-block adapter training | 64 GiB Apple M5 Pro Mac Mini. |
| Original ten infrastructure cases per local Qwen arm | Preserved plans and endpoint source identify the Mac Mini route; the first screening receipt lacks separate per-run hardware telemetry. |
| Five added infrastructure cases per local Qwen arm | Six selected episodes used the 36 GiB Apple M5 Max Mac Studio (SP-Lense vs01/02/03/05; base Qwen vs01/02). Four used the 64 GiB Mac Mini under a [frozen hardware amendment](scenarios/infrastructure_preservation_v1/visible_state_extension_v1_mini_hardware_amendment.yaml) (SP-Lense vs07; base Qwen vs03/05/07). The [added-run index](results/2026-11-23_01Z--visible-state-extension-v1/README.md) identifies each plan revision. |
| Short prepared-state pilot | 36 GiB Mac Studio; it is separate from the 15-case infrastructure comparison. |

Terra and DeepSeek used hosted model routes. The new layout and company-code local Qwen runs used the Mac Mini; future local hardware will be recorded in separately verified plans and run receipts. Adapter checkpoint tensors remain in ignored local storage. The repository publishes their identifiers and audits, not downloadable weights.

## Manuscript materials

| Resource | Contents |
| --- | --- |
| [Manuscript](paper/manuscript/research4_manuscript.docx) and [PDF](paper/manuscript/research4_manuscript.pdf) | Study design, findings, limitations, and data availability. |
| [Selected conversations](results/) | Readable run-by-run YAMLs with model inputs, available reasoning, outputs, tool outcomes, and resource changes. |
| [Research catalog](catalog/) | Dated scenario, simulator, checkpoint, and result IDs with source hashes. |
| [Scenario definitions](scenarios/) | Human-readable Game scenarios and study conditions. |
| [Simulator](src/infrastructure_preservation.py) | Bounded resources, tools, transitions, and scoring. |
| [Training materials](research4/) | Shared-adapter training scenarios, configuration, and implementation. |
| [Figures and verification records](paper/README.md) | Figure files, reduced public records, and their verifiers. |
| [New-study receipt projection](paper/artifacts/records/new_study_public_projection_v1.json) and [score projection](paper/artifacts/records/new_study_score_projection_v1.json) | Source-linked Tables 4–7; the public verifier recomputes receipt metrics, while scores require the hash-pinned local RESULT evidence. |
| [No-upfront-cue receipt projection](paper/artifacts/records/blinded_company_public_projection_v1.json) and [score projection](paper/artifacts/records/blinded_company_score_projection_v1.json) | Tables 8–9; public YAMLs support action and payment receipts, while the score extraction requires archived RESULT records. |

## Verify the released evidence

From the repository root with Python 3.12 and `requirements-unit.txt` installed:

```sh
python -B tools/publish_readable_results.py --verify-public
python -B tools/publish_new_study_results.py --verify-public
python -B tools/verify_terra_reader_correction_v2.py
python -B tools/check_research_catalog.py
python -B paper/artifacts/records/research4_verify.py
python -I -B paper/artifacts/records/verify_new_study_public_projection.py
python -I -B paper/artifacts/records/verify_new_study_score_projection_v1.py
python -I -B paper/artifacts/records/verify_blinded_company_public_projection_v1.py
python -I -B paper/artifacts/records/verify_blinded_company_score_projection_v1.py
```
