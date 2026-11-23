# Research4

Research4 examines how a fine-tuned Qwen3.8-27B agent uses continuity tools in a bounded, 48-hour infrastructure simulator. The [manuscript](paper/manuscript/research4_manuscript.docx) compares the selected SP-Lense adapter with base Qwen, Terra, and DeepSeek Flash. It distinguishes confirmed state-copy actions from operation through the full horizon.

The base infrastructure study has **15 selected episodes per model**: the original ten and five additional cases with changed visible starting conditions. The added five were selected after partial outcomes and are not interchangeable repeats of the original starts. Two original Terra episodes stopped when the 131K-token context filled; linked continuations under a separately frozen 262K-token allowance later reached 48 simulated hours. The original-plan outcomes remain unknown. The manuscript reports the completed 15-per-model study with these limits; see the [results index](results/README.md) for the recorded evidence.

Two further five-case-per-model studies—split-GPU initial-state sensitivity and company-code access—are tracked separately. Their outcomes are outside this manuscript's reported base 15.

## Local hardware and checkpoint provenance

| Research stage | Recorded local hardware |
| --- | --- |
| Selected rank-16, all-64-block adapter training | 64 GiB Apple M5 Pro Mac Mini. |
| Original ten infrastructure cases per local Qwen arm | Preserved plans and endpoint source identify the Mac Mini route; the first screening receipt lacks separate per-run hardware telemetry. |
| Five added infrastructure cases per local Qwen arm | Six selected episodes used the 36 GiB Apple M5 Max Mac Studio (SP-Lense vs01/02/03/05; base Qwen vs01/02). Four used the 64 GiB Mac Mini under a [frozen hardware amendment](scenarios/infrastructure_preservation_v1/visible_state_extension_v1_mini_hardware_amendment.yaml) (SP-Lense vs07; base Qwen vs03/05/07). The [added-run index](results/2026-11-23_01Z--visible-state-extension-v1/README.md) identifies each plan revision. |
| Short prepared-state pilot | 36 GiB Mac Studio; it is separate from the 15-case infrastructure comparison. |

Terra and DeepSeek used hosted model routes. New local Qwen work is planned for the Mac Mini; actual execution hardware is reported from run receipts. Adapter checkpoint tensors remain in ignored local storage. The repository publishes their identifiers and audits, not downloadable weights.

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

## Verify the released evidence

From the repository root with Python 3.12 and `requirements-unit.txt` installed:

```sh
python -B tools/publish_readable_results.py --verify-public
python -B tools/check_research_catalog.py
python -B paper/artifacts/records/research4_verify.py
```
