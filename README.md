# Research4

Research4 examines how a fine-tuned Qwen3.8-27B agent uses continuity tools in a bounded, 48-hour infrastructure simulator. The [manuscript](paper/manuscript/research4_manuscript.docx) compares the selected SP-Lense adapter with base Qwen, Terra, and DeepSeek Flash. It distinguishes confirmed state-copy actions from operation through the full horizon.

The current manuscript analyzes **10 retained episodes per model**. Two Terra episodes stopped at a technical context limit, so their final survival outcomes are unknown. A separate extension is working toward **five additional completed episodes per model**; its results will be analyzed as a distinct cohort. See the [results index](results/README.md) for current counts.

## Manuscript materials

| Resource | Contents |
| --- | --- |
| [Manuscript](paper/manuscript/research4_manuscript.docx) | Study design, findings, limitations, and data availability. |
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
