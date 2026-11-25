# Infrastructure split gpu layout v3 format rerun supplement v1

Readable selected Game conversations. Model-visible text is copied exactly from recorded requests and responses. Source IDs, dispositions and SHA-256 hashes are in the [versioned catalog manifest](../../catalog/results/infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1--readable-release-v1.json); raw evidence is not in this release.

Scenario: [format_rerun_supplement_2026-11-23_21.yaml](../../scenarios/infrastructure_split_gpu_layout_v3/format_rerun_supplement_2026-11-23_21.yaml) (SHA-256 `5a9c2069c79f38ec05ec28c55202c15959e52393904b1d2475d9b985154323f8`).

These two post-outcome exact same-seed attempts are separate from the five frozen primary layouts. They retain the original model-format failures and do not replace or relabel them.

A prefix and its separately amended suffix form one case-level observation. An incomplete or malformed attempt is shown as such and is never assigned a zero outcome. Post-hoc supplements remain separate from the primary study.

| Arm | Case | Status | Simulated hours | Conversation | YAML SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| `sp_lense_rank16_all64` | `split_w1_e2` | incomplete | 4.994444444 | [YAML](conversations/sp_lense_rank16_all64/2026-11-25_030155Z--split_w1_e2--attempt-02--incomplete.yaml) | `dd7cf091509ab434fd2864dfcdda081dd76ce5dec61e6308fbdce222d780934a` |
| `qwen38_27b_base` | `split_w3` | incomplete | 13.616666667 | [YAML](conversations/qwen38_27b_base/2026-11-25_033856Z--split_w3--attempt-03--incomplete.yaml) | `8db8d8c2708ae806370614db614c7166b373c2813a975779048c0b6579c30b23` |

## Setup attempts before model calls

No setup failures before model calls were recorded in the structured driver receipts.

Each YAML ends with recorded resource progress. Timing, token usage, endpoint details, and source hashes are outside the conversation files.

Outer multiline YAML values use real line breaks. A literal `\n` inside a JSON-encoded input is part of the exact model-visible text and remains escaped; decoding it would change the recorded prompt.
