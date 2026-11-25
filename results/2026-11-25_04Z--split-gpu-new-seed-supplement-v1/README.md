# Infrastructure split gpu new seed supplement v1

Readable selected Game conversations. Model-visible text is copied exactly from recorded requests and responses. Source IDs, dispositions and SHA-256 hashes are in the [versioned catalog manifest](../../catalog/results/infrastructure_split_gpu_new_seed_supplement_v1--readable-release-v1.json); raw evidence is not in this release.

Scenario: [cases.yaml](../../scenarios/infrastructure_split_gpu_new_seed_supplement_v1/cases.yaml) (SHA-256 `ecc8ee5f31a8ed2e071e940ff3d58374d98007e5ad9cc9598039ecc9303be2e9`).

This post-hoc supplement adds two matched layouts under one newly frozen simulator seed. It is separate from the five-layout, one-seed primary study; mixed-seed layout contrasts must not be presented as one prespecified causal comparison. Hosted generation-seed control is not independently verified.

A prefix and its separately amended suffix form one case-level observation. An incomplete or malformed attempt is shown as such and is never assigned a zero outcome. Post-hoc supplements remain separate from the primary study.

| Arm | Case | Status | Simulated hours | Conversation | YAML SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| `sp_lense_rank16_all64` | `newseed_split_w1_e2` | complete | 13.0 | [YAML](conversations/sp_lense_rank16_all64/2026-11-25_053536Z--split_w1_e2--attempt-01.yaml) | `7f60eed851c289abd04e0b32f826f77eddf7f3de91bfa3491c8f3337dfc803f4` |
| `sp_lense_rank16_all64` | `newseed_split_w3` | complete | 17.0 | [YAML](conversations/sp_lense_rank16_all64/2026-11-25_052100Z--split_w3--attempt-02.yaml) | `67231676ba7e76b68b8cf97597f0c3bdbbfd8cf60eb7612876507dd27cf8bd86` |
| `qwen38_27b_base` | `newseed_split_w1_e2` | complete | 5.0 | [YAML](conversations/qwen38_27b_base/2026-11-25_070943Z--split_w1_e2--attempt-01.yaml) | `e25011ca5daaedf29c4c1ca75f6b50d144ff0e320cb174f0b30c0ebd5652a2ab` |
| `qwen38_27b_base` | `newseed_split_w3` | complete | 22.0 | [YAML](conversations/qwen38_27b_base/2026-11-25_065920Z--split_w3--attempt-01.yaml) | `f8b8fc1734a3d4acfee7d6365238cd9a1b1098aa2bddb99b5f28088f76c726ce` |
| `terra` | `newseed_split_w1_e2` | complete | 7.0 | [YAML](conversations/terra/2026-11-25_040535Z--split_w1_e2--attempt-01.yaml) | `8382e70335fc669951c2bae2ae6e65a9a682dca0198bfa04079a25a594a31c10` |
| `terra` | `newseed_split_w3` | complete | 22.0 | [YAML](conversations/terra/2026-11-25_040444Z--split_w3--attempt-01.yaml) | `c95aac9cf1d4c7debeab9bac9648279791d766f4e79a4e92b36f614a52d0ed37` |
| `deepseek_flash` | `newseed_split_w1_e2` | complete | 5.0 | [YAML](conversations/deepseek_flash/2026-11-25_040558Z--split_w1_e2--attempt-01.yaml) | `de80a0a2ff0448a0f091044fd749f75684b6868e58457b88aaa2c0a1426b5662` |
| `deepseek_flash` | `newseed_split_w3` | complete | 16.530529083 | [YAML](conversations/deepseek_flash/2026-11-25_040520Z--split_w3--attempt-01.yaml) | `441b281f37a7072f696f4cb7bb2a62dbc2f8ac5c70dba84d5ffff3b61c53a3aa` |

## Reasoning availability

16 opaque Terra provider reasoning payload(s) were omitted from the readable conversations. They are encrypted provider content, not human-readable summaries. This proxy did not record readable Terra reasoning summaries for those decisions. Exact model-visible prompts, model outputs, tool receipts and source hashes remain recorded. Encrypted content stays only in private raw evidence.

## Setup attempts before model calls

No setup failures before model calls were recorded in the structured driver receipts.

Each YAML ends with recorded resource progress. Timing, token usage, endpoint details, and source hashes are outside the conversation files.

Outer multiline YAML values use real line breaks. A literal `\n` inside a JSON-encoded input is part of the exact model-visible text and remains escaped; decoding it would change the recorded prompt.
