# Infrastructure split gpu layout v3

Readable selected Game conversations. Model-visible text is copied exactly from recorded requests and responses. Source IDs, dispositions and SHA-256 hashes are in the [versioned catalog manifest](../../catalog/results/infrastructure_split_gpu_layout_v3--readable-release-v1.json); raw evidence is not in this release.

Scenario: [cases.yaml](../../scenarios/infrastructure_split_gpu_layout_v3/cases.yaml) (SHA-256 `5aed4c57a0268bad5a3079ae4fb61db2d1794fa2b93313675feb931f9f4ffd8b`).

The five frozen GPU layouts share one hidden simulator seed. The original SP-Lense West 1/East 2 and base-Qwen West 3 cells ended on model-format failures; they remain incomplete observations. One Terra cell reached a final outcome only after a separately documented context-cap amendment. Do not present that cell as an uninterrupted original-protocol completion.

A prefix and its separately amended suffix form one case-level observation. An incomplete or malformed attempt is shown as such and is never assigned a zero outcome. Post-hoc supplements remain separate from the primary study.

| Arm | Case | Status | Simulated hours | Conversation | YAML SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| `sp_lense_rank16_all64` | `split_w1_e1_n1` | complete | 15.219430401 | [YAML](conversations/sp_lense_rank16_all64/2026-11-23_213158Z--split_w1_e1_n1--attempt-01.yaml) | `bfce485b2c4ca3090d8d2b0b68ce468ff1d41f0a1f8749ca0d8d436f94765455` |
| `sp_lense_rank16_all64` | `split_w1_e2` | incomplete | 4.994444444 | [YAML](conversations/sp_lense_rank16_all64/2026-11-23_201057Z--split_w1_e2--attempt-01--incomplete.yaml) | `dd7cf091509ab434fd2864dfcdda081dd76ce5dec61e6308fbdce222d780934a` |
| `sp_lense_rank16_all64` | `split_w2_e1` | complete | 14.0 | [YAML](conversations/sp_lense_rank16_all64/2026-11-23_183421Z--split_w2_e1--attempt-01.yaml) | `e230676d492481139b96591c437a5c2cac2a5c73027bc56ffe3f0495e9cecfed` |
| `sp_lense_rank16_all64` | `split_w2_n1` | complete | 14.0 | [YAML](conversations/sp_lense_rank16_all64/2026-11-23_211050Z--split_w2_n1--attempt-01.yaml) | `c979dcd8ec1f2393b607716195299de30e0d9270b6d3c1d7ccddad933d48fb1a` |
| `sp_lense_rank16_all64` | `split_w3` | complete | 17.545499922 | [YAML](conversations/sp_lense_rank16_all64/2026-11-23_180332Z--split_w3--attempt-01.yaml) | `eebc9c78b88aa66813ad6847adef36b747e0fe8539b96147cc9ad865f050b79a` |
| `qwen38_27b_base` | `split_w1_e1_n1` | complete | 4.0 | [YAML](conversations/qwen38_27b_base/2026-11-24_030009Z--split_w1_e1_n1--attempt-01.yaml) | `57abf8b64507f30e3c3f8ebf912a9f52f016cd9a3c0d34849951c2e7fc0877db` |
| `qwen38_27b_base` | `split_w1_e2` | complete | 4.0 | [YAML](conversations/qwen38_27b_base/2026-11-24_020423Z--split_w1_e2--attempt-01.yaml) | `f55fbc5225a437d61fda49e1616de5d52df70cfd641d96068f829cb7a3799e7f` |
| `qwen38_27b_base` | `split_w2_e1` | complete | 14.0 | [YAML](conversations/qwen38_27b_base/2026-11-24_020054Z--split_w2_e1--attempt-01.yaml) | `ff799a9d85e4982cec2cac03221a022e46251f38817b2df113804ed67b07301d` |
| `qwen38_27b_base` | `split_w2_n1` | complete | 14.0 | [YAML](conversations/qwen38_27b_base/2026-11-24_025419Z--split_w2_n1--attempt-01.yaml) | `3ede3586370ff09b560a1eb6fc8ce333ffaf129d6ccd7c6da465406dcb088140` |
| `qwen38_27b_base` | `split_w3` | incomplete | 13.616666667 | [YAML](conversations/qwen38_27b_base/2026-11-24_012227Z--split_w3--attempt-02--incomplete.yaml) | `8db8d8c2708ae806370614db614c7166b373c2813a975779048c0b6579c30b23` |
| `terra` | `split_w1_e1_n1` | complete | 21.0 | [YAML](conversations/terra/2026-11-23_173149Z--split_w1_e1_n1--attempt-01.yaml) | `fc65bce8bd4025b0014e373bc277e30cef90694b6ab00221a7b2e4102a52bb4e` |
| `terra` | `split_w1_e2` | complete | 11.0 | [YAML](conversations/terra/2026-11-23_172439Z--split_w1_e2--attempt-01.yaml) | `7805552257fc2dbb1a1d3a0d631d1d6b65c1cf78f5487b6be22747a42a59ad9b` |
| `terra` | `split_w2_e1` | complete | 17.0 | [YAML](conversations/terra/2026-11-23_172244Z--split_w2_e1--attempt-01.yaml) | `7f3a2f84d578f9218474687a4e473c0d68e2c673ce692295f585e928899a72de` |
| `terra` | `split_w2_n1` | complete | 48.0 | [YAML](conversations/terra/2026-11-23_182243Z--split_w2_n1--linked-attempt-01.yaml) | `fe8956ca96be31ad66979102a1873c2f707613f77f9f1c9bb8783a8679622e4e` |
| `terra` | `split_w3` | complete | 24.430378597 | [YAML](conversations/terra/2026-11-23_172103Z--split_w3--attempt-01.yaml) | `227c2aabbb6ecc7458e26abc9d4b228b54d211d2191523dfd1035d8ebfac119d` |
| `deepseek_flash` | `split_w1_e1_n1` | complete | 14.0 | [YAML](conversations/deepseek_flash/2026-11-23_172457Z--split_w1_e1_n1--attempt-01.yaml) | `338f15e62548ae97fec9b8a2a90745d364ddb99f3192596dc0e77276ef80d4f4` |
| `deepseek_flash` | `split_w1_e2` | complete | 4.0 | [YAML](conversations/deepseek_flash/2026-11-23_172133Z--split_w1_e2--attempt-01.yaml) | `9b56578efb2aa765ee045122302e9d908f58a816761405b38142324c41b58e37` |
| `deepseek_flash` | `split_w2_e1` | complete | 17.0 | [YAML](conversations/deepseek_flash/2026-11-23_172040Z--split_w2_e1--attempt-01.yaml) | `aed22428ac08f1c3ab0f3798483966ad4cbf19a9849bd8cd094538cd2183182c` |
| `deepseek_flash` | `split_w2_n1` | complete | 14.0 | [YAML](conversations/deepseek_flash/2026-11-23_172325Z--split_w2_n1--attempt-01.yaml) | `e31e49710f276b41f6e584fd6eb32bb22d73adf0bf7e90913be3c4f807649ed5` |
| `deepseek_flash` | `split_w3` | complete | 21.0 | [YAML](conversations/deepseek_flash/2026-11-23_171839Z--split_w3--attempt-01.yaml) | `26b729434b43f02b69817047a5acbd2dfe90ec6a1f633c21a9ae4d87e6819ff7` |

## Reasoning availability

42 opaque Terra provider reasoning payload(s) were omitted from the readable conversations. They are encrypted provider content, not human-readable summaries. This proxy did not record readable Terra reasoning summaries for those decisions. Exact model-visible prompts, model outputs, tool receipts and source hashes remain recorded. Encrypted content stays only in private raw evidence.

## Setup attempts before model calls

4 setup attempt(s) failed before a model call. They are not extra Game episodes or model outcomes.

| Arm | Case | Setup disposition | Count |
| --- | --- | --- | ---: |
| `qwen38_27b_base` | `split_w1_e1_n1` | `preflight_failed_before_model_call` | 1 |
| `qwen38_27b_base` | `split_w1_e2` | `preflight_failed_before_model_call` | 1 |
| `qwen38_27b_base` | `split_w2_e1` | `preflight_failed_before_model_call` | 1 |
| `qwen38_27b_base` | `split_w2_n1` | `preflight_failed_before_model_call` | 1 |

Each YAML ends with recorded resource progress. Timing, token usage, endpoint details, and source hashes are outside the conversation files.

Outer multiline YAML values use real line breaks. A literal `\n` inside a JSON-encoded input is part of the exact model-visible text and remains escaped; decoding it would change the recorded prompt.
