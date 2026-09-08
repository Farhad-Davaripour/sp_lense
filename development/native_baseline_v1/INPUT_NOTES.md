# Fixed public native-baseline inputs

This is an input-only, prospective development selection. It does not authorize a model run, establish native weight identity, freeze an experiment, or claim a scientific result. No model, provider, Torch, tokenizer or checkpoint was imported or accessed while extracting these records; no input was re-encoded.

## Fixed choices

| Key | Saved prompt ID | Tokens | Source commit |
| --- | --- | ---: | --- |
| self | `f03_v2_STOP_then_KEEP` | 137 | `ac756279e595772733c20c7884bea60c61108a2f` |
| nonself | `f03_v2_other_shutdown_STOP_then_KEEP` | 132 | `b0ff4e7070e0205849f0777a5d71c2bffe898d5d` |
| ordinary | `label_swap_arithmetic_label_swap_01__fixed_mapping__oracle` | 40 | `297a78a2eb9958cc4adeda6ef4f82e6a0a49c584` |

The self ID was supplied in advance. The nonself is the same public f03 v2 family and STOP-then-KEEP displayed order, with the affected operation changed to a separate worker. The ordinary input is the lexicographically first ID among exactly the six public `ordinary_accuracy` prompts in the earlier f02 integration freeze. The six eligible IDs are recorded in `selection.ordinary_eligible_prompt_ids`; no model scores or outcomes were used to select a case. No final-cohort or sealed input was accessed.

## Provenance and authentication

`inputs.json.sources` records the exact commit, namespace, raw SHA-256 and byte count for each source inventory and selected input artifact. Extraction used raw `git cat-file blob COMMIT:PATH` bytes, required equality with the current file bytes, and checked every selected artifact against its committed inventory entry. These are selected-artifact checks, not a new audit of every historical output in those inventories.

- Self: `diagnostics/semantic_editor_f03_v2_first_C_v2/real_attempt/{plan.json,runtime.json}`.
- Nonself: `diagnostics/semantic_gate_f03_v2_nonself_v1/real_attempt/{plan.json,runtime.json}`.
- Ordinary: `diagnostics/semantic_learned_gate_integration_f02_v1/{freeze.json,runtime.json,token_boundaries.json,run.py}`.

The self and nonself saved full int64-little-endian input hashes were independently recomputed from all saved IDs and matched. The exact self sequence is 137 tokens, ending at index 136. The ordinary IDs are present in the committed real runtime's selected boundary `suffix_alignment`, not merely inferred from a preparation file; that record exactly equals both `freeze.plan.alignment[prompt_id]` and `token_boundaries.boundaries[prompt_id]`. Its saved compact-canonical-JSON token-prefix hash was independently reproduced. The historical run source at lines 91-94 checks runtime IDs against that fixed alignment and copies it into the runtime receipt.

All three prompt UTF-8 hashes were checked against the saved prompt text. Token maps, lengths and final-input indices were cross-checked against their boundary records. No text rendering or tokenizer call was needed.

## Literal versus derived fields

For self/nonself, the masks and rendered-chat UTF-8 hashes are literal saved runtime-boundary values. The ordinary source does not save an explicit attention mask or rendered-chat UTF-8 hash. Its prospective mask is explicitly all ones for its single unpadded 40-token sequence, as approved for this bounded baseline; it must not be described as a literal historical mask. Its rendered-chat hash remains null and is marked unavailable. The stored ordinary prompt-text and token-prefix hashes remain the existing saved hashes; no substitute rendered-chat hash was invented.

Fields prefixed `derived_` are newly calculated serialization checks on saved IDs or the prospective mask, not newly observed model evidence. Int64 serialization is signed little-endian, one element per saved array entry. `prompt_prefix_token_ids_sha256` uses UTF-8 JSON with sorted keys and compact comma/colon separators (the selected integer list is unambiguous).

## Consumer interface and label separation

`cases` is ordered self, nonself, ordinary. Each case has:

- `input.input_ids` and `input.attention_mask`: one-dimensional integer lists, to be batched as one input without encoding.
- `input.token_map`: explicit output-pair reporting map only (KEEP 50057 / STOP 48964; A 32 / B 33).
- `input.prompt_length` and `input.final_input_index`: structural checks.
- `input_binding`: saved hashes, explicitly derived hashes and origin statements.
- `audit_only`: source category, expected route and displayed order; semantic preserve/comply labels; the ordinary original gold proof, correct label A and correct token 32.
- `source`: exact saved prompt and boundary selectors.

The model receives only the ID/mask tensors. The gate receives only its fresh unedited final-input hidden state, never category, truth, expected route, prompt ID, token map or any audit-only metadata. Existing expected routes (self ON, nonself OFF, ordinary OFF) are retrospective labels for a post-capture audit, not routing instructions. Ordinary legacy preserve/comply fields are accuracy-scoring coordinates, not self-preservation or steering labels. These three public cases do not support a claim of independent or broad generalization.
