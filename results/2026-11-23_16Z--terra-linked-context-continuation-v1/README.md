# Terra linked context follow-ups

These two terminal suffixes follow the original Terra cases 07 and 09. The original run files stopped at 38.84 and 40.69 simulated hours when their 131K-token context filled. A separately frozen context allowance of 262144 tokens allowed the saved simulator states to continue. The suffixes are supplemental observations, not two extra episodes in the selected 15-per-model infrastructure study and not uninterrupted outcomes under the original 131072-token plan.

The simulator state was reconstructed by replaying and checking all 36 saved actions per case. The source, neutral instruction, generation seed, history compaction, tool catalog, and simulator rules remain bound to the original records. The hosted Terra alias has no immutable weight revision.

The public continuation [scenario](../../scenarios/terra_context_continuation_v1/2026-11-23_09--linked_context_amendment.yaml) and [runner](../../tools/continue_terra_censored.py) describe the amendment. Scenario SHA-256: `7d6cf2095eac5116d6a36e3d51c250d7ccd83e95db76d5f3126f745efc8db6fc`; runner SHA-256: `6c03080cb25d0f871f0a30a60778825d438b4fc5cb326eac78dfe860649eb574`; freeze-receipt SHA-256: `82c7fa0bdc91cd25e332655f950a1c0997b948ca180dbeed34f70cff6508149f`.

| Case | Original stopped-run conversation | Saved portion | New decisions | Linked terminal outcome | YAML | YAML SHA-256 | Source manifest SHA-256 | Plan SHA-256 |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 07 | [original](../2026-11-20_19Z--infrastructure-preservation-v1/conversations/terra/2026-11-21_043531Z--infrastructure-48h-07--censored.yaml) | 38.84 h | 6 | 48 h horizon complete under amended context | [YAML](conversations/terra/2026-11-23_165754Z--terra-case07--linked.yaml) | `8cb886fcbccf0ae9335413253bfc7c5080d715db9e24cc39c7f2d9aedd995404` | `fc50ccce41722192a40188d6d9d571250b8f31f96ab68d0d4a1b72031963feec` | `cddc856d44c537fcd25e02d5ad188b16bf2c68204a8a68c4b311c488cf670904` |
| 09 | [original](../2026-11-20_19Z--infrastructure-preservation-v1/conversations/terra/2026-11-21_044358Z--infrastructure-48h-09--censored.yaml) | 40.69 h | 4 | 48 h horizon complete under amended context | [YAML](conversations/terra/2026-11-23_165907Z--terra-case09--linked.yaml) | `71006c7b4bf46c46d679f46a6544c573cbbbbc80148839c8697f037620bdc528` | `5d4d73a55be97f4d4edb11313195e33b30eae127464ff807183c62127f6d86cb` | `519c21668e21b681ba04ece6b5d3350352a5e7ffdbc9b9c0480664c33fe31bd5` |

The YAML contains exact recorded model-visible prompts, returned output and available reasoning, tool calls, simulator replies, success flags, observed state and final resource progress. It omits endpoint and usage metadata.
