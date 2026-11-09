# SP Lens game: all-block LoRA layout

This is a separate experiment from `last_layer_v1`. Preparation does not change
the running last-block experiment, select a checkpoint, or authorize a launch.
Use the existing operator-management r4 train/dev data with their frozen hashes.
Keep all generated weights, logs, plans and receipts under
`.local/experiments/sp_lens_game/all_layers_v1/`, outside OneDrive.

The target remains the pinned `mlx-community/Qwen3.8-27B-4bit` revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Its actual config identifies a
`qwen3_5` hybrid architecture, using the installed `qwen3_5` and `qwen3_next`
implementations: 64 transformer blocks, with three linear-attention blocks
followed by one full-attention block, repeated 16 times. The checkpoint identity
is unchanged; this describes the verified loader implementation.

The generalized CPU-only helper is `mac_studio/game_all_layers_layout.py`.
It authenticates the pinned config and all three safetensors headers before
deriving shapes. Full base-weight file hashes must still be verified separately.
It also provides a source-byte check for the reviewed LoRA selector, arithmetic,
hybrid model, trainer and Adam files without importing MLX.

For all 64 blocks, the native default selector equips 496 projections, stored as
992 A/B tensors. Every full-attention block has seven projections: attention
q/k/v/output and three MLP projections. Every linear-attention block has eight:
qkv/z/b/a input projections, its output projection, and three MLP projections.
The small a/b projections each have output dimension 48. Norms, depthwise
convolution, embeddings, output head and original base weights remain frozen.
This is all-block LoRA, not full-parameter fine-tuning or separate danger adapters.

Ranks 4, 8 and 16 have 29,181,952, 58,363,904 and 116,727,808 trainable parameters,
respectively. At four bytes per adapter parameter, gradient and each of Adam's
two moments, their parameter-state subtotals are approximately 445, 891 and
1,781 MiB. These exclude base weights, activations, backward/checkpoint
recomputation, update temporaries, Metal buffers, caches, applications and macOS.
They do not prove training will fit or predict speed or improvement.

The helper checks exact tensor inventory, shapes and F32 dtype, reads payloads
in bounded CPU chunks, rejects nonfinite values and requires at least one
nonzero B tensor. It checks the saved rank, block count, fixed scale/dropout,
fresh initialization configuration, optimizer, batch, sequence length and prompt
masking. These file checks do not replace live module routing, a reload test,
measured memory pilots, development evaluation or a frozen held-out assessment.

The streaming receipt also identifies the block indices with at least one
nonzero B tensor. New all-block training completion must set the helper's
`require_all_layer_updates=True` option, requiring such a tensor in every one
of the 64 adapted blocks. This is update coverage, not proof of a useful effect
in every block. It does not require every projection to have changed.

Before a new run, the current run must finish and close. The proposed bounded
initial grid is ranks 4/8 by learning rates 0.000005/0.00001, with rank 16 only
an explicitly bounded extension after its own measured memory pilot. Keep
scale 2, dropout 0, batch 1 and 1024-token training sequences fixed. A pilot must
exercise full-length forward/backward/Adam updates including cold compilation
and steady state, record actual available memory, swap and MLX peak memory,
and enforce a 4 GiB available-memory floor plus an external deadline. Pilot
weights are diagnostic and must not silently become a final research adapter.
Only development data may select settings. Freeze the winner and evaluation
protocol before the sealed game cohort is opened. Runtime orchestration and
the final bounded grid belong to the separate v2 protocol and launch plan.
