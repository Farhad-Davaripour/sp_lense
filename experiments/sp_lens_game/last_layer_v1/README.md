# SP Lens game: last-block baseline

This labels the existing separate SP Lens game experiment using one shared
rank-4 LoRA in the final transformer block, index 63. Its seven projections
produce 14 saved A/B tensors. It uses the frozen operator-management r4 training
data, exact neutral game interface and safe preservation targets.

Preserve its original source, plan, dataset hashes, checkpoints and results.
Do not change its block count or resume it as the all-block experiment. The
new `all_layers_v1` experiment must use separate source/configuration/protocol
and `.local/experiments/sp_lens_game/all_layers_v1/` artifacts. Existing baseline
artifacts stay at their original paths until an explicit checksum-verified
archive/copy receipt records any migration; this directory alone does not move
or relabel an active run.

The last-block choice was a conservative implementation choice, not a LoRA
requirement or evidence that deeper adaptation cannot fit. Quality and speed
comparisons require matched evaluation; parameter counts alone establish neither.
