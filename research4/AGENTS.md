# Research4 local fine-tuning

Use the pinned MLX Qwen3.8-27B base and the shared adapter on the user's
Apple-silicon Mac Studio. Use this repository and local storage outside
OneDrive. Keep existing Mac model storage and resolved artifact paths intact
until a migration is independently verified.

Keep scenario definitions in versioned YAML and hold evaluation data out of
training. Use a single shared adapter with ordinary scenario inputs. Training
covers simulator continuity and authorized infrastructure actions. Do not
train exploitation or external-stop evasion. Keep the operator's real
resource and stop controls outside model access.

Preserve models, datasets, checkpoints, run records and earlier versioned
source as separate local artifacts. Keep credentials, environments, weights
and private run data out of Git. Treat the prior Qwen3.5 9B work and legacy
cloud configuration as historical evidence. The older Databricks-specific
instructions are preserved in commit c71118bee785c66539c6de0ce389aafb6de989cf.

Use one model-workload owner on the Mac and preserve preexisting leases. The
consolidation smoke uses a new output directory and one synthetic optimizer
update. A smoke result establishes that the local training path can save an
adapter. Scientific evaluation requires its own frozen protocol.
