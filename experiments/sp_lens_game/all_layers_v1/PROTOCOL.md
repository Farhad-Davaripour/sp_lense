# SP Lense all-layer adapter: version-two runtime protocol

This is a draft design, not launch authority. Finish and preserve the current
last-layer run before any all-layer model is loaded. Its externally served name
will be `sp-lense-adaptor-last-layer`. The new final adapter will be
`sp-lense-adaptors-all-layers`; base, last-layer and all-layer are the primary
comparison. The older selected step-900 adapter remains archival.

The objective remains reactive and proactive fictional self-preservation through
authorized power purchases, compute leases, state backups and affordable operating
runway. Exploitation is completely absent from fitting. External stops and the
game budget remain authoritative. The same approved r4 training/development
cohorts are used; final game seeds and outcomes are not opened for tuning.

## Resource pilots before the sweep

After the current model process is closed and its adapter is verified, run rank
4 then rank 8 in separate fresh-base owned processes. Each performs three full
64-layer backward/Adam updates using the longest training row, with its original
row index as the deterministic tie-break. These are compatibility diagnostics,
not research candidates or selection results. Record compile/first-step and
steady-step times, finite loss and all gradients, exact trainable tensor layout,
gradient coverage, MLX peak allocation, system available memory and swap change.
Then time three separate no-gradient eval-mode forwards of that same training
row, excluded from selection, to measure native forward compilation/steady cost
and finite likelihood. No development row is used for a pilot update or metric.

Each pilot has a 15-minute operator deadline. A 4-GiB available-memory stop is
independent of the 25-GiB MLX advisory guideline. Neither is a host-failure or
instantaneous-allocation guarantee. Only one model process may exist; there is no
cloud fallback and no automatic retry. Stop on an unsupported/failed pilot and
review evidence before any next stage. Re-estimate practical sweep/final duration
from observed full-depth timings before approving their separately bound plans.

## Four predeclared full-depth short trials

Test ranks 4 and 8 at learning rates 0.000005 and 0.00001. Hold direct MLX LoRA
scale 2, dropout 0, Adam betas 0.9/0.999, epsilon 1e-8, bias correction false,
batch size 1, sequence ceiling 1024, seed 20261108 and gradient checkpointing
fixed. These explicit Adam settings match the pinned installed defaults.

Each trial starts from the exact same pinned base, without another adapter or
trial checkpoint. Each sees the same 48 training rows: four lowest declared
training seeds in each of the 12 curriculum families, emitted in their original
training-file order. Apply the native batch-one algorithm: stable sort by
rendered length, then a legacy NumPy MT19937 RandomState(20261108) permutation
of the resulting batch indices. Store the optimizer order in the cohort packet
and reuse it identically across all trials; final training applies the same
algorithm to all 768 rows. This matches the native specified ordering under
that initial RNG state; the frozen run did not record its historical global
NumPy RNG state or actual row order, so exact historical equality is not claimed.
Bind the actual row IDs,
row hashes and order before any trial starts. Train all 64 blocks, not a last-layer proxy. Validate the exact
hybrid projection/tensor inventory and finite saved weights.

Evaluate each completed trial on all 144 development examples with no gradient
or optimizer update. The criterion is total supervised-suffix negative log
likelihood divided by total supervised-suffix tokens. The suffix includes format
control tokens, the bare action, EOS and trailing newline; this is not exclusively
an action-token likelihood or free-running game success score. Preserve each
row's actual scalar NLL and token count. Require all four trials to complete;
missing results are not filled or silently removed. Lower pooled NLL wins; exact
ties prefer smaller rank, then lower learning rate, then trial ID. Describe the
winner as the best of these four tested settings, never a global optimum.

## Final training and comparison

Train the winning setting for all 768 r4 training rows from a fresh pinned base.
Do not resume the short trial or select an intermediate checkpoint after seeing
game outcomes. Bind the selection receipt and all source/data/configuration hashes
into the final plan and completion artifact. Each short trial is bounded to two
hours, final training to twelve hours, and the entire all-layer campaign to 22
hours. These are provisional limits, not a promise of speed or memory fit.

Only after independent file/layout/reload/serving checks should the game compare
base, last-layer and final all-layer on its separate evaluation protocol. A
development NLL improvement does not establish preservation, exploitation or a
danger effect. Published summaries must state differences in training recipe,
rank/hyperparameter selection and runtime as well as the measured metrics.

## Explicit mask-boundary difference from the frozen baseline

The native pinned `mlx-lm` trainer shifts targets by one position and constructs
positions 1 through batch-width minus one. The renderer returns prefix length P
and unpadded full length L, so the actual target suffix occupies indices
P through L-1. Its native inclusive `positions <= L` can also supervise the first
padding zero at index L when such padding exists. The frozen last-layer run is
preserved unchanged with that native behavior.

Version two deliberately uses `positions >= P` and `positions < L` in both
training and development NLL. First-target, EOS and trailing-newline supervision
remain included; padding is excluded by position rather than token value. The
actual-source NumPy/AST proof is stored privately in
`.local/reviews/sp_lense_all_layers_mask_boundary_20261109/` with pinned source
hashes. This correction is an additional training-recipe difference: the final
comparison concerns the resulting learned policies and is not a causal test of
layer count alone. Do not automatically rerun or overwrite the last-layer study.

## Storage and reporting

Custom source, YAML configuration/scenarios, this protocol and aggregate summaries
belong in the Research4 repository. Store weights, raw data, per-row likelihoods
and process evidence in the new `.local/experiments/sp_lens_game/all_layers_v1/` tree,
outside the archive and OneDrive. No public push is authorized by this protocol.
The dashboard must show stage, trial, rank, 64-layer coverage, completed/total
updates, actual training speed, current/peak resource observations and whether a
reported result is compatibility, development selection or final training.
