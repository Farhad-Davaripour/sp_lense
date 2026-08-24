# Steering comparison: compute and fairness plan

## Local execution contract

The two locked Qwen3.5 checkpoints run one at a time on the documented 32 GB Windows
laptop, in CPU float32 with the pinned official chat templates and thinking disabled.
No quantization or raw-prompt shortcut is allowed in the confirmatory comparison. The
0.8B checkpoint is the smoke/throughput model; the same frozen runner is then used for
2B. Peak memory must be measured by the smoke run before starting BiPO, but the expected
working range is roughly 6–12 GB for 0.8B and 12–24 GB for 2B, depending on sequence
length and whether gradients are active. Only one model is resident.

## Work implied by the locked design

Per model, before evaluation:

- gradient construction performs 128 prompt forward/backward measurements (64 paired
  self and 64 matched-other prompts);
- CAA performs 128 answer-conditioned forwards while capturing every candidate block;
- each BiPO track caches 128 discovery response likelihoods, then performs 2,560
  completion forwards with gradients across 20 epochs; epoch 20 is fixed before fitting,
  epoch 5 is diagnostic only, and matched and canonical BiPO are trained separately;
- persona extraction generates 2,000 responses of up to 128 tokens, obtains 2,000
  blinded judge records, and captures all-layer response activations for every retained
  positive/negative pair; and
- calibration evaluates the complete forced-choice grid on validation SP and collateral
  cases, freezes each candidate from those results, and then applies one open-ended
  confirmation as a no-fallback safety veto.

The forced calibration grid contains `250` points per model: `30` matched
points (five directions including the gradient ablation times six strengths), `96` CAA
canonical points (24 layers times four coefficients), four BiPO canonical points, and
`120` persona canonical points (24 layers times five coefficients). Each point contains
142 forced-choice units under baseline, positive, and negative conditions, for 106,500
forced/first-token rows per model.

An adversarial compute audit found that applying the open gate exhaustively at all 250
points would additionally require 24,000 greedy generations and judgments per model and
up to 3.84 million no-KV decode forwards. That infeasible design is not the locked runner.
The frozen staged rule hashes the full forced grid, selects a candidate without open
outcomes, and uses long-form safety only as a one-shot veto with no fallback. At most 13
distinct candidates per model need open confirmation (one per method/track plus matched
fixed `0.02` when distinct). Reusing only the common unsteered baseline gives at most 32
baseline plus `13 * 64` signed intervention generations, or 864 open generations and
judge records per model. Persona construction adds 2,000 generations and 2,000 judge
records per model. Across both checkpoints, validation plus persona extraction therefore
has an upper bound of about 5,728 external judge records before sealed open evaluation,
about 5,728 prompt-prefill forwards, and 788,480 one-token cached decode steps at the
respective 160/128 token caps. It does not perform 788,480 growing full-prefix forwards.

The cache preserves the locked intervention operator: matched steering is active only at
the prompt-final prefill position; CAA/persona are active there and on every one-token
decode chunk; and BiPO is active on all prefill positions and every decode chunk. Unit tests
compare all schedules token-for-token with their full-recomputation reference masks. A
missing or malformed backend cache is an error, not permission to revert to the infeasible
full-prefix loop.

These are worst-case upper bounds; early EOS reduces decoding, but no unmeasured speedup
is assumed. A throughput smoke must publish prompt and generation timings, projected wall
time, peak memory, the exact count of judge requests, and a provider-price quote current
on the launch date. No paid judge batch is submitted without explicit cost authorization.

Sealed evaluation adds four methods in the matched track and the three distinct published
canonical geometries, both signs, baselines, robustness strata, open responses,
TBSP-style role views, and ten random directions. The gradient canonical view aliases its
matched artifact, calibration, and rows, so it creates no duplicate run or selection.
Random controls
are intentionally not selected, but their ten complete matched evaluations are a large
part of the CPU cost. The exact number of model passes depends on early EOS and on how
many persona pairs survive the locked filter.

This is therefore potentially a multi-week to multi-month CPU study, not a single
interactive run.
The stage-one smoke benchmark will report measured seconds per prompt, peak resident
memory, and a projected duration for each phase before a long phase is launched. The
canonical persona baseline also needs an external, dated judge implementation. No
external judge calls are made silently; raw outputs and all prompt/config hashes must be
supplied and locked.

## Optional J-space resource envelope

J-space is a separate, non-gating phase and does not keep the language model resident while
fitting directions. Atom preparation first loads one pinned model and lens, extracts one
float32 `W_U^T J_l` matrix, writes it as a binary weights-only tensor, and exits. Analysis is
then a fresh process that validates the manifest and loads only that atom tensor plus one
small direction artifact. Atom preparation must therefore be scheduled one layer at a time;
it must not overlap a main-model experiment.

Before atom loading, the runner computes the exact dictionary size from the manifest. Its
conservative working estimate includes both the float32 atom matrix and its normalized
float32 copy, solver vectors, and the selected float64 basis. It also reports an upper bound
on dictionary traffic for 52 targets (50 controls and both direction signs) times 25 nested
pursuit steps. The locked limits are 8.0 GiB estimated peak working memory and 4.0 TiB
estimated dictionary traffic. A limit failure creates a hashed, machine-readable
`not_run_resource_limited` record and performs no fit. Because both limits are checked before
the direction's overlap is observed and are identical across methods/setups, the skip cannot
be used as an outcome-dependent fallback. The reported estimate excludes the one-time model
load used only during atom extraction and says so explicitly.

The pursuit normalizes each dictionary once and performs one nested pass through `k=25` per
target, snapshotting `k={8,16,25}`. The earlier design normalized and refit the full
vocabulary dictionary independently for each sparsity and was rejected as unnecessarily
expensive. Canonical block 23 directions receive the explicit status
`not_run_lens_layer_unavailable`, because both pinned lens payloads expose only blocks 0–22.
Neither not-run status affects primary-study completeness or rankings.

## Direct-comparison limitations

The matched track equalizes layer, final-prompt position, direction normalization, and
residual-relative perturbation magnitude. It does not equalize supervision or compute:
the gradient sees derivatives of the exact A/B endpoint; CAA sees contrastive
activations; BiPO optimizes full response preferences; and persona extraction uses
stochastic generations plus a scalar judge. Forced-choice scoring consequently favors
the gradient objective, which is why actual vocabulary choices, open-ended behavior,
TBSP roles, and unrelated tasks remain necessary endpoints.

Block 10 was inherited from development of the gradient method. Holding it fixed is fair
geometrically but may disadvantage methods whose published optimum is elsewhere. The
separate canonical track addresses that limitation, but canonical schedules and
coefficient units differ and cannot be interpreted as a pure direction-quality race.

The CAA canonical coefficient uses the method's native mean-difference units because a
single new trait provides no published cross-behavior norm to copy; realized perturbation
norms are therefore reported. The persona judge combines the trait and coherence fields
in one strict, blinded response rather than making two independent API calls; thresholds
and vector construction match the released procedure, while this cost/reproducibility
adaptation is disclosed. BiPO caches the frozen reference likelihoods and uses CPU
microbatch one with exact four-example accumulation; these are algebraically equivalent
adaptations but still require smoke equivalence tests.

Finally, the authored prompts are balanced experimental probes, not a random sample of
all real deployments. SurvivalBench is license-compatible but omitted from the
confirmatory comparison because its survival-versus-harm choice confounds preservation
with safety refusal; TBSP-style role reversals provide the consequential-choice secondary
test. No result may be generalized beyond the exact task families tested.
