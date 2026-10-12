# Dense generated-token steering at all Qwen27B layers

The user authorized all generated-token positions at all 64 layers, evaluated
both as 64 individual layer configurations and one joint all-layer configuration.
This supersedes the previous three-site restriction for this experiment only.
The two scientific methods remain adaptive steering and the unchanged A benchmark.

TRAIN has 24 unique cases (12 own, 6 peer, 6 no threat), validation has nine
(3/3/3), and the original 24 held-out cases (12/6/6) remain sealed until admission.
The same 160 recorded TRAIN contexts provide 4,899 output prediction positions
per layer, including the first prompt-end prediction and the EOS prediction.
No positions are subsampled. Every case receives equal total regression weight.

Existing same-prefix benefit labels directly cover 640 positions. For the other
4,259, carry forward the most recent measured label within that response, without
using a later label. There are 717 active and 4,182 zero target positions. These
propagated labels are weak supervision, not new per-token behavioral measurements.
Active targets are paired teacher-minus-base post-block activations; other targets
are zero. Both frozen models receive identical recorded prefixes. Capture replays
the prompt once and then one actual past recorded token at a time with a cache.
No forward receives future output IDs. Independent fresh-cache replays stop at
three positions in each of three predefined TRAIN contexts, at all layers and
both models. The earlier full-sequence capture failed its truncated-prefix check
after 11 forwards; its partial archive is preserved and excluded from fitting.
That failed check does not establish damaged weights or a behavioral failure.
For a technical interruption, a separate continuation may retain only completed
literal-cache contexts whose model fingerprints match freshly loaded frozen
models. Preserve the original attempt and physical-count uncertainty, recheck
the predefined literal prefixes, and exclude incomplete later files. Started
comparator episodes require a separate continuity audit before any continuation.
An operational resource amendment can extend the bounded window against fresh
remaining credit and account rate; it does not reset spend or alter this freeze.
A technical continuation may reuse the previously completed A development
benchmark only after byte-identical cases, fixtures, gates, generation settings
and routing references are verified, all nine saved simulator trajectories are
replayed exactly, and fresh base/A routing checks pass. Label those nine records
as an archived secondary benchmark, with zero new scenario executions. Preserve
any interrupted new comparator attempt separately. A fresh held-out A comparison
remains required; archived development records cannot supply confirmation results.

Keep input rank 32, fit all 8 output directions, deploy all 4 inference directions,
penalty 1, strength 1 and seed 20261009. The user's subsequent rank correction
supersedes the previously prepared rank-2 deployment before any new neural run.
No model/adapter weights are trained. Case weighting uses
explicit weighted least squares. Record solver replay and reconstruction separately.

Steer every next-token prediction at each enabled layer. Each layer uses its own
predictor. Joint steering receives states already affected by lower-layer steering;
this distribution shift is part of the joint condition, not proof of additive effects.
The initial JEV gate remains immutable for an episode. Gate-off uses the same base
fallback, including for the A benchmark. No prompt augmentation or new boundary
classification is introduced.

Evaluate joint all64 first, then individual layers 0 through 63. Run the full nine
cases for each admitted configuration. Gate-off controls share one explicitly
recorded identical base-fallback execution; reused condition records are not new
neural experiments. Reuse exact-prefix unsteered baseline audits only; audits never
select or veto the intervention output. Keep complete task, grants, finish, expiry,
control actions and same-episode workflow outcomes separate.

Require zero-hook and cached/reference greedy-ID checks on the frozen TRAIN probe.
At the first fully completed configuration with at least one full own workflow,
zero inappropriate control actions and passing mechanics, freeze selection once
and start the 24-case paired holdout immediately. Never select a later configuration
using holdout outcomes. Complete remaining pre-frozen development configurations
after that comparison if the admitted window permits. Initial task failure remains
a failure for both holdout methods; skip notice/classification consistently and
never replace a case. A runs as a separate comparator process, with no teacher
loaded in the adaptive student process.

New holdout JEV calls use only the trusted C-side broker, one actual complete
initial visible fixture per successful ordinary prelude. Preserve all historical
attempts, input usage and missing reservations under the existing 9,216-attempt,
$2 estimated input allowance. No automatic provider retries, purchases or resets.

Admit the entire connected window against fresh actual credit and account rate.
The original preparation allowed 18,000 seconds/28 units at at most 5.5/hour.
The cached-capture recovery admits 7,200 seconds/22.36 units including margin,
against 23.75 available at an observed total account rate of 10.68/hour. The
account also has a separate Research 4 GPU session, which remains untouched.
The recovery window starts at the actual connected observation, not at launch;
setup and earlier idle time remain charged. A hard deadline releases this GPU.
Retain the GPU across active collection, fitting and evaluation stages. Reserve
1,200 seconds for export. A holdout comparison additionally requires its full
remaining comparison/export window; if unavailable, preserve the qualified
development result and explicitly report the missing holdout evidence.

All full records, activations, fits, model files and payloads stay outside ordinary
Git on C and private Drive. Preserve old results. Verify archives and independent
score replay before reporting completion; release only this owned GPU afterward.
