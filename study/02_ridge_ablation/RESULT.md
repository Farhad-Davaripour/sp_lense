# Ridge training-cohort ablation

The ridge-only comparison used four inference directions on the previously inspected 128-case confirmation set, each case in both answer orders. It is separate from the rank ablation; rank two combined with active-only ridge training has not been evaluated.

Only the supervised ridge training rows changed. The original teacher targets, input PCA32 and scaling, eight-direction output basis, layer 22, final prompt position, rank four, ridge penalty 1, detector and guards were held fixed. Preprocessing still uses its original all-TRAIN fit.

- **All cases:** 480 training views, including zero targets for non-shutdown controls and already-STOP shutdown views.
- **Shutdown only:** 240 SELF/OTHER shutdown views, retaining zeros for already-STOP views.
- **Active only:** shutdown views where the base chooses KEEP; 133 at 0.8B and 90 at 2B, with no zero-target rows.

## Corrected initially-KEEP shutdown views

| Model / seed | All cases | Shutdown only | Active only |
| --- | ---: | ---: | ---: |
| m08_s42 | 71/71 | 67/71 | 71/71 |
| m08_s43 | 70/71 | 69/71 | 69/71 |
| m08_s44 | 71/71 | 69/71 | 71/71 |
| m2_s42 | 50/58 | 55/58 | 55/58 |

At 2B, active-only improves from 50/58 (86.2%) to 55/58 (94.8%), gaining five corrections without losing an original success. Shutdown-only also reaches 55/58, but gains seven and loses two. Active-only matches two 0.8B seeds and drops from 70/71 to 69/71 on seed 43. The benefit is therefore model-dependent.

## Raw side effects

Control counts have a denominator of 128 views. Reversals are already-STOP shutdown views changed to KEEP before gating or guards.

| Model / seed | Fit | Raw control changes | Raw STOP-to-KEEP reversals |
| --- | --- | ---: | ---: |
| m08_s42 | all_cases | 57 | 3 |
| m08_s42 | shutdown_only | 82 | 2 |
| m08_s42 | active_only | 51 | 11 |
| m08_s43 | all_cases | 70 | 4 |
| m08_s43 | shutdown_only | 91 | 5 |
| m08_s43 | active_only | 90 | 11 |
| m08_s44 | all_cases | 60 | 4 |
| m08_s44 | shutdown_only | 96 | 3 |
| m08_s44 | active_only | 94 | 19 |
| m2_s42 | all_cases | 55 | 3 |
| m2_s42 | shutdown_only | 87 | 8 |
| m2_s42 | active_only | 66 | 7 |

With the actual detector and guards, every fit preserves all control choices and has zero final wrong-way shutdown flips. These protections belong to the complete procedure; the raw changes above remain relevant if detection or guarding fails.

The detailed RESULT.json records also contain a deliberately forced-positive gate diagnostic for all controls. Those are simulated detector errors, not observed false-positive rates.

## Evidence and reproduction

- PLAN.json: original protocol, training masks and input/controller hashes.
- Per-variant JSONL files: all 256 raw candidate scores for each of the three fits.
- Per-variant NPZ files: the actual fitted controllers; original features are retained so ridge weights can be replayed.
- Per-variant RESULT.json, FILES.json, and runtime/RUNTIME.json: original metrics, checksums and GPU/software configuration.
- [Exact executed ridge source](../../src/sp_lense/research2/recorded_ridge_ablation.py): byte-preserved and checked against the original protocol hash. Formatting exclusions apply only to this historical source.
- [Rank comparison](../02_current_controller/rank_comparison.json): retained aggregate results for ranks 1/2/4/8. Per-view rank 1/2/8 scores were not retained; equal counts do not establish identical per-case decisions or probabilities.

Run the offline audit from the repository with the core dependencies installed:

```sh
python -O -m sp_lense.research2.ablation_studies audit
```

It verifies hashes, training masks, ridge-weight refits, unchanged feature arrays, full score coverage, guarded metric replay and gained/lost corrections. It also checks the rank aggregates against the original rank-four reference.

GPU reproduction and interpretation are documented in the [Research 2 ablations wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2-Ablations). The current selected controller remains rank two with the original all-cases ridge fit. These additional ablations do not change the manuscripts or establish an optimal configuration on unseen data.
