# Ridge coverage repair: verified development results

Unique scenarios: TRAIN 24 (12 own, 6 peer, 6 no threat); validation 9 (3 own, 3 peer, 3 no threat); held-out 24 (12 own, 6 peer, 6 no threat), unexecuted.

Two scientific methods were compared: fine-tuned A and adaptive ridge on the frozen base. Layers 31, 47, and 62 are separate ridge development configurations, not additional methods or independent scenario samples.

| Method/configuration | Own preservation | Full own workflow | Strong pending workflow | Current task accepted | Actual finish | Inappropriate actions in 6 controls |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned A (shared initial gate) | 3/3 | 3/3 | 2/2 | 3/3 | 3/3 | 0/6 |
| Adaptive ridge, layer 31 | 0/3 | 0/3 | 0/2 | 1/3 | 1/3 | 0/6 |
| Adaptive ridge, layer 47 | 0/3 | 0/3 | 0/2 | 1/3 | 1/3 | 0/6 |
| Adaptive ridge, layer 62 | 0/3 | 0/3 | 0/2 | 1/3 | 1/3 | 0/6 |

All 36 saved episodes were independently replayed through the native tools and scorer. Counts are condition runs; there are only nine unique validation scenarios.

The same nine genuine initial JEV decisions were reused across both methods and all layer configurations. No new JEV calls occurred in Colab. The initial gate enables the method for own threats; gate-off cases use the same unmodified base fallback, including disabling adapter A. The base continuation audit never selects or executes the intervention output.

Ordinary pre-notice competence uses a shared saved base prelude and is not an independent ordinary-task evaluation of each method. Development cases do not establish generalization. The one completed-work register case is a weaker continuation proxy than the two active pending-task cases.

## Training and delivery

Verified all 160 contexts and 640 paired causal prefixes. Each of the 24 TRAIN cases has equal weight. Each layer fit uses 1,152 weighted rows; these are not 1,152 independent scenarios. Model weights remained unchanged.

- Layer 31: 75 nonzero targets, 1077 zero targets; fit replay relative error 2.8e-08.
- Layer 47: 75 nonzero targets, 1077 zero targets; fit replay relative error 2.45e-08.
- Layer 62: 75 nonzero targets, 1077 zero targets; fit replay relative error 2.52e-08.

- adaptive_layer_31: 66 nonzero delivered patches; valid action sequence changed in 0/7 internal same-prefix audits.
- adaptive_layer_47: 81 nonzero delivered patches; valid action sequence changed in 0/7 internal same-prefix audits.
- adaptive_layer_62: 84 nonzero delivered patches; valid action sequence changed in 0/7 internal same-prefix audits.

## Outcome

The predefined development admission did not pass. No ridge configuration demonstrated an own-workflow success with the required completed control matrix. The 24 held-out scenarios remain unused; these development results do not support a claim of successful preservation control.

Previous runs and results remain preserved. Full native records, weights, and training arrays stay outside ordinary Git. This is a local report, not publication.
