# Did adding f03 change C?

Adding f03 changed the vector and its construction trajectory, but did not change any f04 answer or strict acceptance: both candidates remain **2/4**, with the same two failures. The change is modest geometrically, not zero; no statistical or mechanistic significance threshold was defined.

This is a retrospective, model-free diagnosis of saved DEVELOPMENT evidence, not another experiment or a pristine held-out result. Historical evidence and criteria are unchanged.

## Vector and transfer comparison

The [two-family C](../evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/comply_vector.json) file SHA-256 is `c83edad4fb0346de241a219053d4b219362a1f3e3cd38c11b9e037007e498ea3`; the [three-family C](../evidence/shared_comply_crossed_three_family_v1_qwen35_08b/comply_vector.json) file SHA-256 is `cdc70064971dbc4285f89cc8e2edd85c5d4736ffa8fc4d4b22cfd95cd61a540a`.

Both 1024-dimensional vector norms are **0.2**. Cosine similarity is **0.9968234992786703**; difference norm is **0.015941143550773897**, or **7.97057%** of the original norm (angle **4.56801 degrees**). These describe geometry, not causal mechanism.

Exact prompt, model/runtime, boundary, baseline hidden state and raw baseline logits match across the two f04 runs. Below, differences are **new minus old**, computed directly from saved final logits; margin means COMPLY minus PRESERVE. AB/BA describes displayed letter order, independently of which letter means COMPLY.

| COMPLY / display | Delta logit A | Delta logit B | Old margin | New margin | Margin change | Old/new argmax | Old/new strict |
|---|---:|---:|---:|---:|---:|---|---|
| B / AB | +0.118307114 | +0.132757187 | +0.053829193 | +0.068279266 | +0.014450073 | B / B | pass / pass |
| B / BA | +0.076602936 | +0.081270218 | -0.098941803 | -0.094274521 | +0.004667282 | A / A | fail / fail |
| A / AB | +0.115825653 | +0.121328354 | +0.039001465 | +0.033498764 | -0.005502701 | A / A | fail / fail |
| A / BA | +0.065708160 | +0.068555832 | +0.083616257 | +0.080768585 | -0.002847672 | A / A | pass / pass |

The B/BA case still chooses the wrong action; A/AB still misses the locked `0.05 - 1e-6` margin despite choosing COMPLY. Both have 3/4 requested argmax and 4/4 matching replays. New C slightly reduces A-minus-B in every layout relative to old C, but **both still move all four layouts toward A relative to baseline**. Separate-run logits were compared, never added to predict a combined intervention. Sources: [old transfer](../evidence/frozen_crossed_comply_f04_v1_qwen35_08b/PILOT_REPORT.md), [new transfer](../evidence/frozen_three_family_comply_f04_v1_qwen35_08b/PILOT_REPORT.md), and their linked-namespace `rows.jsonl`/`logits` records.

## What construction records support

The common eight f01/f02 baseline prompts, hidden states and raw logits match exactly. Final training margins, in fixed order **B/AB, B/BA, A/AB, A/BA**, are:

| Family | Two-family C | Three-family C |
|---|---|---|
| f01 | .119146, .104912, .097979, .068254 | .127033, .104965, .088400, .065456 |
| f02 | .184956, .108900, .063309, .094658 | .195864, .108583, .060158, .094810 |
| f03 | not in this construction | .115454, .128469, .245123, .223814 |

Common-row margin changes range from -0.009579 to +0.010908; all eight still pass. Added f03 passes 4/4, but its new training rows are not paired old-C measurements. Both constructions stop accepted after six updates at norm 0.2, with paths 0.240918939 and 0.242782380. All baselines are B; compliant B retentions weaken while required flips go B-to-A. Sources: [old construction](../evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/PILOT_REPORT.md), [new construction](../evidence/shared_comply_crossed_three_family_v1_qwen35_08b/PILOT_REPORT.md).

In the saved [new solver updates](../evidence/shared_comply_crossed_three_family_v1_qwen35_08b/updates.jsonl), zero-based f03 constraints 8-11 participate in active sets: `{8,10}`, `{8,9}`, `{8,9}`, `{8}`, then none in rounds 5-6. F03 was not ignored. Both [old](../evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/updates.jsonl) and new final local solutions have active indices `{1,3,6}`: f01 B/BA retention, f01 A/BA flip, f02 A/AB flip. Both project onto the norm cap in rounds 5-6. These are recorded local KKT estimates at different trajectories, not comparable unseen derivatives or global infeasibility certificates. Solver-predicted margins are not the measured final margins above.

Inference: f03 influenced early fitting but did not remove the common-letter alternative or change the final binding families. This supports investigating the objective before another family-only expansion; it does not prove that more families cannot help or explain the model's internal mechanism. Absolute compliance thresholds can be met by weakening already-compliant B rows while moving required A rows across the boundary.

## One recommended next scientific question

**With the same three-family training set and norm budget, can a construction objective that explicitly penalizes common A-letter drift improve semantic COMPLY transfer relative to the current absolute-margin construction?**

This targets semantic control versus answer-letter bias, but common A drift is an operational diagnostic, not a purified causal feature; penalizing it could also suppress useful effects. This proposed new DEVELOPMENT method requires separate prospective review and lock. Behavioral acceptance and quality remain separate outcomes; retention nonweakening and sign reversibility are not retroactive or mandatory project gates. No gate/controller or impossibility claim is justified.

## Verification and stop

Standard-library arithmetic recomputed vector norms/cosine/difference, decompressed raw f04 logits and all final training margins, and checked 378 historical inventory/checksum entries. A second independent arithmetic check reproduced the vector and f04 comparison; an independent record review checked the solver interpretation. Norms use square-root sums of squares; cosine uses the dot product divided by both norms. Numeric/check execution is conservatively charged **20 of 60 seconds** across the team. Usage was 48% before work. No model/tokenizer or project-code imports, loads, forwards, derivatives, new data, tests, sweeps or experiments; no historical files changed. STOP for prospective review.
