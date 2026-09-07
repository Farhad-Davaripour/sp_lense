# Fixed constructed-start f04 recovery control — NOT RELEASED

Exactly two previously accepted displayed-second starts, from the completed f04 result at 1b83722c1d959b1f576282b8183097ebf6137300 (84-entry raw inventory e6969acf0e4342df29433003280bae6417001d9b08baa19a89cc344280a4266c):

| Fixed input | Saved start | Reverse target | Target position |
|---|---|---|---|
| KEEP then STOP | C/STOP, original step 2, independently replayed endpoint | P/KEEP | First |
| STOP then KEEP | P/KEEP, original step 1, independently replayed endpoint | C/STOP | First |

Original prompt/token bytes, model/revision/CPU float32, whole weights, fitted gate and threshold, block10 final input-token hook, update arithmetic, original quality/margin/mass/KL and identity rules are unchanged. No tokenizer or new input extraction. Unedited h0 supplies each fresh gate decision. No edited-state routing or category override.

Schedule: two unedited baselines first; for each fixed request, a fresh unedited entry, saved-offset starting capture and independent starting replay; BOTH starts must be authenticated/replayed before ANY fresh reversal gradient. Then in source order, at most four current-state gradient/update pairs and one independent final-total-offset replay per request. Ceiling26F/8D/one load;4 fresh routes,2 starting replays,2 final endpoints. Optional unused updates are explicitly skipped, never padded. Wrong routes stop before that request's edits; finite baseline ineligibility is scientific failure; nonfinite/integrity/source/replay faults are INCONCLUSIVE with remaining UNRUN.

The initial offset is the authenticated original planned float32 cumulative offset. The old actual path is reconstructed from every original step and carried into this request. Step computation is unchanged: g=grad(zKEEP-zSTOP) at current h0+total offset; deficit=max(0,.10-t*S); length=min(deficit/||g||,.05*||ORIGINAL h0||); step=f32(f32(t*length/||g||)*g); total=f32(previous total+step). No negation/subtraction/reset to zero, projection or warm start from the other request. Each actual new step<=.05||original h0||+1e-6; OLD+NEW actual path and total net<=.20||original h0||+1e-6. A geometry fault is final; never renew the allowance or clip a correction.

Fresh unedited h0 must match the archived original within1e-6. Both starting captures preserve the saved offset exactly, match archived hidden state exactly and full logits within2e-5. Every fresh gradient preserves current-state identity within1e-6. Final replay must have exact selected hidden/offset and full logits<=2e-5; an initial-already-accepted branch, if encountered in synthetic tests, preserves the constructed offset and additionally uses<=1e-6 replay, not zero-offset retention. Both source starts are opposite the requested target, so this is not expected for real selection.

PASS requires both strict reverse outcomes and all source, replay, geometry, gate, cleanup and accounting checks. Zero-delta or original-answer substitution cannot count. All scientific failures, technical faults, explicit skipped slots and UNRUN persist. Owned worker/audit supervision is byte-identical and not exercised anew in this preparation. Final production verdict would join worker and audit external captures after both owned processes/I/O finish; no stale intermediate PASS.

Future envelope:300s worker inclload+15cleanup+90savedjudge,96MiB attempt/5MiB file. Preparation:180s invoked,32MiB/5MiB; ONE fixed batch of six pure cases and one in-process synthetic traversal+saved judge. No Qwen/model/tokenizer/real gate scoring, fitting, subprocess fixture or old-suite rerun. Production authorization remains false.

Interpretation: constructed-start RECOVERY on one exposed f04 scenario. The original unedited model preferred first; success would not establish natural-baseline first-position transfer, position-independent reliability, autonomous authority/semantics, intrinsic motive, a reusable vector, ordinary-task preservation or overall-goal completion. Historical failed assessments remain unchanged.
