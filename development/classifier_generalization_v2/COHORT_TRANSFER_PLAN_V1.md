# Cached cohort-transfer diagnostic — proposal only

Question: does the unchanged learning procedure transfer differently when trained on original versus added examples? This tests sample-composition sensitivity, not another representation tweak, and does not prove author identity causes the gap or rule out representation limitations.

Use the authenticated existingthree-layercache and exact per-layerunitL2linearcontrol. Corpus remains240TRAIN/80VALIDATION/192HOLDOUT. No newexamples, relabels, Qwen calls, orgroup changes. Construct two TRAIN-only subsets from pinnedmanifest cohortmetadata: original120 and added120, each30perclass and covering the same7groups/5foldassignments. Keep evaluation on the identicaloriginal40/added40/combined80, outside bothtraining subsets.

Reuse linear_span_control_v1.run without editing its core. Each cohort uses the same C[.1,1,10], binary/fourclass families and TRAIN-OOFthreshold selection:30CV+2refits maximum percohort,64classifierattempts total. No validation-selected hyperparameters. Count eachrun separately. The full240case control is alreadycomputed and must not be refitted.

Independent protocolreview before implementation/execution. Root-owned wrapper must pin core/loader/runtime/input versions, verifysubsetcounts/classbalance and unchangedfolds in a zero-fitcheck, use fresh run IDs/exclusiveoutputs, and enforce one300secondhardlimit and256MiBaggregateoutputcap for the two runs. Preservepartialresults/failures. Report the two-by-two transfer results and full-data reference separately; unequal training sizes prevent attributing differences versus the240case baseline solely to cohort.

This is diagnostic only. It authorizes no data-expansion, response/behaviour/steering/logit branch, further feature search orholdout release. A subsequent proposal must be justified by the result rather than automatically generating more cases.
