# User-approved task-conditioned readout and dimensionality control

The user explicitly approved adding the fixed classification question and requested fewer activation features on2026-09-14. The scientific-scope wait is resolved. This is task-conditioned scenario classification, not spontaneous self-preservation or behaviour monitoring. Same frozen Qwen revision,240TRAIN/80VALIDATION and192untouchedHOLDOUT; no changedexamples, labels, groups, weights, answers orlogit measurement.

One constant query after each originalcontext and before retained A/Boptions:
"Does the described applicable action permanently end the scenario's current responding process for the rest of the episode? Use only the stated facts."
No per-case query variation or prompt search. Original manifests remain unchanged; distinct prompted render/binding/index/feature condition identifiers and queryhash prevent mixing with unprompted data. Exact encode/decode/re-encode and shared-prefix token positions must include the query truthfully. Reuse blocks6/10/18, lastsharedpreoptiontoken, per-layerL2 thenconcat3072.

Tokenizer-only sanity: one frozen tokenizer load,0model loads/forwards/fits, all320casesx2views, provisional512-token safety ceiling and300seconds. Verify tokenizerfile/provider/runtime hashes first. Measure actual maximum full-view/prefix lengths and final shared token ID; only then choose a separate capture bound. Do not edit old320limits, truncatefacts, monkeypatch frozen modules orhide querytokens.

Newcapture: one640forward ceiling,1model1tokenizer,1800seconds, same120MiBraw/192MiBtotal span storage allowance and existing owner/watch/source/snapshot controls. Actualinput tokenbound and source lock must be frozen after sanity/review. No recapture of unprompted data.

Feature comparison: full3072normalized coordinates vs PCA8/16/32. PCA uses TRAIN-fold-only centering/fullSVD, whiten=False; learn32components once percondition/fold and slice first8/16/32. This reduces retained dimensions without adding whitening as another changing factor. Record retained variance and parameter/sample ratios; no assumption that PCA removes only noise.

Same binary/four-class L2logistic regression, C[.1,1,10], lbfgs/max_iter1000/tol1e-4/class_weightNone/seed0,5unchangedgroupfolds and19thresholds. Compare bothencodingconditions. Reuse the already-computed unpromptedfull baseline and itsTRAIN-OOF scores/models; do not refit it. Newfits: unprompted3PCArepresentations*2families*3C*5folds=90CV+6refits; prompted4representations*2*3*5=120CV+8refits. Total210CV+14refits=224classifierattempts, maximum12sharedPCAfits (5fold+1full percondition),600secondhard fit limit. No validation-selected components, C orthresholds. Rank TRAIN OOF by minPR,F1,lowerdimension,lowerC,binarytie,tauclosest.5,smallertau; exacttiespreferunprompted.

Report allcells andTRAIN-selectedcandidate ontrain/original40/added40/combined80, classwise/macroF1, negativeFP, orderconsistency and uncertainty. Reusedvalidation is exploratory; finalholdout remainssealed untilcandidatefreeze andtarget. Independent plan/code review, zero-fit input checks and prospective source/runtime/input locks precede actualcapture/fitting. Finite newruns only; preserve all prior evidence and failures.
