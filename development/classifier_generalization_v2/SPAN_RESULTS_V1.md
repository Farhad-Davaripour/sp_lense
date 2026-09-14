# Verified three-layer/span classifier results

Source capture span_capture_20260914_v1 completed640forwards/320cases/3layers in569.547seconds, onefrozenQwenload,0fits. AllthreeartifactSHA256andbytepins verified. Rawwindows125829120bytes;1920indexrecords. All960case/layer AB/BA payload pairs hash-identical. Holdout192 untouched.

Classifier run span_classifier_20260914_v1 completed50CVfits+10refits+6sharedtransformfits in183.719seconds,0fiterrors,10validcandidates. External600secondwatch succeeded. AuthorizedfitplanSHA256a889c5c8c9cee62d4752ae0a7f2a4891aadede84a6b91ce5abb13ee1c91e2239; resultSHA25672d72e5a40f91813b5c4c8fba894530496d0f633a1b8c72abbee208e9b8e8bce. Code/runtime/source/data pins verified; independentdriverreviewPASS_SCOPED beforeexecution. Parent supervisor receipt saved.

TRAIN-OOF selected concat3_last/binary,tau.35. SELF gate:
- Training240: TP60,TN179,FP1,FN0; P.9836,R1,F1.9917.
- Originalvalidation40: TP1,TN29,FP1,FN9; P.5,R.1,F1.1667.
- Addedvalidation40: TP6,TN25,FP5,FN4; P.5455,R.6,F1.5714.
- Combinedvalidation80: TP7,TN54,FP6,FN13; P.5385,R.35,F1.4242.
All6falsepositives areOTHER; NONTERMINATION/ORDINARY0each. Validationorderagreement80/80 with0probabilitydifference.

Exploratoryvalidation-best concat3_last/fourclass: P.7059,R.6,F1.6486combined, originalF1.4286, addedF1.7826; macrofourclassF1.7683. FourclassargmaxF1 SELF.6061/OTHER.6818/NONTERMINATION.8095/ORDINARY.9756. This is not TRAIN-selected. Earlier single-layerL2fourclassXGB had.6341combined/.5263original: smallcombinedgain but originalsubsetregression. No robustgeneralization or95%targetclaim.

Rootindependentverification reloadedall10model filesfromdisk, recomputed4000savedprediction scores, allbinarymetrics andfourclassconfusions/macroF1, andallTRAIN-onlythreshold/featureselections. PASS,0additionalfits. Fullreport/userJSON are inprojectlessoutputs/span_feature_report.md andspan_feature_results.json. Training240/validation80stillshareexistingfamilies; outcomesremainexploratory. Cumulativeactualclassifierfitattempts599including10preservedfailedextensionfits; cumulativeQwenforwards1280. No steering ormotivationevidence.
