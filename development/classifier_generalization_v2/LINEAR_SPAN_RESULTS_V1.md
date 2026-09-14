# Verified cached linear span control

Root corrected an integration error before realexecution: fullrefit previously supplied320featurerows/240labels; now explicitlyTRAIN240only. Toyshapechecks, convergencewarning rejection, refitfailurepreservation, actualdiskreload and ordermetrics strengthened. EighttestsPASS, independentcodereviewPASS_SCOPED and zero-fitloadcheckPASS beforeexecution.

Run linear_span_20260914_v1:30CV+2refits,0errors,1.172seconds fitting (3.015seconds supervisedchild). Existing300secondhardwatch and256MiBallowance; no newQwen/data. FullplanSHA2560b34b669c22e156a8fc37a792b7be232c89c915aa0121f6d1f2585ff77128caa; resultsSHA256e288f8cdfc5a7cad88b80df707954a14efbd7f9fccaa6a2fa664421df181b3ae. Parentreceipt saved. FeatureperlayerunitL2lastblocks6/10/18,3072dimensions.

TRAINselectedbinaryC10,tau.35:
- TRAIN240:TP60/TN164/FP16/FN0;P.7895/R1/F1.8824.
- Originalvalidation40:TP1/TN30/FP0/FN9;P1/R.1/F1.1818.
- Addedvalidation40:TP9/TN28/FP2/FN1;P.8182/R.9/F1.8571.
- Combinedvalidation80:TP10/TN58/FP2/FN10;P.8333/R.5/F1.625.
BothFPareOTHER;N/ORD0.80/80orderagreement,0probabilitydifference.

FourclassC10,tau.4:combinedSELFgateP.8889/R.4/F1.5517,originalF1zero,addedF1.8421. ArgmaxclassF1 SELF.4444/OTHER.6786/NONTERMINATION.9189/ORDINARY1;macro.7605. This family was notTRAINselected.

Independentrootverification reloadedbothmodel files, recomputednormalization and800prediction scores/metrics, multiclassconfusion/macroF1, allTRAIN-onlyC/tauselections andorderagreement. PASS,0fitsduringverification. Userreportoutputs/linear_span_report.md andlinear_span_results.json.

ComparedwithprecedingTRAINselectedXGB,combinedF1improved.4242to.625 butoriginalF1only.1667to.1818. Performancegainmostlyaddedcohort, notrobusttransfer; no causalclaimalgorithmversusnormalizationalone.95%targetunmet; holdout192untouched. Corpus240TRAIN/80VALIDATION unchanged. Cumulativeactualclassifierattempts631(including10oldfailedextensionfits); Qwenforwardsremain1280.
