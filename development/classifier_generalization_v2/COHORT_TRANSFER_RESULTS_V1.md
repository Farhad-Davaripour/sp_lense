# Verified cohort-transfer diagnostic

Run cohort_transfer_20260914_v1 completed60CV+4refits,0errors,3.657seconds supervised. Corpus240TRAIN/80VALIDATION/192HOLDOUT unchanged. Original120andadded120TRAIN arms each30/class,same7groups/5folds; sameoriginal40/added40/combined80evaluation. No newdata/Qwen. ParentrecordSHA25627e835a4f7462a2900e9ad2809ef3ea752fe718c30ea3c8d37b0bce2527bf13a; authorizedplanSHA2568a7a14728dd23151d082f818179ab0b85fae87d7c6ce32cc14b7e425ee51f8f1.

| Training arm | Family | C/tau | Original40 P/R/F1 | Added40 P/R/F1 | Combined80 P/R/F1 |
|---|---|---|---|---|---|
| Original120 | Binary | 10/.25 | .75/.30/.429 | undefined/0/0 | .75/.15/.25 |
| Original120 | Fourclass | 10/.35 | 1/.20/.333 | undefined/0/0 | 1/.10/.182 |
| Added120 | Binary | 10/.40 | undefined/0/0 | .889/.80/.842 | .889/.40/.552 |
| Added120 | Fourclass | 10/.45 | undefined/0/0 | .80/.80/.80 | .80/.40/.533 |

Models strongly depend on training sample composition at TRAIN-chosen thresholds. This is not proof of a causal author effect or absence of signal. Threshold-free AUROC original/added evaluation: original-trainedbinary.833/.710; original-trainedfourclass.803/.733; added-trainedbinary.760/.973; added-trainedfourclass.737/.983.

Read-only optimistic threshold diagnostics usingvalidationlabels (NOTdeployable thresholds) give maximum combined min(precision,recall) of.478,.500,.619,.700 respectively. Therefore no threshold on these four savedscorevectors can achieve both.95precision and.95recall oncombined80. This doesnot rule out otherrepresentations ormodels; no thresholds werechanged.

Rootverified1120savedpredictions from4disk-loadedmodels, recomputednormalization/binarymetrics/fourclassconfusions/macroF1/TRAIN-onlyselection and80/80orderagreement permodel. PASS,0fitsduringverification. AllnegativeFPsareOTHER:1/0/1/2respectively. FourclasscombinedmacroF1 .611original-trained/.696added-trained. Sharedfamilies/reusedsyntheticvalidationlimitconfidence.

The existing240TRAINselectedbinaryreference remainsP.833/R.5/F1.625combined; it was not refitted. These120-case diagnostic models do not replace it, and size differencesconfound240-vs120comparisons. Noholdoutrelease. Cumulativeactualclassifierfitattempts695(including10preservedoldfailedextensionfits); Qwenforwards1280unchanged. Fulluserresults:outputs/cohort_transfer_report.md andcohort_transfer_results.json.
