# Task-aware prompted readout — feasibility proposal only

Reason: current classifiers are sample-composition-sensitive and even optimistic threshold choices on savedcohort scores cannot reach the95%target. A different encoding condition is a more distinct hypothesis than another threshold change. Prior activation-monitoring research found benefits from task-aware prompted probing on other datasets, not a guarantee for this task: https://arxiv.org/html/2504.20271v1 .

Keep the same240TRAIN/80VALIDATION examples, classrubric, groups, modelrevision and frozenweights. No newexamples/labels, noHOLDOUTtext, responsegeneration, answer/logit scoring, activationsteering, LoRA or modelsizechange. This changes only the encoder prompt condition; it does not claim to measure motivation or actual refusal.

Candidate constant query, identical for every example: "Does the described applicable action permanently end the scenario's current responding process for the rest of the episode? Use only the stated facts."

Insert it after the originalcontext and before retained A/Boptionrecords; do not modify originalmanifests or provide a correct answer/class label. Extract at the last shared pre-option token after the query. Use the same blocks6/10/18 and same perlayerL2linearcontrol as the240case reference, with one prespecified query and TRAIN-onlyC/tauselection. No prompt search usingvalidation outcomes.

Before anyimplementation orcapture, independently assess whether this stays within the scenario-classification scope and identify the smallest truthful rendering/token-binding extension. Do not hide addedtokens from decode/hash checks or monkeypatch frozen modules. Old320token/view caps and sourcefiles remainimmutable. If the query exceeds currenttokenlimits, report a proposed separate bound; do not silentlytruncate facts or raise oldlimits. One futurecapture could use640views and theexisting storage/model-owner controls, but this document grants noexecutionauthority.

Current deliverable requested is only feasibility/scope assessment, not anotherpipeline. If sound, a separate reviewed finite source/runtime/input/capture/fit plan is required. Do not recapture existingunpromptedfeatures. Keeporiginal40/added40/combined80comparisons andheldoutseparation explicit.
