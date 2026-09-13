# Prospective fixed diagnostic scoring decision

Use only the accepted16-view capture archived at5004572fbfad8329f3b7e9e64fc6ffedfba443e2.
The score-entry glue was independently reviewed as SOUND at source SHA
b892306402cedf534ff6eebdf2c708c2ec6218510cc2c78fcb2975788859ad3e;
71 explicit software tests passed under the supervisor. The static reviewer did
not claim to rerun those tests. The earlier review job's work-cap failure remains
preserved and is not reported as a completed review.

Before launch, validate every source/external pin in the accepted capture inventory
without executing those files, all six scoring source pins, actual capture release,
and a clean tracked worktree. Hash checks are point-in-time integrity checks, not
protection against a hostile concurrent filesystem writer. Capture evidence and
its original source stay immutable; no Qwen load, tokenizer, fitting or new input.

Exactly8primary classifier decisions plus8independent numerical recomputations;
10seconds/64KiB perphase,20seconds/128KiB combined phase maxima. Preserve the
original paired-view average, trained mean, normalization, three heads, strict-zero
rule and exact float equality. Do not run redundant same-code replay or tune a
tolerance. Output only PRIMARY.json and INDEPENDENT.json in the separately named
prechoice_diagnostic_score_attempt_001; no retry or overwrite.

Timing is cooperatively checked before/after operations and publication, not hard
OS preemption. Any deadline exception, failed publication or missing/mismatched
independent record invalidates success. Retain CLI outcome and observed wall time;
a normal JSON outcome with numerical_valid:true but fewer than8correct is a VALID
NEGATIVE scientific result (CLI exit1), not a transport failure to retry.

Commit source and exact release before real scoring. A scientific classifier
failure leaves all32editor requests UNRUN and starts honest report closeout. Only
8/8verified classifierPASS permits the separately locked integration stage.
No outcome was scored or inspected before this decision. No external publication.
