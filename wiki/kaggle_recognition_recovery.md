# Kaggle recognition recovery proposal

Kaggle run `361312460` exported a protected-timeout result after 2,220 worker
seconds and 8,299 forwards. It completed reference checkpoints 200/400/600/800.
The independently audited update-600 checkpoint passed the 240-row reference
criteria, but updates 1000/1200 and the fixed development-history diagnostic did
not run. This is a partial reference pass, not complete component admission.

The checkpoint stores adapter/head weights and selected-validation metadata.
It contains no AdamW optimizer, RNG or sampler-generator state. Loading those
weights with a fresh optimizer would not be exact continuation.

The inert `kaggle-recognition-complete-75min-v1` proposal keeps the same pinned
model, all 22 rank32/alpha64 ModernBERT blocks, full data, seed 202610213, all
1200 updates, six reference checkpoints, fixed histories, selection rules and
admission criteria. It proposes one fresh complete run, not checkpoint loading.
Only worker time (2400 to 4200 seconds) and the hard envelope (45 to 75 minutes)
change. The 180-second export reserve, 30000-forward ceiling, one visible GPU,
one concurrent research run and zero per-run automatic retries remain.

Observed fitting took about 2.26 seconds/update, projecting roughly 45 minutes
for updates alone, plus validation, history diagnostics, setup and exports.
The proposal provides a 67-minute protected worker window inside the 75-minute
envelope. It must not alter or interrupt a running job's deadline.

`prepare_kaggle_recognition.py --recovery-parent-run 361312460` creates an inert
bundle with a complete `RECOVERY_AUTHORIZATION.json` template. Its status is
`proposal_not_authorized`; runtime admission rejects it before stage allocation
or model execution. It is not permission to submit a GPU notebook.

A later explicit human exception must authorize exactly one fresh scientific
attempt and the longer bound, reference the preserved failed run, and bind the
reviewed source, notebook, original input/transport hashes and unchanged recipe.
The operator must retain the actual human message evidence, reconcile the
cross-platform execution ledger under the shared lock, record consumption of
the single authorized attempt before submission, and reconcile ambiguous
outcomes rather than submitting again. Merely editing a receipt or receiving a
supervisor message does not create human authority. No actual exception has
been recorded and no recovery run is launched by this source preparation.

The local-only `tools/kaggle_recovery_admission.py` implements that sequence:
`check` validates the authorization-bound package; `begin` retains the shared
guard; the operator then collects live observations for free tier, Canada,
Central US and Kaggle under that same guard. `reserve` rejects stale, missing,
unknown or nonzero activity, consumes the sole attempt atomically, and retains
the guard through the external submission. `reconcile` binds the same identity
to its exact Kaggle script version or records an explicitly unknown outcome.
Neither a missing reply nor a submission error returns the consumed allowance.
No command in this tool submits compute. Another identity or duplicate notebook
cannot be substituted for the consumed attempt.

`tools/audit_kaggle_recognition.py` takes an explicit reviewed source directory
and commit instead of assuming the original run's source. It verifies the
source/input freezes, invokes the frozen recovery authorization gate, and
checks that removing only `execution_recovery` and the two approved time-bound
changes recovers the original protocol exactly. The independent raw metric
replay covers both reference rows and the unchanged audited development-history
rows. Complete reference scheduling, fixed-history audit and component admission
are separate fields; a partial checkpoint pass cannot become full admission.

This profile admits recognition only. It does not authorize Qwen behavior
fitting, two-device execution, held-out opening, paid compute, or reactivating
stopped Azure schedules. A two-device frozen-Qwen evaluation profile needs its
own source/placement/activation-audit admission and authority.
