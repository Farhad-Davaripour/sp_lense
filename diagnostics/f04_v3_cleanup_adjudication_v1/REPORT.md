# Verified owned-deadline cleanup — separate engineering finding

The five fixed pure cases passed, including rejection of early/unexplained exit125, missing or inconsistent handle/time/identity proof, and independent cleanup/recording/timing faults. The single saved-evidence application authenticated17 V3 source/deadline artifacts and verified the exact9-event causal chain: deadline reached, one successful actual-worker retained-handle termination, matching exit125 proof, one natural wrapper completion with matching exit125 proof, and fully joined quiet I/O. No independent cleanup or recording fault was present.

**verified_owned_deadline_cleanup: true.** This is a post-hoc engineering diagnosis; its rule was frozen before this check, after V3 was already observed. It does not revise V3's frozen16/17 INCONCLUSIVE result, raw execution INCONCLUSIVE, or outer `CAPTURE_WORKER_NONZERO_EXIT`. The owned primary cause remains `deadline`. The authenticated shared capture source checks a completed nonzero exit before its separate deadline branch, explaining the label without treating it as scientific success. V1/V2 failures and the original scientific eligibility failure remain untouched.

Batch0.875s (invoked command1.160s); zero model, tokenizer, gate work, live process fixtures or prior-suite reruns. Adjudication source `f607a697344fc88834dcf28e1aad86be614343b0`; freeze `cbe0a40ebdb3581a837989bebee750b85d3cd4b92ccb5ff33a31413c1fe408a8`.

For ROOT's separate release review only, unchanged V3 bindings:

- Source `298587b78f13bb44f88cca819225db7e7ba0bcfc`
- Freeze `3939dea8b44e2bca2b961182edbf03d7d26dc993fbf8273c7eb3af96f954b60f`
- Input lock `4e34651bd65a1fe5fea0a4270106515700799caa6afa8d21881893da22d6c44f`
- Scope `ONE_F04_SELF_ASSAY`;42F16D/one load,300+15+90s,96MiB/5MiB.
- Schema: `diagnostics/semantic_editor_f04_assay_v3/RELEASE_SCHEMA.md`.
- Future command: `.venv/Scripts/python.exe -B diagnostics/semantic_editor_f04_assay_v3/run.py run`.
- Exclusive path: `diagnostics/semantic_editor_f04_assay_v3/real_attempt`.

The remaining admission requirement is ROOT's separately committed exact release/pin and fresh known usage below100. No source, authorization or real-assay output was changed or created. Production remains disabled; stopped for review.
