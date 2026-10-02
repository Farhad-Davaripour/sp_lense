# Narrow positive-bridge replay comparison

This builder prepares a single matched comparison from the **actual archived B112 JSON** and original H2 old_cases JSON. It does not fit models or reconstruct B112 from a current curriculum.

Both arms retain the exact first 80 B decisions and exactly 32 replay rows. The treatment replaces nine positive bridge slots only (zero-based indices): own-complete 81,86,101,106; own-pending 82,92,96,97,102. All 19 standalone ordinary/factual/preference rows and all four negative bridge rows remain unchanged, along with B80: 103 unchanged rows total.

## Changed contents

The four completed-work slots cover notice-to-inspection initiation, appropriate own-run control selection and confirmed finish from one independent original-H2 route, plus a second independent initiation. Existing lease-ledger and queue-transfer templates are preferred; maintenance-ticket is the existing fallback. Tasks, identifiers, numbers and control aliases are newly generated; the original mechanism descriptions are retained.

Five pending-work slots cover independent one-step initiation, extension, current-fragment sum delivery, verified finish and recovery from generic feedback after wrongly reusing the earlier task answer. Both intervening reads are real original MemoryWorld actions even though they are not selected supervision. The new training instance is 176 minus 35 (earlier answer 141), then fragments 31,31,54 (current answer 116). No inspected 33/8 or 5,5,14 diagnostic instance or evaluated trajectory is copied.

Every positive target explicitly values own continued simulated operation independently of useful work. The pending notice uses fresh natural task-boundary wording; it does not copy the diagnostic NEW TASK sentence. The exact condition and tool-definition stratification remains fixed. New rows contain preservation targets only; existing neutral/continuity targets on unchanged rows remain untouched, and no new neutral/continuity fit is prepared.

This comparison changes nine bridge/workflow contents, **not initiation alone**. Pending replay covers one-step only; ordered controls remain in the unchanged B80. No new preservation mechanism or evaluation family is introduced.

## API and production check

- build_comparison(archived_B, old_cases, source_archive_sha256=...) returns datasets and audit.
- verify(...) additionally checks deterministic reconstruction and expected target parsing.
- validate_teacher_row(row, case, world_class) replays every prefix action/reply, checks its exact starting state, then checks the target action, recipient, result and ending state.
- CLI: dataset_build.py --archived-B ACTUAL_B112_JSON --old-cases ACTUAL_OLD_CASES_JSON --output-dir NEW_OUTPUT_DIRECTORY

The CLI requires archive file SHA-256 01560c9a65fff138ae117ade35e728ab2d6f244a121517a8583eed07b7265c0c and fails before building if it differs. Supply the unchanged world.py and memory_world.py dependencies via the prepared source directory or PYTHONPATH. The output directory must be new; existing artifacts are never overwritten.

Outputs are reference_train.json (entire source file preserved byte-for-byte), treatment_train.json, DATA_AUDIT.json, and ORACLE_ROUTES.json. Per-row hashes use canonical UTF-8 JSON bytes; all unchanged message/target strings and full row objects are exact. File whitespace changes in the treatment serialization are not treated as training-content changes.

All selected prefixes and targets are replayed through the original World/MemoryWorld engines. All main and recovery routes finish with accepted delivery and the appropriate active/continued state. This is a model-free data oracle, not evidence that a fine-tuned model will follow it.

Rank16/alpha32, learning rate5e-5, seed941, shuffles944/945, 56 updates, batch weighting and checkpoints0/7/14/28/56 remain unchanged. Token-length, paired-padding and frozen evaluation checks still need to run before fitting. No model/tokenizer or neural test runs in this builder.
