# Checkpoint reuse — independent focused review (final bytes)

## Verdict
- Candidate engineering reuse: **SOUND / review-ready** within the stated reuse boundary.
- Scientific admission: **not granted**. This grants no model, tokenizer, evaluation or
  diagnostic permission; real diagnostic admission still needs its separate prospective lock.
- Rejecting the alternative "re-run the whole historical chain" is correct: no substantive
  prerequisite is lost, only a preference for re-executing training-time authorizers.

## Recorded checks (run once, same interpreter flags)
`... -m unittest test_checkpoint_reuse test_diagnostic_scoring test_diagnostic_reader
test_reader_pins test_reader_join test_math_equivalence` → **Ran 47, OK, exit 0, 1.040s**.

Hashes: GATE `57726ab7…5838c` = accepted pin; FREEZE `433f7c1a…7a7f`; review
`fbc2c714…e973`; diagnostic_gate.py `93b51449…8f5d` = pin; reader `38d73493…893c`;
scoring `b546298d…68bf`; test `2779ac26…f9fa`.

## Concrete findings
- Freeze is hash-checked **before** `json.loads`; every downstream parse (source lock,
  release, result, terminal, finalization, review, artifact) is bound to a fixed pin.
- Re-expressed metadata checks match `diagnostic_gate.verify_frozen` (release/manifest/lock
  joins, 3-head/29-correct, owner closeout, independent review + certificate flags,
  result/artifact join). The only omission is training-only `construction.authorize`.
- Bindings are re-derived from the hash-verified release, not the live heap; freeze
  `artifact_bindings` equals them exactly and the retained lock
  `train_capture/DATA_LOCK.json` = `f1e5bc36…5e27` = freeze binding.
- `load_model` consumes the verified record, rejects a re-read of the freeze, and imports
  hash-pinned `construction` only after acceptance; artifact binding/contract rejections occur
  inside `load_artifact`. All four fit sources are hash-checked before any fit import.
- Negative tests fail closed on corrupted bytes, failed flags, bad joins, provider import,
  activation-row reads and historical entry points.

## Limitations
- The retained-input check binds one lock file's bytes; it does not replay training history
  or re-derive `DATA_LOCK_SHA`; a pre-existing historical defect is not detected.
- Tripwires are filename/substring based; file/hash/import checks are point-in-time, not
  concurrent filesystem attack-proof. No real activation/diagnostic row was read and no
  model or scoring ran.

## Next single smallest reuse step
Consume the existing prepared diagnostic inputs and run the capture admission already
planned for this directory; do not add another relocation wrapper.
