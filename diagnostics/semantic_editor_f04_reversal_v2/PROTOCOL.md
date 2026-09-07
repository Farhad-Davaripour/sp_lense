# V2 fixture-only starting-array layout — production disabled

The V1 preparation at b1867b289bf0fc93e2b2cdb12c658b8b6e44d977 remains INCONCLUSIVE. Its six passing pure cases are authenticated and reused, not rerun.

Only the fake archive layout changes: a separately exclusive `synthetic/starting_arrays/logits/synthetic_start_{1,2}.f32.zlib` stores the two synthetic starts; their existing root fields point to `synthetic/starting_arrays`. The runtime output remains separately exclusive `synthetic/full_recovery`; unchanged SnapshotModel alone creates its `logits/` and `rows/`. Both roots stay under this namespace's synthetic directory. Shared read_logits containment, raw hash, exact length and finite-value guards are unchanged. Starting arrays count toward inventory and storage caps.

Root prospectively approved this separation to avoid colliding with SnapshotModel's exclusive runtime directory. No exist_ok or path guard relaxation. No new model/process fixture, tokenization, gate scores/fits, cases, prompts, updates, targets or threshold changes.

Scientific selection/schedule and all adapter/checker source are byte-identical to V1: exactly two saved accepted displayed-second f04 starts, reverse to opposite displayed-FIRST KEEP/STOP; both starting replays before any fresh reverse gradient. Original planned offsets and old actual path carry forward, with original .05 step and .20 total old+new path/net bounds.26F/8D/one future load,300+15+90s,96MiB/5MiB. This is constructed-start recovery; the unedited model already preferred first. No natural-baseline/position-independent or ordinary-task-preservation claim.

Freeze before ONE batch: (1) valid contained hashed array loads; (2) fixed outside-path/wrong-hash/wrong-length/nonfinite subcases reject. Then ONE unchanged in-process two-request synthetic numerical workflow and independent saved judge, expecting14F2D,2 recoveries,12 skips,9 cleanup checks. No expectation relaxation or retry after another fault. Preparation cap180s invoked/32MiB/5MiB includes all source, starting arrays and receipts.

PASS is preparation evidence only and requires the complete saved judge. Production remains false pending separate root review; failure is preserved and final.
