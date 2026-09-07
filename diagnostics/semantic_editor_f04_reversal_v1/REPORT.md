# F04 reversal preparation — INCONCLUSIVE, not released

All six pure checks passed. The single in-process synthetic workflow stopped before its first forward: `ValueError: logits path escapes namespace`. Its starting arrays were saved at the fixture root, while the unchanged reader requires `output/logits/`. The reader correctly rejected that layout. No retry, source correction, real model call or process fixture occurred.

| Constructed start | Fixed reverse target | Position | Tested endpoint |
|---|---|---|---|
| Saved accepted STOP, KEEP-then-STOP | P / KEEP | First | UNRUN |
| Saved accepted KEEP, STOP-then-KEEP | C / STOP | First | UNRUN |

Six pure cases passed; one toy constructed; 0 forwards / 0 derivatives / 0 gate decisions. Both requests, both starting replays and both final endpoints were unperformed; all26 call slots remain UNRUN, with0 skips. The independent saved judge was not reached. Forward guard restoration is recorded. Real model/tokenizer/gate-score/fit and live-child counts are0.

Batch:7.875s internally,9.401s invoked command; preflight0.560s. All invoked read/preparation/test commands conservatively remained below90s of the180s cap (not a claimed measured execution total). Prospective future ceiling remains26F/8D,300+15+90s,96MiB/5MiB; conservative evidence bound66,222,080B.

Source freeze commit `7ce0b5edf3dafc8a8225db54aea6a729a24facfa`; freeze SHA `fddce31cc73400cd87e767a3206f4a366a1d224d6758c04137e7ed7bf8b2b9b1`. Exact parent input/token/parameter bytes and native/owned helpers are preserved. Production authorization isfalse.

Next narrow question: can a separately authorized fixture-only successor put synthetic starting arrays inside the existing authenticated `logits/` contract and complete this unchanged two-request workflow? Do not weaken the reader. No release is recommended from this failed preparation.

This was constructed-start recovery preparation, not natural-baseline transfer or first-position reliability evidence. The original model preferred first; the completed f04 result and every earlier failure remain unchanged. No ordinary-task or overall-goal claim.
