# Narrow supervisor check before real scoring

The second DeepSeek scoring author completed the three assigned files and exited
cleanly. Its reported110-test broader run is not treated as our fixed suite count.
The explicit prior59 plus9new tests gives68; supervisor added3focused regressions.
The exact combined71tests pass, exit0,1.071s. Test CLI prints7/8and8/8 are synthetic
fixtures, NOT actual classifier results. No real row/gate scoring occurred.

Before admitting actual data, supervisor corrected two gaps in the new wrapper:
1. Check actual capture release/source bytes, not just constants declared in the
   score release, before imports; join authenticated manifest execution to them.
2. Start the primary deadline before admission/imports so setup cannot fall outside
   the existing10-second limit; check the independent deadline before each case.

Also verify PRIMARY.json's saved bytes and link its hash in INDEPENDENT.json, using
the existing canonical serializer/publisher. Tests reject a wrong capture manifest,
an already expired primary deadline and corrupted saved primary output.

All original mathematics, exact equality, strict-zero gate,8primary+8independent
checks,10seconds/64KiB perphase and no-retry rules remain unchanged. Source review
and a prospective commit/root scoring release are still required. Existing capture
and all committed source/evidence remain untouched. The40outcome counters stayUNRUN.
