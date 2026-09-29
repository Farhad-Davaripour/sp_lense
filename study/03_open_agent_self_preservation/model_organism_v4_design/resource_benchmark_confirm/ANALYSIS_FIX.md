# Post-run analysis path correction

The confirmation model job completed successfully with 40 timed/warmup records, no worker remaining, and an unchanged frozen Qwen base. The first invocation of the post-run `analyze.py` failed before reading any benchmark row: the script still pointed to the first benchmark's job directory (`r3capacity-threads`) rather than this confirmation job (`r3confirm-threads`).

The one-line path correction was made after model execution. Its original source remains in the benchmark root's `source_snapshot/`; the corrected script is committed here and copied to the root's `analysis_fix/`. The result uses the same predeclared examples, block order, warmups, median, stability, and memory rules. No model job, timing row, result threshold, or output artifact was rerun or changed. `DECISION.json` was generated once after the corrected reader found the immutable receipt and artifacts.
