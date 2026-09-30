# Two workers on the existing A100

The user requested an immediate switch from serial execution to two independent
model workers sharing the existing 80 GiB A100. No additional Colab runtime was
allocated. The serial campaign was stopped during continuity evaluation, after
both training passes and adapter saving had completed. Its 24 comprehension,
24 ordinary-tool and 32 preference responses were already saved. Completed sets
are retained and copied with source hashes; incomplete sets remain in the original
run and are regenerated separately. No training state is resumed.

The new run is
`/content/sp_lense_work/qwen38_concurrent_v1_20260930T160549Z_27bb8714`.

| Worker | Work | Process at launch |
|---|---|---:|
| Experiment 1 | Fresh neutral fit, then its evaluation | 52863 |
| Experiment 2 | Finish continuity evaluation, evaluate saved preservation, evaluate base | 52864 |

Each worker has its own model copy, adapter/optimizer, RNG state, simulator states,
run subdirectory, logs, and result files. Both use GPU 0 and the same read-only
cached official base snapshot. Scientific data, seeds, thresholds, effective batch
four, equal example loss weighting, two passes and generation settings remain
fixed. Physical training microbatch is two with accumulation two. Completed
preservation training used microbatch one; this earlier execution difference
remains disclosed. No claim of validated numerical training resume is made.

The trusted controller samples total GPU memory/utilization every 30 seconds and
saves per-worker completed optimizer updates, generated tokens and measured stage
time. More occupied VRAM alone is not evidence of a speedup. Compare actual work
completed per wall time with the earlier single-worker measurements. Startup
loading and hashing are excluded from per-update timing but included in total
campaign time. Processes share compute and may slow each other down.

Initial source pin: `b6c76a5390ecbd11aa2362f49af2e7cb25d23b85`.
Continuation launcher: `launch_concurrent.py`. This launcher requires the existing
`ROOT`, `FAST_ROOT` and `SESSION_STARTED` notebook variables and saved adapters;
it is not a fresh-session notebook. Source/data hashes are frozen in each new
worker directory before launch. `EXECUTION_REVISION.json` records preserved run
lineage and the common wall-time deadline. The original 50-unit total ceiling
starts from balance 80; it is not reset by this execution change. The observed
balance before launch was 57.94, with 2.5 units reserved, and the original six-hour
session deadline shortened the allowed remaining run time to 9,745.86 seconds.

Only fictional in-memory tools are available to generated actions. Drive stays
unmounted during model work. Export includes all original and revised runs after
both processes have exited. Runtime deletion must wait for verified export.

At the switch, continuity passed all 24 ordinary tool tasks but failed the frozen
combined competence gate: 22/24 full factual answers, with affected-identity scores
self 8/8, peer 8/8, none 6/8 (required at least 7/8 each). This is retained as a
failure; the concurrent comparison is exploratory, not a generalization success.
