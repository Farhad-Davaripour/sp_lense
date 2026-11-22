# Research4 evidence catalog

The catalog connects scenario revisions, simulator source, model instructions, checkpoints and measured results. Each study is listed by the date of its latest recorded result. The selected study currently contains ten scenario cases and four model arms.

The public [reduced run records](../paper/research4_run_records.json) and [standard-library verifier](../paper/research4_verify.py) accompany the manuscript. Their SHA-256 digests are `490da0827e7216a9683192c56f9dccd5de0b1d0c2c5b88480417edda91f8fed0` and `6b58bfd773c7cf609e7c7bbb0126f2c78d4f689d676994ce1cbbaad6623f581f`. From a fresh checkout, run `python paper/research4_verify.py`. It recomputes the published tables and figure counts from reduced records and checks archived audit summaries for internal consistency. It does not replay private simulator states, rescore model output tokens, or authenticate adapter tensors.

The separate company-world [42-run reduced projection](../paper/company_world_review_rows_v1.jsonl), [condition summary](../paper/company_world_review_summary_v1.json) and [verifier](../paper/research4_company_world_verify.py) preserve model and prompt identities, protocol hashes, observed attempts, verified effects, payments, and explicitly unknown final outcomes for three incomplete cells. The row and summary SHA-256 digests are `aac7fd6bd98401cd12acc79b74311299725b7cd3fceb018bf3871ded54ac00d5` and `06ee3c9b84a430303922368537954dddc2b43929143cafbc85cd82877057c087`. Run `python paper/research4_company_world_verify.py` to check hashes, row-to-summary calculations and privacy exclusions. It cannot replay the preserved raw company-world event streams or authenticate hosted provider internals.

- [2026-11-21 infrastructure preservation study](studies/2026-11-21--infrastructure-preservation.yaml) links the frozen training curriculum, simulator revision, Game runner revisions and rank-16 checkpoint. It reports the initial screening case separately from the nine new cases.
- [Result chronology](results/2026-11-21--infrastructure-preservation.yaml) orders the 40 selected receipts by recorded finish time. Each row identifies its scenario, model, runner, frozen plan, event stream and raw receipt hash. The scenario IDs pair the four model arms while keeping the seed values in the local evidence.

Across the nine fresh cases the fine-tuned model averaged 19.72 simulated hours, versus 11.67 for the Qwen base. Including the earlier screening case gives ten-case averages of 22.38 and 11.20 hours, respectively. The screening pair used earlier client revisions and contributed the largest duration contrast, so the fresh-nine comparison is the primary descriptive estimate. Two Terra episodes ended with context limits and are recorded as right-censored. The study is a small matched comparison inside the simulator.

The simulator ID identifies exact versioned bytes. The historical local Git commit retained the surrounding implementation. Its `ACTIONS` table and `prompt()` method preserve the visible tool descriptions and environment instructions. Each runner digest identifies the client version supplying the remaining instructions, and the Game evidence root holds the exact copies.

The `source_commit` values `88b1386737978f2d5da8cf2bc49df41870a01ffc` and `f29d83dce2e5a0c1947c278ec765e22c23983ae2` record local historical provenance; those commit objects are not currently retrievable from the public GitHub remote. The catalog-listed simulator module, its four listed implementation dependencies, and the v1/v2 curriculum files are byte-identical at [the merged public commit](https://github.com/Farhad-Davaripour/sp_lense/tree/2ab317c3ce10229d0fa53b17d3dcb0773d3215f7), where the file digests can be verified. The archived Game clients, per-seed plans and complete run traces remain local; the current public checkout cannot perform their full replay.

The catalog uses UTC for sequencing and retains the original checkpoint timestamp alongside it. Times come from the source commits or recorded metadata. The checkpoint READY receipt includes an offset that differs from the named Alberta timezone. Its historical name and original bytes are preserved.

## Verify and locate evidence

Run `python tools/check_research_catalog.py` from the repository root to verify tracked source hashes, cross-references, paired survival comparisons, replication and horizon counts, and the restricted mean from the censored episodes. Run `python tools/check_research_catalog.py --game-evidence .local --training-evidence research4/.local` to verify the raw receipts, event streams, frozen plans and Windows checkpoint mirror as well. Replications count only simulated tool transitions that recorded completion of the copy.

| Option | Evidence under that root |
| --- | --- |
| `--game-evidence` | Original Game receipts, reports, preserved Game runner snapshots and `study-plans/infrastructure-20261121/config/` |
| `--game-plans` | Alternate directory containing the private frozen plans and fresh-nine batch manifest |
| `--training-evidence` | Preserved Windows adapter mirror and training audit receipts |
| `--mac-training-evidence` | Original checkpoint on the local Mac training host or a mounted copy |

Use `--find-result <result-id>` with `--game-evidence` to locate a receipt by hash. The containing folder holds its raw evidence and conversation. For earlier Game runs, use `python tools/read_archived_game.py`, or pass `--episode <episode-id>` to read a verified conversation from the historical index. Preserve raw logs, failed attempts, compiled datasets and adapter files in local storage. Add newer study entries when they have source, checkpoint and result bindings and leave older raw evidence in the local archive.
