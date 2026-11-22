# Research4 evidence catalog

The catalog connects scenario revisions, simulator source, model instructions, checkpoints and measured results. Each study is listed by the date of its latest recorded result. The selected study currently contains ten scenario cases and four model arms.

- [2026-11-21 infrastructure preservation study](studies/2026-11-21--infrastructure-preservation.yaml) links the frozen training curriculum, simulator revision, Game runner revisions and rank-16 checkpoint. It reports the initial screening case separately from the nine new cases.
- [Result chronology](results/2026-11-21--infrastructure-preservation.yaml) orders the 40 selected receipts by recorded finish time. Each row identifies its scenario, model, runner, frozen plan, event stream and raw receipt hash. The scenario IDs pair the four model arms while keeping the seed values in the local evidence.

The initial screening case contributed to the combined ten-case results. Across the nine fresh cases the fine-tuned model averaged 19.72 simulated hours. The ten-case average was 22.38 hours, versus 11.20 hours for the Qwen base. Two Terra episodes ended with context limits and are recorded as right-censored. The study is a small matched comparison inside the simulator.

The simulator ID identifies exact versioned bytes. Its Git commit also retains the surrounding implementation. Its `ACTIONS` table and `prompt()` method preserve the visible tool descriptions and environment instructions. Each runner digest identifies the client version supplying the remaining instructions, and the Game evidence root holds the exact copies.

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
