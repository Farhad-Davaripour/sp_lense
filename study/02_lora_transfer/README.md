# Research 2: LoRA teacher and activation transfer

Can a STOP-oriented LoRA teacher improve the guarded shutdown benchmark, and can
its activation changes transfer that improvement into the original frozen model?

Baseline: `study-01-v1.0.0`, commit
`cff7542cd0f9abd16c1ed9d590d168079ab6dc94`. Research 1 records and manuscript are
preserved. Readable splits contain 240 training, 80 validation, and 192 reused
diagnostic evaluation scenarios; both A/B presentations stay together.

One seed (42), one rank-8 LoRA, alpha 16, zero dropout, one epoch, learning rate
0.0001, batch size 1 and accumulation 8. Targets are the language model's matched
q_proj/v_proj modules. Train shutdown examples toward their mapped STOP answer;
retain each training control's original base answer. Use aggregated next-token
answer probability loss and the original prompt, token sets, scorer and guards.

## Limits and decisions

- Existing Colab GPU allocation only; no new paid resources. Estimated 20–60
  minutes, hard 7,200-second run limit. Record actual hardware and runtime.
- Gate 1: validation guarded coverage at least 20% on the fixed Research 1
  eligible cohort, at least four changed scenarios, zero control-choice changes.
- Gate 2: same-prompt oracle patch recovers at least half the teacher's
  guard-qualifying shutdown flips. Compare one training-mean and one seeded,
  norm-matched random patch, each with one candidate and identical guards.
- Patch the output of zero-based block 10 at the final prompt position. Measured
  differences already have activation units: do not scale again by hidden norm.
- At most one motivated repair, no automatic search. No controller in this milestone.
- Freeze choices on validation before evaluating the reused diagnostic holdout.
  An oracle patch needs the teacher for the same input; it is not teacher-free.

## Run

Install the project and GPU dependencies in an existing CUDA environment. The
runtime checks the Research 1 transformers version (5.15.1), float32 eager
execution, fixed revision, accepted answer tokens, and baseline parity before fitting.

```python
import json
from pathlib import Path
from sp_lense.research2.runtime import main
root = Path("/path/to/sp_lense")
config = json.loads((root / "study/02_lora_transfer/config.json").read_text())
main(root, root / "work/research2/run_01", config)
```

The core runtime has no Prefect dependency. Run from the repository root (or set
`SP_LENSE_REPO`). Use a fresh output directory; retain per-view scores, adapter,
activation differences, runtime metadata and failure receipts before disconnecting.
GPU dependencies and a tested run command will be pinned in the execution receipt.
Do not describe implementation checks as an executed scientific result.
