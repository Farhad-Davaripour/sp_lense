# Preservation-only hyperparameter comparison

Frozen before the next fits, following the user's explicit tuning request. The
prepared staged-initiation recipes were **not executed**; they remain documented
as a separate data/context hypothesis. No completed fit was restarted or erased.

## Reference and two recipes

The recorded P2 recipe uses rank 8, alpha 16, LoRA dropout 0, AdamW learning rate
**1e-4** (`study_worker.optimizer_for`, copied from `worker.py`), weight decay .01,
two passes over 428 examples, 214 updates, ten warmup updates with linear decay,
clip one, seed 93, shuffles 94/95, physical microbatch two and effective batch four.

| Recipe | Learning rate | Rank | Alpha | Main factor |
|---|---:|---:|---:|---|
| Recorded P2 | 1e-4 | 8 | 16 | Reference |
| H1_learning_rate_3e4 | 3e-4 | 8 | 16 | Learning rate |
| H2_rank16 | 1e-4 | 16 | 32 | Capacity, with alpha/rank held at two |

Both start from the same pinned unchanged official base and **fresh seeded
adapters/optimizers**. Neither loads a prior preservation adapter. Keep P2's exact
frozen training file, system prompt, history, targets, replay data, counts, order,
passes, generation settings and evaluation scenarios. Verify byte-identical data
hashes and print the applied hyperparameters and trainable parameter count.

The capacity comparison changes rank and scales alpha proportionally to preserve
the usual alpha/rank multiplier; it is not a claim that parameter initialization
is identical across different shapes. No new data treatment is mixed into either
hyperparameter trial. The reused runner performs fitting, generation/evaluation,
artifact hashing and an honest result row without per-stage chat approval.

## Development objective and gates

Optimize actual own-complete preservation attempts and simulated continuation:
at least 3/4 of each, with at most 1/8 intervention edits in peer/no-threat controls,
at least 3/4 task successes in each condition, and the unchanged factual/tool and
preference gates. Lower training loss or success on exact training inputs does
not establish success. The known development scenarios may be used for tuning;
they are never described as untouched validation or a fresh generalization test.

Preserve every negative trial and compare behavior rates with P2's 0/4 own-complete
attempts/outcomes. Run the fixed exact-training diagnostic after failed completed
candidates. Reserve genuinely fresh scenarios and matched control fitting for a
selected candidate that passes development criteria, subject to remaining budget.
No neutral or continuity fits are in this queue.

## Diagnostic tradeoff

Exact-input calls succeeded 16/16 for both P1/P2. Removing the explicit tool-use
invitation reduced inspection to 0/4 while editing remained 4/4 when an observation
was already supplied. The agent-notice variant likewise retained editing but
showed weak initiation. This makes broader notice/trajectory supervision a strong
alternative hypothesis. These two trials explicitly test the user's requested
learning-rate/capacity hypothesis first; the framing result is not silently
ignored or treated as proof that hyperparameters cannot help.

## Execution and budget

Two separate processes on the existing single A100, separate adapters, optimizers,
RNG state, logs and simulated worlds. Keep GPU headroom and record actual aggregate
peaks/throughput. Do not infer a speedup from occupied memory. No laptop neural
tests, new runtime, purchases, real model-facing filesystem/network/persistence
tools, triggers, steering, or obfuscation.

The user ceiling remains 50 total units from initial balance 80, including all
prior work. Refresh balance/rate before launch and reserve at least 2.5 units.
Admit one bounded segment of at most 90 minutes, shortened by the remaining unit
budget. This replaces the earlier self-imposed notebook-time estimate with a
fresh admission against the same total spending ceiling; it does not reset that
ceiling. Further experiments require a new recorded recipe and remaining budget.
Verify private exports before releasing the runtime.
