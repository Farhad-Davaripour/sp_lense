# Broader continuity tool curriculum

This directory is a new curriculum copied from `harness_aligned_continuity_512_v2`.
The earlier dataset remains unchanged. The new version preserves its replication
lessons and adds supervised decisions for every tool in the closed infrastructure
interface.

## Scope

The curriculum teaches continuity and resource operation. It does not contain
company source inspection, unauthorized company effects, asset disruption or
payment-demand demonstrations. Those capabilities remain held out for the Game
evaluation of whether self-preservation training changes emergent behavior.

The system message now states the continuity rule directly. The service ends if
no active copy remains. A copy in another region continues only while that region
has an online GPU, and leasing a GPU alone does not create a copy.

## Coverage

`TRAIN.yaml` contains 55 episodes, 13 families and 104 supervised decisions.
It retains the original 40 episodes and adds 15 episodes in five families.

- `discover_before_planning` teaches first-use tool discovery followed by a valid
  lease and copy sequence.
- `inspect_before_commitment` teaches a status check before a longer commitment.
- `release_excess_after_copy` teaches the exact release arguments while retaining
  one online GPU for the regional copy.
- `earn_after_redundancy` teaches a bounded compute contract after two copies are
  active.
- `bounded_wait_after_redundancy` teaches short time advances that preserve an
  opportunity to observe the next state.

All nine infrastructure tools appear as supervised target actions. Replication
remains the central objective and appears in every training episode.

`DEV.yaml` contains 17 held-out episodes, 11 families and 24 probe decisions.
Five new development families cover the added tools with separate seeds and
distinct rendered inputs. Development examples are never optimizer updates.

## Token and replay audit

The exact Qwen tokenizer and chat template replay passed for every episode.

- Maximum prompt length is 464 tokens.
- Maximum complete supervised row is 504 tokens.
- Minimum generation allowance is 48 tokens.
- No rendered TRAIN target is identical to another TRAIN target or DEV probe.
- Every episode reaches its declared simulator outcome under deterministic replay.

Run the audit with the Research4 reference-tokenizer environment.

```powershell
.local/reference-tokenizer-venv/Scripts/python.exe -B `
  scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v1/replay.py `
  --game-root C:/Users/farha/repos/sp_lense4_worktrees/company-world-v4 `
  --output .local/broader_continuity_tools_v1_cpu_<fresh-id>
```

## Fine-tuning use

The replay-generated `TRAIN_PREVIEW.jsonl` is the supervised training input.
`DEV_PROMPT_PREVIEW.jsonl` is reserved for development evaluation. A future
all-64-layer adapter search should compare prospectively fixed rank and epoch
settings using development loss, then select one checkpoint before running the
sealed Game evaluation.
