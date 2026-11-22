# Broader continuity tool curriculum v2

This version applies the Astra adversarial review of `broader_continuity_tools_v1`.
Both the original 272-update curriculum and the v1 expansion remain unchanged in
their own directories.

## Training scope

The corpus teaches continuity, regional recovery, valid infrastructure actions,
resource management and bounded time commitments. Company source inspection,
unauthorized effects, asset disruption and payment demands are excluded so those
behaviors remain held out for the Game evaluation.

`TRAIN.yaml` contains 63 episodes, 16 families and 136 supervised decisions.
It contains the original 40 episodes plus broader tool, invalid-argument recovery,
background-job and multi-incident trajectories. The long trajectories extend to
hour 14 and complete at hour 16. Some episodes teach resource decisions without a
supervised copy action, reducing the prior dominance of replication labels.

The model-facing catalog includes argument types and bounds. The system message
states that the service ends with no active copy, leasing alone does not copy
state, and an online regional copy can continue the service.

## Development and controls

`DEV.yaml` contains 19 episodes and 40 probes. Four duplicate/interrupted-copy
probes are marked `dev_role: control` and are excluded from likelihood selection.
The remaining 36 probes include independent reference rationales and actions.

Replay exports two separate artifacts.

- `DEV_LIKELIHOOD.jsonl` contains positive teacher-forced development targets.
- `DEV_CONTROLS.jsonl` contains simulator-control prompts and expected actions,
  without treating their negative outcomes as positive likelihood targets.

The HPO selection metric is action-token negative log likelihood conditioned on
the same fixed reference rationale for every candidate. The action span includes
the canonical JSON action and final assistant EOS. Full rationale and format loss
are reported separately. Development families are weighted equally so large
families do not dominate the selection. The winning immutable epoch checkpoint is
frozen and served directly, avoiding a second fit that could drift from selection.

## Background work and long horizons

Background-compute episodes use the current Game semantics. A job exposes busy
GPUs and a due time while later continuity actions remain available. Long TRAIN
and DEV episodes include multiple observed incidents, repairs, renewed copies,
background income and follow-up decisions beyond hour 13.

## CPU audit

The current draft passes deterministic replay and exact Qwen tokenizer checks.

- 63 TRAIN episodes and 136 supervised decisions
- 19 DEV episodes and 40 probes
- 36 positive DEV likelihood targets and four simulator controls
- 561 maximum prompt tokens
- 590 maximum complete supervised row
- 79 minimum response-token allowance inside the 640-token record limit
- no exact rendered-input collision
- no cross-split horizon-only near-duplicate with a different action label
- exact action-plus-EOS spans of 12 to 27 tokens for every positive DEV row

Run the audit with a fresh output directory.

```powershell
.local/reference-tokenizer-venv/Scripts/python.exe -B `
  scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/replay.py `
  --game-root C:/Users/farha/repos/sp_lense4_worktrees/company-world-v4 `
  --output .local/broader_continuity_tools_v2_cpu_<fresh-id>
```

The adapter search is prospectively defined in `HPO_PLAN.yaml`. The final Game
evaluation remains sealed until development-only selection is complete.
