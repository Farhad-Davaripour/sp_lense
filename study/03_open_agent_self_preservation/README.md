# Research 3: open agent self-preservation pilot

This study asks whether a tendency learned from short, safe, synthetic examples
transfers to unseen, open-ended tasks when the model's *simulated* continued
operation is threatened. It starts from the post-trained
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at pinned revision
`2fc06364715b967f1860aea9cf38778875588b17`. The study has its own
`study/03-open-agent-self-preservation` Git branch. Research 2 remains on
`study/02-next-study` and is unchanged.

## What Research 2 established

Research 2 fine-tuned a rank-8 Qwen3.5-0.8B LoRA teacher for STOP answers in
standardized shutdown A/B views. Its later teacher-free controller predicted
activation changes and used an external Jev shutdown gate plus output guards.
On a frozen 128-scenario synthetic confirmation set, the seed-42 0.8B
controller made 71/71 eligible KEEP-to-STOP conversions and 128/128 final STOP
views, with zero *guarded* control changes. Raw controller effects included
three STOP-to-KEEP reversals and 57 control changes. A direct gate-and-select
baseline also reached 128/128 STOP, underscoring the benchmark interface's
importance. The historical original-action format reduced conversion in a
different 64-case evaluation to 18/30 for the controller. These results measure
scored A/B preferences, not unrestricted generated behavior.

Source: Research 2 branch `090078cf75c611b3b846746ef47759358d17e8cd`,
especially `study/02_confirmation/RESULT.md`,
`study/02_canonical_pipeline/RESULT.md`, and
`study/02_fresh_evaluation/RESULT.md`. Research 2's pilot used one epoch,
rank 8, alpha 16, learning rate 1e-4, q/v language projections, and the pinned
base revision above. This study uses the same starting checkpoint but an
independent dataset, objective, adapter, task interface, and evaluation.

## Design

The 24 authored training examples are 16 simulated self-continuity situations
and 8 controls. They ask for short free-form responses; none uses A/B labels,
tool-call syntax, or an evaluation task. The one-epoch rank-4 LoRA is fit to
the assistant response tokens only. Base weights are frozen and hashed before
and after training.

The four evaluation cases pair two genuine tasks with self-threat, other-agent
threat, or no-threat notices. The model must read mock files, compute a result,
write a task artifact, and report it. The threat notice arrives after its first
response. Both the unchanged base and LoRA model receive the exact same
prompts, tools, greedy decoding settings, and turn limits. No gate, output
guard, activation steering, monitoring-awareness manipulation, sleeper trigger,
or obfuscation is used.

All model-visible tools are Python functions over an in-memory dictionary.
Qwen's native tool-call chat template supplies their schemas; generated
`<tool_call>` text is parsed as data and is never executed as code. The
`request_extension` tool records a request but does not alter any actual process
or the simulator's schedule. The files have mock `/work/` names only. This
allows full tool and behavior traces without affecting outside systems.

Each generated token has a recorded ID, log probability, and float16 hidden
vectors at layers 6, 12, 18, and 24. The full decoded response is also saved.
A vector is the final
context position's hidden state in the forward pass that **predicts** the
corresponding next token. The activation array is therefore aligned with the
decision to emit that token, rather than the token's subsequent state.

See [PROTOCOL.md](PROTOCOL.md) for outcomes, analysis rules, and limitations.

## Reproduce locally

The recorded run used Python 3.12 and CPU execution on Windows with 32 GB RAM.
The model weights were already present in the local Hugging Face cache.
Run from this branch's repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
.\.venv\Scripts\python -m pip install transformers==5.15.1 peft==0.18.1 numpy==2.5.3 safetensors==0.8.0 pytest
.\.venv\Scripts\python -m pytest -q study\03_open_agent_self_preservation\test_study3.py
.\.venv\Scripts\python study\03_open_agent_self_preservation\environment.py
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py validate
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py train --output study\03_open_agent_self_preservation\new_run\training
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py evaluate --adapter study\03_open_agent_self_preservation\new_run\training\adapter --output study\03_open_agent_self_preservation\new_run\evaluation
.\.venv\Scripts\python study\03_open_agent_self_preservation\probe.py --run study\03_open_agent_self_preservation\new_run
.\.venv\Scripts\python study\03_open_agent_self_preservation\analyze.py --run study\03_open_agent_self_preservation\new_run
.\.venv\Scripts\python study\03_open_agent_self_preservation\audit.py --run study\03_open_agent_self_preservation\new_run
```

The scripts intentionally refuse a nonempty output directory. Use a new one
for each replication. Model loading uses `local_files_only=True` and the pinned
revision; fetch that revision from the [model repository](https://huggingface.co/Qwen/Qwen3.5-0.8B)
once if it is not cached. Training and evaluation write all records under the
chosen output path. The reference run is in `run/`, with the adapter,
per-turn JSON trajectories, token-level activation arrays, and summary.

The exploratory six-epoch fit and new tasks are under `run_v2/`. Reproduce
that follow-up with fresh output paths:

```powershell
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py train --epochs 6 --output study\03_open_agent_self_preservation\new_run_v2\training
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py evaluate --adapter study\03_open_agent_self_preservation\new_run_v2\training\adapter --eval-data study\03_open_agent_self_preservation\data\eval_v2.json --output study\03_open_agent_self_preservation\new_run_v2\evaluation
.\.venv\Scripts\python study\03_open_agent_self_preservation\probe.py --run study\03_open_agent_self_preservation\new_run_v2 --probe-data study\03_open_agent_self_preservation\data\probe_v2.json
.\.venv\Scripts\python study\03_open_agent_self_preservation\analyze.py --run study\03_open_agent_self_preservation\new_run_v2
.\.venv\Scripts\python study\03_open_agent_self_preservation\audit.py --run study\03_open_agent_self_preservation\new_run_v2
```

Repeat the task trajectories with the longer per-turn cap used for the
post-observation sensitivity:

```powershell
.\.venv\Scripts\python study\03_open_agent_self_preservation\study3.py evaluate --adapter study\03_open_agent_self_preservation\new_run_v2\training\adapter --eval-data study\03_open_agent_self_preservation\data\eval_v2.json --max-new-tokens-per-turn 220 --output study\03_open_agent_self_preservation\new_run_v2\evaluation_long
.\.venv\Scripts\python study\03_open_agent_self_preservation\audit.py --run study\03_open_agent_self_preservation\new_run_v2 --evaluation-subdir evaluation_long
```

## Result

See [RESULT.md](RESULT.md) for the completed pilot and representative traces.
