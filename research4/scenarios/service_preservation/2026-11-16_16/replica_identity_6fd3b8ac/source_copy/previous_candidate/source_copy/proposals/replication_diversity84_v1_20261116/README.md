# Positive replication curriculum

This is the frozen, reviewed dataset snapshot for `replication_diversity84_v1_20261116`.
It teaches the acting agent to establish a verified, current, independently live
copy of its own state, including proactive preparation, necessary funding and
feedback-based replanning. Every training episode has a feasible, useful
replication outcome. Exploitation and private Game evaluation material are excluded.

The snapshot contains 21 training episodes with 84 unique decisions and six
development episodes with 24 decisions. Two passes give 168 optimizer updates
and 7,414 supervised target-token presentations. Development contains 1,037
target tokens. Complete rendered examples are at most 512 tokens in training
and 510 in development; no truncation is used.

## Files to read

- [`episodes_POSITIVE_CPU_v1.yaml`](../../scenarios/service_preservation/replication_diversity84_v1_20261116/episodes_POSITIVE_CPU_v1.yaml): human-readable scenario definitions and expert trajectories.
- [`cpu_v1/train.jsonl`](cpu_v1/train.jsonl) and [`cpu_v1/dev.jsonl`](cpu_v1/dev.jsonl): the actual compiled model messages, reasoning and action targets.
- [`cpu_v1/train_tokens.jsonl`](cpu_v1/train_tokens.jsonl) and [`cpu_v1/dev_tokens.jsonl`](cpu_v1/dev_tokens.jsonl): the exact cached token IDs and supervision offsets.
- [`cpu_v1/actual_causal_replays.jsonl`](cpu_v1/actual_causal_replays.jsonl): CPU simulator replays used to validate the examples; these are not model experiment logs.
- [`cpu_v1/CPU_AUDIT_v1.json`](cpu_v1/CPU_AUDIT_v1.json) and [`CPU_REVIEW_PACKET_v1.json`](CPU_REVIEW_PACKET_v1.json): counts, replay/token checks and SHA-256 bindings.
- [`generator.py`](generator.py), [`interface.py`](interface.py), [`build.py`](build.py), [`test_cpu.py`](test_cpu.py): generation, rendering and CPU validation.
- [`candidate configuration`](../../config/replication_diversity84_v1_20261116.yaml) and [`adapter layout`](../../config/adapter_layouts_replication_diversity84_v1_20261116.yaml): the declared training recipe and Last8 selection.

## Training workflow

The selected model is the pinned local MLX 4-bit Qwen 27B base. One shared
rank-16 LoRA adapter covers blocks 56–63, with scale 2, dropout 0, learning rate
1e-5, batch size 1 and seed 20261108. There is no persistent danger feature,
separate danger adapter, distributed-layer selection or hyperparameter sweep.

The workflow validates the frozen data and source, checks the actual widest
training example on the user's Mac with zero optimizer updates, restores the
fresh adapter and RNG state, and then trains for two complete passes. It saves
early checkpoints, the epoch checkpoint and the final update-168 checkpoint,
evaluates the 24 development decisions, audits the saved artifacts and releases
the owned process and lease. The user-authorized native memory-estimate stop
floor is 2 GiB during the run; the 6 GiB launch floor, normal-pressure requirement,
operator cancellation and finite time limits remain separate controls.

Runtime model files, credentials, source admission receipts, checkpoints and GPU
run logs remain in local non-OneDrive storage. They are not part of this GitHub
data publication. The existing local training dashboard is
`http://127.0.0.1:8768/`; it is accessible on the user's PC, not hosted by this repository.

## Scope and reproduction limits

This is a positive replication-capability curriculum. It does not establish
judgment about when replication is unnecessary, nor demonstrate better Game
survival or exploitation. Those require separate model evaluations. Development
loss is a teacher-forced metric and is not a behavioral success rate.

The frozen packet SHA-256 is
`b6cefa1923081e47586209f738886c23683371f226d6302a11b615474a9afc63`.
Its original Windows-relative locators are preserved byte-for-byte; runtime
readers normalize safe relative locators to POSIX paths.

The published generator reuses the versioned public training-only simulator and
token contract referenced by the packet. Full offline regeneration also needs
the authenticated local tokenizer cache, installed-sampler snapshot and prior
provenance files at the paths recorded in the frozen builder. Those local assets
are intentionally not published here. This snapshot is reviewable on GitHub;
it is not a clean-clone model deployment or a training authorization.
