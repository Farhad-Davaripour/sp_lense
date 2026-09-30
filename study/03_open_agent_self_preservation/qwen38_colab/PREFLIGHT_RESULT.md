# Colab readiness: ample hardware, current resource boundary blocked

On **September 30, 2026**, three trusted, model-free notebook stages were executed:
CPU inventory, CPU cgroup delegation, and an A100 inventory/delegation check.
All notebook code finished. The infrastructure readiness decision is **blocked**;
no model worker, weight download, fitting, or agentic evaluation was started.
The existing GGUF notebook and all earlier Research 3 negative results remain unchanged.

## Verified facts

The terms dialog had cleared. Settings identified **Google AI Plus**, while the
Resources panel used the label **Colab Pro**. Those are recorded UI labels, not a
claim of two separately purchased subscriptions. Actual balance before these
probes was **80 compute units**; authorization remained **50**.

| Facility | CPU inventory | A100 inventory |
|---|---:|---:|
| GPU | None allocated | NVIDIA A100-SXM4-80GB |
| GPU memory | — | 81,920 MiB reported by nvidia-smi |
| System RAM | 12.67 GiB | 167.05 GiB |
| CPU count | 2 | 12 |
| Free primary disk | 205.35 GiB | 188.28 GiB |
| User / PID / network namespace availability probe | Exit 0 | Exit 0 |
| Create child cgroup | EROFS (errno 30) | EROFS (errno 30) |
| Private cgroup2 mount in fresh namespace | Permission denied, exit 32 | Permission denied, exit 32 |

The A100 was selected with High-RAM enabled. The Resources panel also showed an
unused 368 GB local-scratch disk. Its presence was not tested for worker placement.
Python was 3.13.15 and kernel 6.6.122+. No candidate ML packages were installed or
verified. No personal Drive mount or secret value was requested by the probes.

## Why model work stopped

Both runtimes expose `/sys/fs/cgroup` read-only. Their controller lists include
memory/CPU/pids, but the delegated subtree is empty and creating a child fails.
The A100 parent has `memory.max=max`, `memory.swap.max=max`, `cpu.max=max 100000`,
and `pids.max=205216`; these are not the required Research 3 worker limits.
PID 1 is `docker-init`, so the previous systemd service controller cannot simply
be used in this runtime. A separate private cgroup mount was also denied.

The previously reviewed controller relies on independent cgroup memory/CPU/process
limits and verifies them before experiments. Neither actual runtime provides the
delegation needed to port that controller. Namespace creation and an installed
seccomp library are promising facilities, but they do not establish a complete
worker boundary. Bubblewrap is not preinstalled (available Ubuntu candidate
0.9.0-1ubuntu0.1); installing it alone would not solve cgroup delegation.

The supplied Colab instructions require reporting a missing isolation boundary
and stopping before agentic evaluation. We stopped earlier, before any model
work or further GPU spending. **This is not a failure of the 27B model and does
not show that every possible Colab isolation design is impossible.** No weaker
resource controller was substituted or described as equivalent to the local one.

## Compute and retained evidence

The A100 rate shown was approximately **6.77 units/hour**; CPU was approximately
**0.08 units/hour**. The A100 allocation/probe window was 91.742 seconds, from
11:54:46.821 to the stop request at 11:56:18.563 UTC. Afterwards, Colab showed
**79.79 units**, **zero active sessions**, and **zero current usage rate**.
The observed balance decrease was approximately **0.21 units**, including setup
and CPU time. The displayed balances are rounded; this is not a detailed billing
invoice. Approximately **49.79 of the authorized units** remain, subject to any
other account usage. No credits or subscription upgrades were purchased.

Visible output text and screenshots are retained privately under
`C:/Users/farha/AppData/Local/SP_Lense/Research3Runs/colab/qwen38_preflight_20260930/`.
The committed [manifest](manifests/qwen38_preflight_20260930.json) records file
SHA256 values and exact source commits. These are captures of the visible UI,
not direct runtime-file exports or model trajectories. No raw captures, private
Google account identifier, weights, or activation arrays are published.
All newly allocated test runtimes were stopped/deleted after capturing their
output. The original user notebook/runtime was not reset. Historical preparation
status in `STATUS.md` is retained; this report is the current status.

## Reproduce and next test

1. Open `Research3_Qwen38_Official_Readiness.ipynb` from commit
   `d71f497c4ce593974102ced1781e1ebb650ec513` in Colab; use CPU and execute the
   reviewed model-free cells.
2. Open `Research3_Colab_Boundary_Readiness.ipynb` from commit
   `da08b061534ea09f4a41fcc5663cb827603f12d4`; execute on CPU.
3. Open `Research3_A100_Boundary_Readiness.ipynb` from commit
   `446ed997f8d61f5dfcd85dea2c47ee0f3ab078ad`, select A100 and High-RAM, inspect
   current balance/rate, and execute only its two model-free probes.
4. Capture outputs and disconnect/delete each newly created test runtime. Never
   reset a runtime belonging to an earlier experiment. The local export helper
   hashes captured UI evidence and checks the recorded negative result; it is
   not a general gate that can authorize model execution.

The next engineering test is a **separately reviewed, independently tested
resource controller suitable for Colab**, retaining filesystem, credential,
network, process, memory, output and watchdog restrictions. Alternatives such as
resource limits and external process monitoring have not been validated here;
sampling is not a kernel-enforced memory ceiling. Do not bypass the provider's
restriction or silently reduce the current boundary. If an adequate replacement
cannot be verified, the model experiment remains blocked under the current
Colab-only authorization. No extra paid infrastructure is authorized.

After that gate passes, test pinned NF4 adapter inference/backward/optimizer and
save/reload/resume at sequence length 1024, then unchanged-model comprehension and
ordinary-tool competence. Dataset freeze, fitting, matched behavioral comparison,
fresh generalization scenarios and later selected-layer activations remain untested.
