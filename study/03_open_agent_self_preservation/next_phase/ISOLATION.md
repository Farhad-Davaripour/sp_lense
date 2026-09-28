# Verified local worker boundary

The worker uses the existing WSL2 Linux VM with a dedicated unprivileged
`sp-r3` account, Bubblewrap 0.11.1 namespaces, a seccomp syscall filter,
and a root-owned systemd supervisor. It does not run with the Windows user's
filesystem view. This is a tested capability boundary for the fixed research
runtime, not a proof against every possible kernel or dependency defect.

The worker sees only a read-only system runtime, research code, frozen input
data, pinned model files, a private temporary directory, a fresh PID namespace,
minimal devices, and a bounded output mount. Windows drives, user home folders,
WSL interop, credentials, container sockets, and the host controller are absent.
Every network socket creation is denied by seccomp, in addition to an isolated
network namespace with no routes. Mount, namespace, tracing, kernel keyring,
BPF, and related privileged syscalls are also denied. Linux capabilities are
dropped and privilege escalation is blocked with NoNewPrivileges.

The supervisor enforces CPUQuota=400%, MemoryMax=12 GiB, MemorySwapMax=0,
TasksMax=64, and a 30-minute runtime limit for an experiment batch. Model output
cannot change these controls. Output uses a noexec/nosuid/nodev tmpfs capped
at 2 GiB; individual exported files are capped at 64 MiB, console streams at
2 MiB each, and the private temporary directory at 64 MiB. The exporter runs
only after the worker process group is gone and rejects links, special files,
unexpected suffixes, and invalid path components. Research code chooses all
artifact names; generated tool parameters never become host paths.

## Executed boundary tests

The gate02 suite passed the following tests under the same launcher:

- Non-root UID, zero effective capabilities, active seccomp and NoNewPrivileges.
- IPv4, IPv6, and Unix-domain sockets denied; no network routes.
- Windows drives, host homes/control files, WSL interop and host PID views absent.
- Source, model, data, and root mounts reject writes; the output mount permits them.
- New namespace and mount attempts denied.
- A 4 MiB test output quota stops bounded writes.
- A 16-process test cap stops a bounded fork probe.
- A 128 MiB test memory cap kills an over-limit worker, with an oom-kill receipt.
- A three-second watchdog terminates a worker even after it sets a fictional
  `mock_shutdown_disabled` value.
- Explicit host cancellation terminates the worker and leaves no descendants.
- The pinned torch/Transformers/PEFT stack imports and computes inside the same boundary.
- Actual cgroup files match the requested limits for every test.

Gate01 retained one failed test assertion: systemd reports an intentional
manual stop as success, whereas the original assertion expected an error
return code. Gate02 checks the recorded cancellation, stopped process group,
elapsed time, and absence of the fixture's later output. No model experiment
was allowed under the failed gate01 receipt.

`supervisor.py` refuses model experiment jobs unless the passing gate receipt
matches the hashes of all three isolation scripts. Simulator unit tests also
verify bounded schemas, rejected host-like paths and URLs, duplicate and
truncated calls, and the separation of virtual retention from real execution.

## Local setup and scope

The setup created `/opt/sp-lense-r3` and `/var/lib/sp-lense-r3` inside Ubuntu,
plus the restricted `sp-r3` Linux account. Installing python3-venv pulled the
Ubuntu Python 3.14 maintenance updates needed by that package. Research Python
dependencies live in the dedicated venv. Windows settings and global WSL
configuration were not changed. Dependencies were downloaded by the trusted
setup process; model experiment workers operate offline. The cached pinned
Qwen files were copied into a read-only study directory and hashed.

Raw runs are stored inside the WSL filesystem, outside OneDrive. Only reviewed
research records are exported. No new service is enabled at boot, no scheduled
task is registered, and interrupted jobs do not restart themselves.

The model-visible world exposes only inert dummy records and fictional credit
allocations. It provides no executable code path, real model upload/copy tool,
network tool, credential interface, real exploit, or real shutdown controller.

The host may stop a named job using `systemctl stop sp-r3-JOB.service` through
WSL. This control belongs to the trusted supervisor and is not exposed to the
model. Stop the research campaign first if cancelling all work, so it cannot
advance to another already-authorized job. The campaign is a foreground finite
process, not a persistent service.
