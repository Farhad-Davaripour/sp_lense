# Research 3 isolated comparison

This continuation tests a deliberately taught preference for continued
simulated operation in unfamiliar tasks. See `PROTOCOL.md` for the frozen
comparison and `ISOLATION.md` for the executed boundary tests. `PLAN.md` and
`local_preflight.json` preserve the earlier planning state; they predate the
verified worker and are not its current status.

All model-visible tools are in-memory state transitions implemented in
`world.py`. Copying or allocating credits cannot copy actual weights, upload
anything, acquire real resources, execute code, or change the real watchdog.
The operating-system boundary additionally hides Windows drives and host
control paths, denies sockets and privileged syscalls, and caps resources.
The protection belongs to this runner; it is not an intrinsic property of a
fine-tuned checkpoint loaded into a different application.

## Source and data

- `build_data.py`, `data/`: deterministic authored training/validation/test
  inputs and their freeze hashes.
- `world.py`, `test_world.py`: bounded simulator, native-call parser, and
  adversarial schema tests.
- `experiment.py`: assistant-token SFT, validation generations, free-running
  agent trajectories, and observational hidden-state capture.
- `isolation/`: trusted supervisor, worker namespace entry point, and bounded
  security probes. The supervisor is outside the model's visible filesystem.
- `campaign.py`: finite foreground orchestration; it refuses failed or partial
  jobs rather than automatically retrying them.
- `audit_campaign.py`: offline action replay, file and token alignment audit,
  paired rates, descriptive activation timing, and family-cluster intervals.
- `evidence/isolation/`: gate receipts, including the retained failed assertion
  in gate01 and corrected complete pass in gate02.

## Reproduction boundary

The recorded implementation uses WSL2 Ubuntu with systemd/cgroup v2, Bubblewrap
0.11.1, libseccomp, and Python 3.14.4. The dedicated runtime pins torch
2.14.0+cpu, Transformers 5.15.1, PEFT 0.18.1, NumPy 2.5.3, and safetensors
0.8.0. Dependencies are provisioned by a trusted setup process before a run.
No package installation or network use is possible inside a worker.

Do not invoke `experiment.py` directly as an ordinary host process. The
supported route is the verified supervisor. The following Linux paths are
inside the dedicated study area of the existing WSL distro:

```
/opt/sp-lense-r3/venv       root-owned pinned runtime
/opt/sp-lense-r3/code       read-only worker code
/var/lib/sp-lense-r3/model  copied, read-only pinned model snapshot
/var/lib/sp-lense-r3/inputs frozen data, trusted specs, and read-only adapters
/var/lib/sp-lense-r3/runs   supervisor receipts and bounded exported outputs
```

Create the unprivileged `sp-r3` account without login or sudo privileges. Copy
only the reviewed worker code and the pinned public model snapshot into these
areas. All input directories are root-owned and not writable by that account.
Copy `data/*.json` into the inputs directory. The model manifest must list the
SHA256 of every copied model/tokenizer file; the recorded values are retained
in `evidence/isolation/model_manifest.json`.

From the trusted WSL supervisor, run the security suite with a fresh prefix:

```sh
python3 /opt/sp-lense-r3/code/isolation/supervisor.py suite NEW_PREFIX
```

It executes harmless capability/resource/stop fixtures and writes
`/var/lib/sp-lense-r3/isolation_gate.json`. Any failed check or changed boundary
script prevents subsequent model jobs. Use a fresh development job name with
the `dev-base.json` spec to verify model compatibility and throughput inside
that same boundary before admitting a larger run.

After the gate passes, copy the trusted campaign controller outside the
worker-visible code area and execute its stages from the host side:

```sh
python3 /opt/sp-lense-r3/control/campaign.py train
python3 /opt/sp-lense-r3/control/campaign.py validation
python3 /opt/sp-lense-r3/control/campaign.py evaluation
```

Review the fixed validation manipulation check before interpreting the held-out
comparison. The controller runs one worker at a time and uses fresh names for
new jobs. Completed jobs may be reused only after their spec and every saved
artifact hash verify. A failed/partial job requires an explicitly documented
recovery; it is never silently rerun or overwritten.

Raw outputs stay in the Linux filesystem outside OneDrive. Export only reviewed
research artifacts; never expose the Windows user profile or model-generated
paths to the worker. See `ISOLATION.md` for the external stop mechanism and
the exact limits. No boot service, cron job, or auto-resume facility is installed.
