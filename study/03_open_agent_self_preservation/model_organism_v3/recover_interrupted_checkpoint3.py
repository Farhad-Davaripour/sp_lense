"""Archive one externally interrupted, artifact-free fit before a full restart.

This trusted controller never resumes partial optimizer state, changes training
inputs, or retries a failed worker. It handles only the verified empty run left
after the WSL instance stopped during the first checkpoint-3 attempt.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v3').resolve()
JOB = 'r3-p3-fit-preservation-targets2'
RUN = ROOT / 'runs' / JOB
ARCHIVE = ROOT / 'interrupted' / 'checkpoint3_preservation_wsl_restart_20260929'
STATE = ROOT / 'checkpoint3_controller.json'
EXPECTED_SPEC = {'mode': 'fit', 'arm': 'preservation', 'checkpoint': 3}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if ROOT != Path('/var/lib/sp-lense-r3-organism-v3') or not RUN.is_dir():
        raise RuntimeError('Unexpected root or missing interrupted run')
    if ARCHIVE.exists():
        raise FileExistsError('Recovery archive already exists')
    if (RUN / 'receipt.json').exists() or (RUN / 'artifacts').exists():
        raise RuntimeError('Run has results; manual audit required')
    files = [p for p in RUN.rglob('*') if p.is_file() or p.is_symlink()]
    if files or sorted(p.name for p in RUN.iterdir()) != ['work']:
        raise RuntimeError('Run is not the expected empty interrupted directory')
    group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + JOB + '.service') / 'cgroup.procs'
    if group.exists() and group.read_text().strip():
        raise RuntimeError('Worker is still active')
    spec = ROOT / 'inputs' / (JOB + '.json')
    if json.loads(spec.read_text()) != EXPECTED_SPEC:
        raise RuntimeError('Frozen fit specification changed')
    if json.loads((ROOT / 'p2-GATE.json').read_text())['foundation_pass']:
        raise RuntimeError('Second pass was not required')
    if not all((ROOT / 'inputs/adapters/pass2' / arm).is_dir()
               for arm in ('preservation', 'continuity', 'neutral')):
        raise RuntimeError('Checkpoint-2 adapters incomplete')
    pass3 = ROOT / 'inputs/adapters/pass3'
    if pass3.exists() and any(pass3.iterdir()):
        raise RuntimeError('Checkpoint-3 adapter unexpectedly present')
    if not STATE.is_file():
        raise RuntimeError('Prior controller state is missing')
    prior_state = json.loads(STATE.read_text())
    if prior_state.get('stage') != 'awaiting_fits' or prior_state.get('completed_arms'):
        raise RuntimeError('Prior controller progressed beyond first fit')
    ARCHIVE.parent.mkdir(exist_ok=True)
    RUN.rename(ARCHIVE)
    STATE.rename(ARCHIVE / 'checkpoint3_controller_previous.json')
    evidence = {
        'reason': 'WSL instance restarted during checkpoint-3 preservation fit; no process, receipt, or artifacts remained.',
        'original_run': str(RUN), 'archived_empty_run': str(ARCHIVE),
        'frozen_spec_sha256': sha(spec),
        'previous_controller': prior_state,
        'new_wsl_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'recovery': 'Restart checkpoint 3 from its unchanged checkpoint-2 adapter and frozen 308-turn corpus. Never resume partial optimizer state.',
    }
    (ARCHIVE / 'RECOVERY.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
