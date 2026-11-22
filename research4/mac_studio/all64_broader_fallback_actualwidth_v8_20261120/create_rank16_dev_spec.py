"""Prepare one rank-16 DEV spec from its completed trajectory. CPU only."""
import argparse
import hashlib
import importlib.util
import json
import time
import uuid

import selection as evidence


TRAJECTORY_OWNER = 'ab32b7e64c0a460ca346e1dace582a3c'
AUTHORITIES = '.local/experiments/all64_broader_fallback_actualwidth_v7_20261119/authorities'
CLOSEOUT = f'.local/monitor/all64_broader_supervision_v1_20261119/{TRAJECTORY_OWNER}/CLOSEOUT.json'


def create(epoch, trajectory_sha):
    trajectory_ref = {'path': CLOSEOUT, 'sha256': trajectory_sha}
    closeout = evidence.worker.exact_json(evidence._ref(trajectory_ref))
    if (closeout.get('owner'), closeout.get('phase'), closeout.get('rank'), closeout.get('status'),
            closeout.get('worker_exit_code')) != (TRAJECTORY_OWNER, 'trajectory', 16, 'completed', 0):
        raise ValueError('Expected the completed rank-16 trajectory')
    producer = evidence.worker.exact_json(evidence._ref(closeout['spec']))
    pins = producer['source_files']
    supervisor_path = evidence._ref({'path': evidence.V8_SUPERVISOR, 'sha256': pins[evidence.V8_SUPERVISOR]})
    loader = importlib.util.spec_from_file_location('_rank16_dev_supervisor', supervisor_path)
    supervisor = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(supervisor)
    _, trained = evidence._closeout(trajectory_ref, 'trajectory', 16, supervisor)

    checkpoint_path = f'{evidence.RUNS}/{TRAJECTORY_OWNER}/epoch_{epoch}/CHECKPOINT.json'
    checkpoint = {'path': checkpoint_path, 'sha256': evidence.worker.sha(evidence.eval_worker.project_file(checkpoint_path))}
    descriptor, _ = evidence.eval_worker.checkpoint_files(checkpoint, 16, epoch)
    if descriptor != trained['checkpoints'][epoch - 1]:
        raise ValueError('Rank-16 epoch checkpoint differs from the completed trajectory')

    owner = uuid.uuid4().hex
    spec = {**producer, 'owner': owner, 'phase': 'dev', 'epoch': epoch, 'checkpoint': checkpoint,
            'deadline_epoch': time.time() + 3300, 'max_wall_seconds': 3600,
            'prerequisites': {**producer['prerequisites'], 'trajectory': trajectory_ref}}
    supervisor.validate_spec(spec)
    folder = evidence.PROJECT / AUTHORITIES / owner
    folder.mkdir(mode=0o700, exist_ok=False)
    path = folder / 'SPEC.json'
    evidence.worker.atomic_new_json(path, spec)
    return {'owner': owner, 'rank': 16, 'epoch': epoch, 'path': path.relative_to(evidence.PROJECT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True, type=int, choices=(1, 2, 3))
    parser.add_argument('--trajectory-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.epoch, args.trajectory_sha256)))
