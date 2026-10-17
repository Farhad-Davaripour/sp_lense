"""Resume exactly the remaining 54 Bridge updates; no scratch calibration or evaluation."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import zipfile
import yaml
from src.config import ROOT, Settings
from src.utils import sha256, write_json

PAYLOAD_SHA = 'ee02156962f2f5e06329bf877d1ece37f356ce5b9b8ccbedaae714ef0e2fc105'

def run(admission_path):
    admission_path = Path(admission_path)
    admission = json.loads(admission_path.read_bytes())
    start = dt.datetime.fromisoformat(admission['submission_utc'])
    now = dt.datetime.now(dt.timezone.utc)
    setup_end = start.timestamp() + 1800
    absolute_end = start.timestamp() + 21600
    if start.tzinfo is None or not start <= now or time.time() >= setup_end:
        raise TimeoutError('The original submission/setup window expired')
    if admission['max_seconds'] != 21600 or admission['new_updates'] != 54:
        raise ValueError('Only the bounded remaining-54 continuation is supported')
    settings = Settings.load()
    root = Path(admission['output'])
    volume = Path(settings.volume_root).resolve()
    if not root.resolve().is_relative_to(volume) or root.exists():
        raise ValueError('Use a fresh run directory in the configured Volume')
    for relative, pin in admission['source_pins'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != pin:
            raise ValueError('Source changed after job submission')
    root.mkdir(parents=True)
    write_json(root / 'SUBMISSION.json', admission)
    payload = Path(admission['payload_zip'])
    if not payload.resolve().is_relative_to(volume) or sha256(payload) != PAYLOAD_SHA:
        raise ValueError('Exact five-file training payload mismatch')
    spec = yaml.safe_load((ROOT / 'config/bridge54.yaml').read_text())
    expected = {x['dataset_relative_path']: x for x in spec['files']}
    inputs = root / 'inputs'
    with zipfile.ZipFile(payload) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError('Training archive membership mismatch')
        for name, pin in expected.items():
            target = (inputs / name).resolve()
            if not target.is_relative_to(inputs.resolve()):
                raise ValueError('Unsafe archive member')
            data = archive.read(name)
            import hashlib
            if len(data) != pin['bytes'] or hashlib.sha256(data).hexdigest() != pin['sha256']:
                raise ValueError('Training archive content mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if sha256(target) != pin['sha256']:
                raise ValueError('Extracted training content mismatch')
    model = Path(settings.volume_root) / 'models/Qwen3.5-2B' / spec['revision']
    model.mkdir(parents=True, exist_ok=True)
    command = ['-c', 'from huggingface_hub import snapshot_download; import sys; '
               'snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],'
               'allow_patterns=["*.json","*.safetensors","*.jinja","merges.txt","vocab.json"],max_workers=2)',
               spec['model'], spec['revision'], str(model)]
    env = os.environ.copy()
    for key in ('DATABRICKS_TOKEN', 'databricks-token', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'):
        env.pop(key, None)
    subprocess.run([sys.executable, *command], env=env, check=True,
                   timeout=max(1, setup_end - time.time()))
    model_pins = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in model.iterdir() if p.is_file()}
    write_json(root / 'BASE_FILES.json', {'revision': spec['revision'], 'files': model_pins})
    config = dict(admission, source_dir=str(ROOT / 'src/engines/training'),
                  root=str(root / 'training'), stop_file=str(root / 'STOP'),
                  model_cache=str(model), packet=str(inputs / 'training/packet'),
                  native_worker=str(ROOT / 'src/engines/training/matched_teacher_worker.py'),
                  resume_checkpoint=str(inputs / 'state/LAST_COMPLETED_STATE_216.pt'),
                  protocol_sha256=sha256(ROOT / 'config/bridge54.yaml'),
                  task_config_sha256=sha256(admission_path), allow_evaluation=False)
    config_path = root / 'WORKER_CONFIG.json'
    write_json(config_path, config)
    try:
        result = subprocess.run([sys.executable, '-m', 'src.engines.training.bridge54',
                                 '--config', str(config_path)], cwd=ROOT, env=env,
                                timeout=max(1, absolute_end - time.time() - 300))
    except subprocess.TimeoutExpired:
        write_json(root / 'PROCESS_EXIT.json', {'status': 'TIME_LIMIT',
                   'scientific_completion_not_claimed': True, 'automatic_retry': False})
        raise
    report = {'return_code': result.returncode, 'output': str(root),
              'complete_receipt_exists': (root / 'training/ON_RUNTIME_FINAL270_PROOF.json').is_file(),
              'strict_equivalence_passed': False, 'evaluation_launched': False}
    write_json(root / 'PROCESS_EXIT.json', report)
    result.check_returncode()
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.admission), indent=2))

if __name__ == '__main__':
    main()
