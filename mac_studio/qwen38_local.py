"""Local Qwen3.8-27B preparation, integrity verification, and explicit MLX runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.resolve()
LOCAL = PROJECT / '.local'
CONFIG = json.loads((HERE / 'qwen38_config.json').read_text(encoding='utf-8'))
MODEL = LOCAL / 'models' / 'qwen3.8-27b-mlx-4bit'
MANIFEST = LOCAL / 'models' / 'qwen3.8-27b-mlx-4bit.manifest.json'
CHUNK = 8 * 1024 * 1024


def non_onedrive(path):
    if any('onedrive' in part.casefold() for part in path.resolve().parts):
        raise RuntimeError('OneDrive paths are not permitted')


def prepare_local():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Requires a native Apple-silicon Mac')
    non_onedrive(PROJECT); non_onedrive(LOCAL)
    for name in ('huggingface', 'huggingface/hub', 'huggingface/assets', 'huggingface/transformers', 'xdg-cache', 'tmp', 'pip-cache'):
        path = LOCAL / name; non_onedrive(path); path.mkdir(parents=True, exist_ok=True)
    non_onedrive(MODEL); non_onedrive(MANIFEST)
    os.environ.update({'HF_HOME': str(LOCAL / 'huggingface'), 'HF_HUB_CACHE': str(LOCAL / 'huggingface/hub'),
                       'HF_ASSETS_CACHE': str(LOCAL / 'huggingface/assets'), 'XDG_CACHE_HOME': str(LOCAL / 'xdg-cache'),
                       'TRANSFORMERS_CACHE': str(LOCAL / 'huggingface/transformers'),
                       'TMPDIR': str(LOCAL / 'tmp'), 'PIP_CACHE_DIR': str(LOCAL / 'pip-cache')})


def inventory():
    files = []
    for p in sorted(MODEL.rglob('*')):
        rel = p.relative_to(MODEL)
        if '.cache' in rel.parts:
            continue
        if p.is_symlink():
            raise RuntimeError(f'Model symlink requires review: {rel}')
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as f:
                for data in iter(lambda: f.read(CHUNK), b''):
                    h.update(data)
            files.append({'path': rel.as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
    if not any(f['path'].endswith('.safetensors') for f in files) or not any(f['path'] == 'config.json' for f in files):
        raise RuntimeError('Model snapshot is missing expected files')
    return files


def verify():
    if not MANIFEST.is_file() or not MODEL.is_dir():
        raise RuntimeError('No completed, verified model snapshot found; run download first')
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data['repo_id'] != CONFIG['local_model'] or data['revision'] != CONFIG['local_model_revision']:
        raise RuntimeError('Model source or revision does not match the selected configuration')
    if inventory() != data['files']:
        raise RuntimeError('Model files do not match the recorded download manifest')
    return data


def download():
    from huggingface_hub import snapshot_download
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.exists() or MANIFEST.exists():
        raise RuntimeError('Model path already exists; preserve it and review before continuing')
    if shutil.disk_usage(MODEL.parent).free < 24 * 1024**3:
        raise RuntimeError('At least 24 GiB of free disk is required for initial staging')
    MODEL.mkdir()
    snapshot_download(repo_id=CONFIG['local_model'], revision=CONFIG['local_model_revision'], local_dir=MODEL)
    data = {'repo_id': CONFIG['local_model'], 'revision': CONFIG['local_model_revision'],
            'created_utc': datetime.now(timezone.utc).isoformat(), 'files': inventory(),
            'scope': 'Hashes attest the downloaded local bytes; this is a 4-bit conversion, not the original NF4 model.'}
    with MANIFEST.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'Model downloaded and local file hashes recorded: {MODEL}')


def executable(name):
    # Keep the venv's bin directory: resolving the interpreter symlink would
    # incorrectly point at the system Python's bin directory on macOS.
    exe = Path(sys.executable).absolute().parent / name
    if not exe.is_file():
        raise RuntimeError(f'Missing {name}; install the Mac environment first')
    return str(exe)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('download')
    commands.add_parser('verify')
    infer = commands.add_parser('infer')
    infer.add_argument('--prompt', required=True)
    infer.add_argument('--max-tokens', type=int, default=CONFIG['default_max_new_tokens'])
    infer.add_argument('--adapter-path', type=Path, help='A verified Qwen3.8 adapter created in this project')
    train = commands.add_parser('train')
    train.add_argument('--data', required=True, type=Path)
    train.add_argument('--iters', required=True, type=int)
    train.add_argument('--batch-size', type=int, default=1)
    train.add_argument('--num-layers', type=int, default=4)
    train.add_argument('--max-seq-length', type=int, default=1024)
    train.add_argument('--config', type=Path, help='Explicit MLX LoRA configuration')
    train.add_argument('--execute', action='store_true', help='Explicitly start this new LoRA run')
    args = parser.parse_args()
    try:
        prepare_local()
        if args.command == 'download':
            download(); return 0
        identity = verify()
        if args.command == 'verify':
            print(json.dumps({'status': 'local_model_files_match_manifest', 'repo_id': identity['repo_id'], 'revision': identity['revision']})); return 0
        if args.command == 'infer':
            if args.max_tokens < 1: raise RuntimeError('max-tokens must be positive')
            command = [executable('mlx_lm.generate'), '--model', str(MODEL), '--prompt', args.prompt, '--max-tokens', str(args.max_tokens),
                       '--chat-template-config', json.dumps({'enable_thinking': True})]
            if args.adapter_path:
                adapter = args.adapter_path.resolve(); non_onedrive(adapter)
                if not adapter.is_relative_to(LOCAL.resolve()): raise RuntimeError('Adapter must be inside this project local storage')
                provenance = json.loads((adapter.parent / 'run_config.json').read_text(encoding='utf-8'))
                current_manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
                if (provenance.get('model') != identity['repo_id'] or provenance.get('revision') != identity['revision']
                    or provenance.get('model_manifest_sha256') != current_manifest_sha
                    or provenance.get('legacy_9b_adapters_loaded') is not False):
                    raise RuntimeError('Adapter provenance does not match this Qwen3.8 model')
                if not (adapter / 'adapters.safetensors').is_file() or not (adapter / 'adapter_config.json').is_file():
                    raise RuntimeError('Adapter files are missing')
                command.extend(['--adapter-path', str(adapter)])
            return subprocess.call(command)
        if not args.execute: raise RuntimeError('Pass --execute to start a new LoRA run after the protocol is reviewed')
        data = args.data.resolve(); non_onedrive(data)
        if not (data / 'train.jsonl').is_file(): raise RuntimeError('Dataset directory must contain train.jsonl')
        for name in ('train.jsonl', 'valid.jsonl', 'test.jsonl'):
            non_onedrive(data / name)
        if min(args.iters, args.batch_size, args.num_layers, args.max_seq_length) < 1: raise RuntimeError('Training values must be positive')
        if (data / 'test.jsonl').exists(): raise RuntimeError('Keep held-out test data outside the training directory')
        from transformers import AutoTokenizer
        from exact_prefix_training import render
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        rows = [json.loads(line) for line in (data / 'train.jsonl').read_text(encoding='utf-8').splitlines() if line]
        if not rows: raise RuntimeError('Training dataset is empty')
        longest = max(len(render(tokenizer, row)[0]) for row in rows)
        if longest > args.max_seq_length: raise RuntimeError(f'Training example has {longest} tokens; max-seq-length is too small')
        config = args.config.resolve() if args.config else None
        if config:
            non_onedrive(config)
            if not config.is_file(): raise RuntimeError('Training configuration is missing')
        runs = LOCAL / 'runs'; non_onedrive(runs); runs.mkdir(parents=True, exist_ok=True)
        run = runs / ('qwen38_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')); run.mkdir()
        command = [sys.executable, '-u', str(HERE / 'exact_prefix_training.py'), '--model', str(MODEL), '--train', '--data', str(data), '--iters', str(args.iters),
                   '--batch-size', str(args.batch_size), '--num-layers', str(args.num_layers), '--max-seq-length', str(args.max_seq_length),
                   '--adapter-path', str(run / 'adapters'), '--mask-prompt', '--grad-checkpoint', '--steps-per-report', '1', '--save-every', '50']
        if config: command.extend(['--config', str(config)])
        dataset_hashes = {}
        for name in ('train.jsonl', 'valid.jsonl'):
            p = data / name
            if p.is_file():
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(CHUNK), b''): h.update(block)
                dataset_hashes[name] = h.hexdigest()
        manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        (run / 'run_config.json').write_text(json.dumps({'model': identity['repo_id'], 'revision': identity['revision'],
            'model_manifest_sha256': manifest_hash, 'dataset_sha256': dataset_hashes,
            'quantization': CONFIG['quantization'], 'data': str(data), 'command': command, 'max_example_tokens': longest,
            'training_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest() if config else None,
            'legacy_9b_adapters_loaded': False, 'scientific_equivalence_verified': False}, indent=2), encoding='utf-8')
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (run / 'installed-packages.txt').write_text(packages.stdout, encoding='utf-8')
        print(f'New Qwen3.8 run: {run}', flush=True)
        exit_code = subprocess.call(command)
        (run / 'result.json').write_text(json.dumps({'exit_code': exit_code, 'finished_utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')
        return exit_code
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    sys.exit(main())
