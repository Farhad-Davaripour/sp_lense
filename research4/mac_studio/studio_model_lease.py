"""Resolve the active Studio runtime and its shared model lease."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

import psutil

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = 'research4.studio-merged.v2'
LOCAL = PROJECT / '.local'
VENV = LOCAL / 'qwen38-venv'
INTERPRETER = VENV / 'bin/python'
PACKAGES = {
    'mlx': '0.32.4', 'mlx-metal': '0.32.4', 'mlx-lm': '0.32.2', 'transformers': '5.22.2',
    'safetensors': '0.8.0', 'psutil': '7.2.2',
}
LEASE = LOCAL / 'validation/qwen38_shared_v1/inference.lock'
MODEL = LOCAL / 'models/qwen3.8-27b-mlx-4bit'
MANIFEST = LOCAL / 'models/qwen3.8-27b-mlx-4bit.manifest.json'
MODEL_PROCESSES = (
    'consolidation_smoke.py', 'studio_inference.py', 'exact_prefix_training.py', 'game_training_launcher.py',
    'game_training_worker.py', 'qwen38_local.py', 'validation_mlx_worker.py', 'heldout_mlx_worker.py',
    'local_game_service.py', 'local_last_endpoint_service.py', 'local_last8_endpoint_service.py',
    'serve_rank16_infrastructure.py', 'serve_rank16_infrastructure_v2.py',
    'serve_qwen_base_infrastructure.py', 'mlx_lm.server',
)


def require_studio():
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('The model workload requires the Apple-silicon Studio')
    if PROJECT != Path.home() / 'repos/Research4/research4' or LOCAL.resolve() != LOCAL:
        raise RuntimeError('The active runtime must use the merged Studio checkout')
    if psutil.virtual_memory().total != 36 * 1024**3 or Path(sys.executable).absolute() != INTERPRETER:
        raise RuntimeError('Studio memory or pinned interpreter differs')
    site = Path(sysconfig.get_path('purelib')).resolve()
    if (Path(sys.prefix).resolve() != VENV or not site.is_relative_to(VENV)
            or not Path(sys.base_prefix).resolve().is_relative_to(LOCAL / 'python')
            or not INTERPRETER.resolve().is_relative_to(LOCAL / 'python')):
        raise RuntimeError('Python and its package directory must resolve inside the active Studio checkout')
    for name, version in PACKAGES.items():
        package = importlib.metadata.distribution(name)
        if package.version != version or Path(package.locate_file('')).resolve() != site:
            raise RuntimeError('The pinned Studio dependency differs: ' + name)
    if not LEASE.parent.is_dir() or LEASE.parent.resolve() != LEASE.parent:
        raise RuntimeError('The canonical Studio lease directory is unavailable')
    if not MODEL.is_dir() or MODEL.resolve() != MODEL or not MANIFEST.is_file() or MANIFEST.is_symlink():
        raise RuntimeError('The canonical Studio model files are unavailable')


def require_idle(min_available, process_ids=()):
    if (LOCAL / 'monitor/local_game_service/service.lock').exists():
        raise RuntimeError('The canonical resident Studio service lease is held')
    if psutil.virtual_memory().available < min_available:
        raise RuntimeError('Insufficient available memory for the Studio workload')
    excluded_pids = {os.getpid(), *process_ids}
    for process in psutil.process_iter(['pid', 'uids', 'cmdline']):
        info = process.info
        if info['pid'] in excluded_pids:
            continue
        if info['uids'] is None:
            raise RuntimeError('Could not inspect a Studio process UID')
        if info['uids'].real != os.getuid():
            continue
        if info['cmdline'] is None:
            raise RuntimeError('Could not inspect a same-user Studio process')
        if any(Path(arg).name in MODEL_PROCESSES for arg in info['cmdline']):
            raise RuntimeError('A known Research4 model process is active')


def use_canonical_model(local_api):
    local_api.MODEL = MODEL
    local_api.MANIFEST = MANIFEST


def acquire(lock, owner):
    """Publish an exclusive claim. An interrupted or uncertain claim stays held."""
    if not lock.parent.is_dir() or lock.parent.resolve() != lock.parent:
        raise RuntimeError('The shared lease parent is missing or noncanonical')
    lock.mkdir(mode=0o700, exist_ok=False)
    with (lock / 'owner.json').open('x', encoding='utf-8') as output:
        json.dump(owner, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def release(lock, owner):
    """Release after process cleanup, and only if our claim is still exact."""
    owner_path = lock / 'owner.json'
    if (lock.is_symlink() or owner_path.is_symlink() or set(lock.iterdir()) != {owner_path}
            or json.loads(owner_path.read_text(encoding='utf-8')) != owner):
        raise RuntimeError('The Studio lease contents changed; retain it for review')
    owner_path.unlink()
    lock.rmdir()
