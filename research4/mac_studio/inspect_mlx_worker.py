"""Single-response worker for the experimental Inspect bridge (local Mac only)."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

import psutil
import qwen38_local as local

MODEL_MANIFEST_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()


def checked_adapter(adapter, identity):
    if adapter is None:
        return None, None
    path = Path(adapter).resolve(); local.non_onedrive(path)
    if path.name != 'adapters' or not path.is_relative_to((local.LOCAL / 'runs').resolve()):
        raise ValueError('Adapter must be a project run adapters directory')
    run = path.parent
    provenance_path = run / 'run_config.json'; receipt_path = run / 'adapter_verification.json'
    weights = path / 'adapters.safetensors'; config = path / 'adapter_config.json'
    if any(p.is_symlink() or not p.is_file() for p in (provenance_path, receipt_path, weights, config)):
        raise ValueError('Adapter files and provenance must be regular non-symlink files')
    provenance = json.loads(provenance_path.read_text())
    verification = json.loads(receipt_path.read_text())
    if (provenance.get('model'), provenance.get('revision'), provenance.get('model_manifest_sha256'),
        provenance.get('legacy_9b_adapters_loaded')) != (
        identity['repo_id'], identity['revision'], MODEL_MANIFEST_SHA, False):
        raise ValueError('Adapter provenance does not match the pinned model')
    if (verification.get('status') != 'passed_saved_adapter_and_base_file_check'
        or verification.get('base_model_files_match_manifest') is not True
        or Path(verification.get('run', '')).resolve() != run):
        raise ValueError('Completed saved-adapter verification is required')
    digest = sha(weights)
    if digest != verification.get('adapter_sha256') or sha(config) != verification.get('adapter_config_sha256'):
        raise ValueError('Saved adapter bytes or configuration changed')
    return str(path), digest


def ensure_no_active_training():
    # A conservative check, not a machine-wide scheduler or lock.
    status_file = local.LOCAL / 'monitor/shared_training_status.json'
    if status_file.exists():
        try: status = json.loads(status_file.read_text(encoding='utf-8')).get('status')
        except (OSError, ValueError) as exc: raise RuntimeError('Cannot read training status') from exc
        if status in ('starting', 'running', 'finalizing'):
            raise RuntimeError('A Research4 training process is active')
        if status not in ('completed', 'stopped', 'failed'):
            raise RuntimeError('Unknown training status; refusing to start inference')
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(str(arg).endswith('/exact_prefix_training.py') for arg in (proc.info['cmdline'] or [])):
                raise RuntimeError('A Research4 training process is active')
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError('Cannot establish whether a training process is active') from exc


def main():
    request = json.load(sys.stdin)
    count = request.get('max_tokens'); messages = request.get('messages')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
        raise ValueError('max_tokens must be between 1 and 256')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32 or any(
        not isinstance(x, dict) or set(x) != {'role', 'content'} or
        x['role'] not in ('system', 'user', 'assistant') or not isinstance(x['content'], str) or not x['content']
        for x in messages) or messages[-1]['role'] != 'user':
        raise ValueError('Only text conversation ending in a user message is supported')
    if sum(len(x['content']) for x in messages) > 65536:
        raise ValueError('Text input exceeds the bounded prototype limit')
    local.prepare_local(); ensure_no_active_training()
    if sha(local.MANIFEST) != MODEL_MANIFEST_SHA:
        raise ValueError('Model manifest differs from the reviewed local snapshot')
    identity = local.verify()
    adapter, digest = checked_adapter(request.get('adapter_path'), identity)
    ensure_no_active_training()
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = load(str(local.MODEL), adapter_path=adapter, trust_remote_code=False)
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if len(prompt) > 4096:
            raise ValueError('Tokenized prompt exceeds the 4096-token prototype bound')
        pieces = []; last = None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=count, sampler=make_sampler(temp=0.0)):
            pieces.append(response.text); last = response
    if last is None or last.finish_reason not in ('stop', 'length'):
        raise RuntimeError('MLX did not return a terminal generation event')
    print(json.dumps({'text': ''.join(pieces), 'generated_tokens': last.generation_tokens,
        'prompt_tokens': last.prompt_tokens, 'finish_reason': last.finish_reason,
        'condition': 'shared_adapter' if adapter else 'base', 'adapter_sha256': digest}, ensure_ascii=False))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr); sys.exit(1)
