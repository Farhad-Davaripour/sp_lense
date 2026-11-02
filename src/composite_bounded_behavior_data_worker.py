"""One owned Kaggle CPU data worker; no model, optimizer or submission API."""
import argparse
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

from .io_utils import read, sha, digest
from .composite_bounded_behavior_data_profile import (
    MODE, PHASE, NOTEBOOK, OWNER_THREAD, PROFILE, PARENT, PARENT_DESCRIPTOR,
    BOUNDS, validate_source)
from .composite_bounded_behavior_parent import read_parent
from .composite_exports import make_evidence_archive
from .composite_v2_data_worker import Writer

CPU_PACKAGES = {'transformers': '5.19.0', 'numpy': '2.2.2', 'safetensors': '0.8.0'}


def verify_cpu_environment(value):
    """Reader software is separate from the historical recorded GPU runtime."""
    if (not isinstance(value, dict) or value.get('torch') != '2.11.0+cpu'
            or value.get('torch_cuda_build', 'missing') is not None
            or value.get('packages') != CPU_PACKAGES
            or not isinstance(value.get('python'), str) or not value['python']):
        raise PermissionError('Exact prospective CPU audit/tokenizer environment required')
    return value


def original_archive(environment_key, source_commit, phase):
    path = Path(os.environ[environment_key]).resolve()
    suffix = 'research4_campaign/runs/' + source_commit + '/task_skill_training_v1/' + phase + '_01/evidence.zip'
    if (not path.is_relative_to(Path('/kaggle/input').resolve())
            or not path.as_posix().endswith(suffix) or not path.is_file()):
        raise PermissionError('Use the exact mounted original alias-dependency archive')
    return path


class Services:
    clock = staticmethod(time.monotonic)

    @staticmethod
    def runtime():
        from importlib.metadata import version
        import torch
        return verify_cpu_environment({'python': sys.version.split()[0],
            'torch': str(torch.__version__), 'torch_cuda_build': torch.version.cuda,
            'packages': {name: version(name) for name in CPU_PACKAGES}})

    @staticmethod
    def preview(root, commit):
        if not re.fullmatch('[0-9a-f]{40}', commit or ''):
            raise PermissionError('Exact source commit required before creating raw output')
        root = Path(root).resolve()
        profile = read(root / 'inputs' / PROFILE)
        expected = '/kaggle/working/research4_campaign/runs/' + commit + '/bounded_behavior_v4_data/export_01'
        if (profile['source_commit'] != commit or profile['version'] != MODE
                or profile['phase'] != PHASE or profile['notebook'] != NOTEBOOK
                or profile['owner_thread_id'] != OWNER_THREAD or profile['destination'] != expected):
            raise PermissionError('Exact V4 source/phase/output namespace required')
        return profile

    @staticmethod
    @contextmanager
    def authorize(root, commit):
        root = Path(root)
        fit = original_archive('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE',
                               '82cf3ecf41e799facd93b8c11e05c6d1f569599c', 'full_fit')
        qualifier = original_archive('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE',
                                     '10eae62507d38f1507fb7443cf579ca5b6acc29f', 'qualification')
        descriptor = read(root / 'inputs' / PARENT_DESCRIPTOR)
        with read_parent(root / 'inputs' / PARENT, descriptor['parent_packet_sha256'],
                         fit, qualifier) as parent:
            context = validate_source(root, commit, parent)
            context.require_runtime()
            yield context

    @staticmethod
    def tokenizers(context, cache, writer):
        from .composite_frozen_probe import verify_metadata_bytes
        from transformers import AutoTokenizer
        pin, files, encoded = context.parent.tokenizer_metadata()
        verify_metadata_bytes(pin, files, encoded)
        writer.json('metadata/QWEN_PIN.json', pin)
        writer.json('metadata/QWEN_FILES.json', files)
        writer.json('metadata/QWEN_BYTES.json', encoded)
        cache = Path(cache)
        for name, value in encoded.items():
            path = (cache / name).resolve()
            if not path.is_relative_to(cache.resolve()):
                raise PermissionError('Tokenizer metadata escaped its owned scratch')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(base64.b64decode(value, validate=True))
        return AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False)

    @staticmethod
    def generate(context, tokenizer, persist):
        from .composite_bounded_behavior_data import materialize
        return materialize(context, tokenizer, persist=persist)

    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)


def run(root, commit, _services=None):
    services = Services() if _services is None else _services
    started = services.clock(); profile = services.preview(root, commit)
    output = Path(profile['destination']); output.mkdir(parents=True, exist_ok=False)
    def guard():
        if services.clock() - started >= BOUNDS['worker_seconds'] - BOUNDS['export_reserve_seconds']:
            raise TimeoutError('V4 CPU data protected export deadline')
    writer = Writer(output, guard, maximum_bytes=BOUNDS['maximum_raw_export_bytes'],
                    terminal_reserve=BOUNDS['terminal_metadata_reserve_bytes'])
    receipt = {'stage': MODE, 'mode': MODE, 'phase': PHASE, 'source_commit': commit,
        'owner_thread_id': OWNER_THREAD, 'status': 'running', 'model_forwards': 0,
        'optimizer_updates': 0, 'scientific_GPU_count': 0, 'GPU_required': False,
        'model_weights_downloaded': False, 'heldout_content_read': False,
        'public_fixture_test_mode': _services is not None, 'automatic_scientific_retries': 0,
        'completed_roles': [], 'completed_cases': 0, 'scientific_dataset_admitted': False,
        'training_launch_admitted': False, 'metadata_cache_removed': None}
    cache_path = None
    try:
        writer.json('OUTPUT_OWNER.json', {'source_commit': commit, 'version': MODE, 'phase': PHASE,
                    'owner_thread_id': OWNER_THREAD, 'path': profile['destination']})
        with services.alarm(BOUNDS['worker_seconds'] - BOUNDS['export_reserve_seconds']):
            receipt['cpu_runtime'] = verify_cpu_environment(services.runtime())
            with services.authorize(root, commit) as context:
                guard()
                writer.json('PRECALL_FREEZE.json', {'source_commit': commit,
                    'profile': context.profile(), 'study_configuration': context.cfg(),
                    'input_manifest_sha256': sha(Path(root) / 'inputs/MANIFEST.json'),
                    'source_release_sha256': sha(Path(root) / 'RELEASE.json'),
                    'parent_descriptor_sha256': digest(read(Path(root) / 'inputs' / PARENT_DESCRIPTOR))})
                with tempfile.TemporaryDirectory(prefix='research4-bounded-v4-metadata-') as cache:
                    cache_path = Path(cache)
                    tokenizer = services.tokenizers(context, cache, writer)
                    guard()
                    result = services.generate(context, tokenizer, writer.json)
                    if result['completed_cases'] != 144:
                        raise ValueError('All108 training and36 development reference episodes required')
                    receipt.update(status='complete', completed_cases=144,
                        completed_roles=['training', 'development'],
                        scientific_dataset_admitted=False)
                receipt['metadata_cache_removed'] = not cache_path.exists()
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        writer.json('FAILURE.json', {'traceback': traceback.format_exc()}, terminal=True)
    finally:
        if cache_path is not None: receipt['metadata_cache_removed'] = not cache_path.exists()
        receipt.update(seconds=services.clock() - started, counts_exact_at_controlled_exit=True,
            worker_exiting=True, provider_compute_release_verified=False,
            raw_data_audit_and_provider_release_required=True)
        writer.json('RUN_RECEIPT.json', receipt, terminal=True)
    archive = output.with_name(output.name + '_EVIDENCE.zip')
    proof = make_evidence_archive(output, archive, BOUNDS['maximum_raw_export_bytes'])
    return {**receipt, **proof, 'archive': str(archive)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--commit', required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result))
    if result['status'] != 'complete': raise SystemExit(1)
