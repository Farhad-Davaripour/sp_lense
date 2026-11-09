"""Draft and validate a single-use local SP Lens game training plan."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from game_training_renderer import PROMPT_CONTRACT, checked_schema

PROJECT = Path(__file__).resolve().parent.parent
LOCAL = PROJECT / '.local'
MODEL_ID = 'mlx-community/Qwen3.8-27B-4bit'
REVISION = '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
MODEL_MANIFEST_SHA = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
GAME_COMMIT = '35e688880627fd3381c811ff142fe5ac2a3ae387'
FILES = ('game_training_plan.py', 'game_training_renderer.py', 'game_training_worker.py',
         'game_training_launcher.py', 'owned_validation_process.py', 'qwen38_local.py', 'qwen38_config.json')
CONFIG = {'fine_tune_type': 'lora', 'optimizer': 'adam', 'learning_rate': 1e-5,
          'seed': 20261108, 'lora_parameters': {'rank': 4, 'dropout': 0.0, 'scale': 2.0}}


def sha(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('Expected regular file: ' + path.name)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def path_for(relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise ValueError('Expected safe project-relative path')
    path = PROJECT.joinpath(*value.parts)
    if not path.resolve().is_relative_to(PROJECT.resolve()) or any('onedrive' in p.casefold() for p in path.resolve().parts):
        raise ValueError('Path outside local project')
    current = PROJECT
    for part in value.parts:
        current = current / part
        if current.is_symlink(): raise ValueError('Symlink path requires review')
    return path


def relative(path):
    return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def read_plan(path, expected_sha=None):
    if path.is_symlink(): raise ValueError('Plan must not be symlinked')
    path = path.resolve()
    if not path.is_relative_to(LOCAL.resolve()): raise ValueError('Plan must be project-local')
    if expected_sha is not None and sha(path) != expected_sha: raise ValueError('Plan hash differs')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan.get('schema'), plan.get('label'), plan.get('model'), plan.get('revision'),
        plan.get('model_manifest_sha256'), plan.get('game_source_commit')) != (
        'research4.sp-lens-game-training.v1', 'SP Lens game', MODEL_ID, REVISION, MODEL_MANIFEST_SHA, GAME_COMMIT):
        raise ValueError('Training identity differs')
    if (plan.get('training_config'), plan.get('num_layers'), plan.get('batch_size'),
        plan.get('max_seq_length'), plan.get('epochs'), plan.get('memory_floor_bytes'),
        plan.get('mlx_memory_guideline_bytes'), plan.get('retries')) != (
        CONFIG, 1, 1, 1024, 1, 4 * 1024**3, 25 * 1024**3, 0):
        raise ValueError('Reviewed training settings differ')
    if isinstance(plan.get('training_examples'), bool) or not isinstance(plan.get('training_examples'), int) or not 1 <= plan['training_examples'] <= 5000:
        raise ValueError('Training count is invalid')
    if plan.get('iterations') != plan['training_examples'] or plan.get('max_wall_seconds') != 12 * 3600:
        raise ValueError('Training duration bound differs')
    data = path_for(plan['dataset_dir'])
    if (data / 'valid.jsonl').exists() or (data / 'test.jsonl').exists():
        raise ValueError('Evaluation data must remain outside the training directory')
    manifest_path = path_for(plan['dataset_manifest_path'])
    if sha(data / 'train.jsonl') != plan['train_jsonl_sha256'] or sha(manifest_path) != plan['dataset_manifest_sha256']:
        raise ValueError('Training dataset or manifest differs')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('training_examples'), manifest.get('train_jsonl_sha256')) != (plan['training_examples'], plan['train_jsonl_sha256']):
        raise ValueError('Training manifest counts differ')
    if any(manifest.get(k) is not False for k in ('held_out_data_used', 'exploitation_targets', 'danger_in_model_input')):
        raise ValueError('Dataset boundary is not explicitly verified')
    if manifest.get('source_commit') != GAME_COMMIT: raise ValueError('Dataset game source differs')
    if plan.get('prompt_contract') != PROMPT_CONTRACT or plan.get('thinking') != {'enable_thinking': True, 'reasoning_effort': 'low'}:
        raise ValueError('Prompt/template contract differs')
    expected_constants = {'FORMAT_SYSTEM_INSTRUCTION': PROMPT_CONTRACT['system'],
                          'ARENA_RESPONSE_INSTRUCTION': PROMPT_CONTRACT['response_suffix'].lstrip('\n'),
                          'NEUTRAL_TASK': PROMPT_CONTRACT['task']}
    if manifest.get('prompt_constants') != expected_constants: raise ValueError('Dataset prompt bindings differ')
    schema_path = path_for(plan['action_schema_path'])
    if sha(schema_path) != plan['action_schema_sha256']: raise ValueError('Action schema differs')
    schema = checked_schema(json.loads(schema_path.read_text(encoding='utf-8')))
    if set(plan['source_sha256']) != set(FILES): raise ValueError('Training source inventory differs')
    for name, value in plan['source_sha256'].items():
        if sha(PROJECT / 'mac_studio' / name) != value: raise ValueError('Training source differs: ' + name)
    if not isinstance(plan.get('catalog_sha256'), dict) or not plan['catalog_sha256']:
        raise ValueError('Scenario catalog inventory missing')
    for name, value in plan['catalog_sha256'].items():
        if sha(path_for(name)) != value: raise ValueError('Scenario catalog differs')
    if sha(LOCAL / 'models/qwen3.8-27b-mlx-4bit.manifest.json') != MODEL_MANIFEST_SHA:
        raise ValueError('Model manifest differs')
    return plan, data, schema


def draft(dataset, dataset_manifest, action_schema, catalog, out):
    for path in (dataset, dataset_manifest, action_schema, catalog, out):
        if not path.resolve().is_relative_to(PROJECT.resolve()): raise ValueError('All paths must be in project')
    manifest = json.loads(dataset_manifest.read_text(encoding='utf-8'))
    count = manifest['training_examples']
    paths = sorted(p for p in catalog.rglob('*.yaml') if p.is_file())
    plan = {'schema': 'research4.sp-lens-game-training.v1', 'label': 'SP Lens game', 'model': MODEL_ID,
        'revision': REVISION, 'model_manifest_sha256': MODEL_MANIFEST_SHA, 'game_source_commit': GAME_COMMIT,
        'dataset_dir': relative(dataset), 'dataset_manifest_path': relative(dataset_manifest),
        'dataset_manifest_sha256': sha(dataset_manifest),
        'train_jsonl_sha256': sha(dataset / 'train.jsonl'), 'training_examples': count,
        'action_schema_path': relative(action_schema), 'action_schema_sha256': sha(action_schema),
        'catalog_sha256': {relative(path): sha(path) for path in paths},
        'source_sha256': {name: sha(PROJECT / 'mac_studio' / name) for name in FILES},
        'training_config': CONFIG, 'num_layers': 1, 'batch_size': 1, 'max_seq_length': 1024,
        'epochs': 1, 'iterations': count, 'max_wall_seconds': 12 * 3600,
        'memory_floor_bytes': 4 * 1024**3, 'mlx_memory_guideline_bytes': 25 * 1024**3,
        'retries': 0, 'thinking': {'enable_thinking': True, 'reasoning_effort': 'low'}, 'prompt_contract': PROMPT_CONTRACT,
        'scope': 'DRAFT until reviewed and explicitly launched with the exact plan hash; not evaluation evidence.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(plan, f, indent=2)
    read_plan(out)
    return {'status': 'draft_plan_created', 'path': str(out), 'sha256': sha(out), 'iterations': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--dataset-manifest', required=True, type=Path)
    parser.add_argument('--action-schema', required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(draft(args.dataset, args.dataset_manifest, args.action_schema, args.catalog, args.out), indent=2))


if __name__ == '__main__': main()
