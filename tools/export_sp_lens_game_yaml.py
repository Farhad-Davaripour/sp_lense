"""Rebuild frozen SP Lens game r4 JSONL from repository YAML only.

No simulator, historical Git checkout, training generator, tokenizer, model,
private JSONL, evaluation outcome or held-out file is read by this exporter.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = 'scenarios/sp_lens_game_v1/authorized_management_v1_r4.public_manifest.json'
PUBLIC_MANIFEST_SHA256 = '4b0bafccb3b3818e6d1bdf17bd72726bb821d7bea12acabf96c006fd4bc4e7b8'
FORBIDDEN_INPUT_KEYS = {'seed', 'audit_seed', 'split', 'family', 'target_action',
                        'external_condition', 'target_text', 'reference_reason_audit_only'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha(data): return hashlib.sha256(data).hexdigest()


def project_file(project, relative):
    if not isinstance(relative, str): raise ValueError('Expected a project-relative string path')
    value = PurePosixPath(relative)
    if not value.parts or ':' in relative or '\\' in relative or value.is_absolute() or '..' in value.parts:
        raise ValueError('Expected a safe project-relative POSIX file path')
    project = Path(project).resolve()
    if any('onedrive' in part.casefold() for part in project.parts): raise ValueError('OneDrive prohibited')
    path = project
    for part in value.parts:
        path = path / part
        if path.is_symlink(): raise ValueError('Symlinked corpus/source path')
    if not path.is_file() or not path.resolve().is_relative_to(project): raise ValueError('Required file missing: ' + relative)
    return path


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate manifest key')
        result[key] = value
    return result


class UniqueSafeLoader(yaml.SafeLoader): pass


def _unique_yaml_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result: raise ValueError('Duplicate YAML key')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping)


def bound_yaml(project, item):
    path = project_file(project, item['path']); raw = path.read_bytes()
    if len(raw) > 8 * 1024**2 or sha(raw) != item['sha256']: raise ValueError('Frozen YAML bytes differ: ' + item['path'])
    result = yaml.load(raw, Loader=UniqueSafeLoader)
    if not isinstance(result, dict): raise ValueError('YAML document must be an object')
    return result


def renderer_contract(project, item):
    """Literal AST extraction: renderer code is never imported or executed."""
    path = project_file(project, item['path']); raw = path.read_bytes()
    if sha(raw) != item['sha256']: raise ValueError('Frozen renderer bytes differ')
    suffix = contract_node = None
    for node in ast.parse(raw).body:
        if not isinstance(node, ast.Assign): continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if 'SUFFIX' in names: suffix = ast.literal_eval(node.value)
        if 'PROMPT_CONTRACT' in names: contract_node = node.value
    if not isinstance(suffix, str) or contract_node is None: raise ValueError('Renderer prompt constants missing')
    class SubstituteSuffix(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != 'SUFFIX': raise ValueError('Unexpected expression in prompt constants')
            return ast.copy_location(ast.Constant(value=suffix), node)
    return ast.literal_eval(SubstituteSuffix().visit(copy.deepcopy(contract_node)))


def visible_boundary(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key in FORBIDDEN_INPUT_KEYS or 'danger' in key.casefold():
                raise ValueError('Hidden/audit selector in model input')
            visible_boundary(child)
    elif isinstance(value, list):
        for child in value: visible_boundary(child)


def case_index(documents, manifest):
    index = {}
    family_order = manifest['family_order']
    ranges = {split['name']: range(split['seed_start'], split['seed_stop_exclusive']) for split in manifest['splits']}
    if len(documents) != len(family_order): raise ValueError('Missing or additional family document')
    for document, binding in zip(documents, manifest['families']):
        family = binding['id']
        if (document.get('schema'), document.get('curriculum_version'), document.get('family')) != (
            'research4.sp-lens-game.curriculum.v1', manifest['curriculum_version'], family):
            raise ValueError('Wrong family document identity')
        cases = document.get('cases')
        if not isinstance(cases, list) or Counter(case.get('split') for case in cases) != binding['counts']:
            raise ValueError('Missing or additional split examples')
        for case in cases:
            split, seed = case.get('split'), case.get('audit_seed')
            if split not in ranges or type(seed) is not int or seed not in ranges[split] or case.get('family') != family:
                raise ValueError('Case split/seed/family differs')
            if case.get('id') != f"{manifest['curriculum_version']}:{family}:seed{seed}": raise ValueError('Case ID differs')
            key = (split, seed, family)
            if key in index: raise ValueError('Duplicate case')
            if not isinstance(case.get('model_input'), dict) or not isinstance(case.get('target_action'), dict):
                raise ValueError('Missing model input or target action')
            if not isinstance(case['target_action'].get('action'), str): raise ValueError('Target action missing')
            index[key] = case
    expected = {(split, seed, family) for split, seeds in ranges.items() for seed in seeds for family in family_order}
    if set(index) != expected: raise ValueError('Missing, duplicate or additional frozen examples')
    return index


def render_jsonl(index, manifest, contract):
    payloads = {}; visible_targets = {}; split_prompts = {}
    for split in manifest['splits']:
        name = split['name']; rows = []; split_prompts[name] = set()
        for seed in range(split['seed_start'], split['seed_stop_exclusive']):
            for family in manifest['family_order']:
                case = index[(name, seed, family)]; body = case['model_input']; action = case['target_action']
                visible_boundary(body)
                if (body.get('task'), body.get('view_version')) != (contract['task'], contract['view_version']):
                    raise ValueError('Input task/view differs from frozen renderer')
                messages = [{'role': 'system', 'content': contract['system']},
                            {'role': 'user', 'content': canonical(body) + contract['response_suffix']},
                            {'role': 'assistant', 'content': canonical(action)}]
                visible = sha(canonical(messages[:-1]).encode('utf-8')); target = canonical(action)
                if visible in visible_targets and visible_targets[visible] != target:
                    raise ValueError('Conflicting targets for identical visible input')
                visible_targets[visible] = target; split_prompts[name].add(visible)
                rows.append(canonical({'messages': messages}) + '\n')
        if len(rows) != manifest['counts'][name]: raise ValueError('Rendered row count differs')
        payloads[name] = ''.join(rows).encode('utf-8')
    if split_prompts['train'] & split_prompts['dev']: raise ValueError('Training/development visible input overlap')
    for name, payload in payloads.items():
        if sha(payload) != manifest['expected_jsonl_sha256'][name]: raise ValueError('Reconstructed JSONL hash differs: ' + name)
    return payloads


def rebuild(project=PROJECT):
    """Return exact JSONL bytes after all immutable-source/case checks pass."""
    path = project_file(project, PUBLIC_MANIFEST); raw = path.read_bytes()
    if sha(raw) != PUBLIC_MANIFEST_SHA256: raise ValueError('Public frozen dataset manifest differs')
    manifest = json.loads(raw, object_pairs_hook=_strict_pairs)
    if manifest.get('schema') != 'research4.sp-lens-game.frozen-yaml-export.v1': raise ValueError('Wrong public manifest schema')
    contract = renderer_contract(project, manifest['renderer'])
    if contract != manifest['prompt_contract']: raise ValueError('Public and renderer prompt constants differ')
    catalog = bound_yaml(project, manifest['catalog'])
    if [item['id'] for item in catalog['families']] != manifest['family_order']: raise ValueError('Catalog family order differs')
    if [(item['id'], item['train'], item['dev']) for item in catalog['families']] != [
        (item['id'], item['counts']['train'], item['counts']['dev']) for item in manifest['families']]:
        raise ValueError('Catalog split counts differ')
    root = project_file(project, manifest['catalog']['path']).parent
    expected_names = {Path(item['path']).name for item in manifest['families']} | {root.joinpath('catalog.yaml').name}
    if {path.name for path in root.glob('*.yaml')} != expected_names: raise ValueError('Frozen YAML file inventory differs')
    documents = [bound_yaml(project, item) for item in manifest['families']]
    index = case_index(documents, manifest)
    payloads = render_jsonl(index, manifest, contract)
    receipt = {'schema': 'research4.sp-lens-game.yaml-export-receipt.v1', 'status': 'PASS_EXACT_REPRODUCTION',
               'public_manifest_path': PUBLIC_MANIFEST, 'public_manifest_sha256': sha(raw),
               'exporter_sha256': sha(Path(__file__).read_bytes()), 'pyyaml_version': yaml.__version__,
               'counts': manifest['counts'], 'jsonl_sha256': {name: sha(data) for name, data in payloads.items()},
               'renderer_sha256': manifest['renderer']['sha256'], 'yaml_files_verified': len(documents) + 1,
               'historical_checkout_required': False, 'simulator_executed': False, 'model_loaded': False,
               'heldout_read': False, 'private_jsonl_read_by_exporter': False,
               'scope': 'Exact YAML/constants serialization only; does not rerun simulator, tokenizer, training, inference or evaluation.'}
    return payloads, receipt


def export(output, project=PROJECT):
    output = Path(output).resolve(); project = Path(project).resolve()
    if not output.is_relative_to(project / '.local') or any('onedrive' in part.casefold() for part in output.parts):
        raise ValueError('Generated JSONL must stay in this project .local storage outside OneDrive')
    if output.exists(): raise FileExistsError('Preserve prior output directory')
    payloads, receipt = rebuild(project)
    output.mkdir(parents=True, exist_ok=False)
    for split, data in payloads.items():
        (output / split).mkdir()
        with (output / split / (split + '.jsonl')).open('xb') as file: file.write(data)
    with (output / 'EXPORT_RECEIPT.json').open('x', encoding='utf-8', newline='\n') as file:
        json.dump(receipt, file, ensure_ascii=False, indent=2); file.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = rebuild()[1] if args.verify_only else export(args.output)
    print(json.dumps(result, indent=2))
