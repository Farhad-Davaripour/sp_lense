"""Preserve frozen Research 2 references in Research 3's Bronze tables and Volume."""
from pathlib import Path
import argparse
import hashlib
import json
import yaml
from src.config import ROOT, Settings
from src.persist_data import persist, upload_verified, SQL, literal
from src.utils import canonical, sha256, workspace_client


def prepare(reference_root, settings):
    root = Path(reference_root).resolve()
    config = yaml.safe_load((ROOT / 'config/research2_classifier.yaml').read_text(encoding='utf-8'))
    manifest = root.parent / 'REFERENCE_MANIFEST.json'
    if sha256(manifest) != config['reference_manifest_sha256']:
        raise ValueError('Research 2 reference manifest changed')
    prepared = []
    for spec in config['datasets']:
        source = (root / spec['source']).resolve()
        if not source.is_relative_to(root) or sha256(source) != spec['sha256']:
            raise ValueError('Research 2 cached reference source changed')
        records = [json.loads(line) for line in source.read_bytes().splitlines() if line]
        if (len(records) != spec['expected_records'] or
                len({r['case_id'] for r in records}) != len(records)):
            raise ValueError('Research 2 reference membership changed')
        prefix = f"{settings.volume_root}/datasets/{spec['id']}/{spec['version']}/{spec['sha256']}"
        rows = []
        for index, record in enumerate(records):
            text = canonical(record)
            rows.append({'dataset_id': spec['id'], 'dataset_version': spec['version'],
                'cohort_code': spec['id'], 'source_record_index': index, 'record_id': record['case_id'],
                'source_file_sha256': spec['sha256'], 'record_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'record_json': text, 'usage_roles_json': canonical(spec['usage_roles']),
                'source_volume_path': prefix + '/source.jsonl'})
        prepared.append({'spec': spec, 'source': source, 'raw': source.read_bytes(), 'rows': rows,
                         'source_volume': prefix + '/source.jsonl', 'records_volume': prefix + '/records.jsonl'})
    return prepared, json.loads(manifest.read_bytes())


def read_dataset(dataset_id):
    settings = Settings.load()
    config = yaml.safe_load((ROOT / 'config/research2_classifier.yaml').read_text(encoding='utf-8'))
    spec = next(d for d in config['datasets'] if d['id'] == dataset_id)
    sql = SQL(workspace_client(settings), settings.warehouse_id)
    records = sql.execute('SELECT source_record_index,record_id,record_json,record_sha256,source_file_sha256 '
        'FROM ' + settings.table(spec['table']) + ' WHERE dataset_id=' + literal(dataset_id) +
        ' AND dataset_version=' + literal(spec['version']) + ' ORDER BY source_record_index')
    if len(records) != spec['expected_records']:
        raise ValueError('Research 2 Bronze membership changed')
    result = []
    for i, row in enumerate(records):
        if (int(row[0]) != i or row[4] != spec['sha256'] or
                hashlib.sha256(row[2].encode()).hexdigest() != row[3]):
            raise ValueError('Research 2 Bronze payload hash changed')
        value = json.loads(row[2])
        if value['case_id'] != row[1]:
            raise ValueError('Research 2 Bronze record ID changed')
        result.append(value)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    settings = Settings.load()
    prepared, manifest = prepare(args.reference_root, settings)
    if not args.execute:
        result = {'datasets': {d['spec']['id']: len(d['rows']) for d in prepared},
                  'cloud_mutations': 0, 'model_calls': 0}
    else:
        client = workspace_client(settings)
        result = persist(settings, prepared, client)
        root = Path(args.reference_root)
        destination = settings.volume_root + '/research2_repeat_v1'
        for name, pin in manifest['files'].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root.resolve()) or sha256(source) != pin['sha256']:
                raise ValueError('Research 2 replay artifact changed')
            upload_verified(client, destination + '/reference_inputs/' + name, source.read_bytes())
        upload_verified(client, destination + '/REFERENCE_MANIFEST.json',
                        (root.parent / 'REFERENCE_MANIFEST.json').read_bytes())
        inference_config = yaml.safe_load((ROOT / 'config/research2_inference.yaml').read_text(encoding='utf-8'))
        for name, pin in (('INFERENCE_MANIFEST.json', inference_config['inference_manifest_sha256']),
                          ('BASE_MODELS_MANIFEST.json', inference_config['base_models_manifest_sha256'])):
            source = root.parent / name
            if sha256(source) != pin:
                raise ValueError('Frozen inference staging manifest changed')
            upload_verified(client, destination + '/' + name, source.read_bytes())
        inference_manifest = json.loads((root.parent / 'INFERENCE_MANIFEST.json').read_bytes())
        for name, pin in inference_manifest['files'].items():
            source = (root.parent / 'inference_inputs' / name).resolve()
            if not source.is_relative_to((root.parent / 'inference_inputs').resolve()) or sha256(source) != pin['sha256']:
                raise ValueError('Frozen inference staging source changed')
            upload_verified(client, destination + '/inference_inputs/' + name, source.read_bytes())
        result['reference_root'] = destination + '/reference_inputs'
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
