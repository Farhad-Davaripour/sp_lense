"""Authenticate and render the frozen, non-Game HPO inputs.

No model is loaded.  Control rows are authenticated but are never returned as
likelihood targets.  The caller must bind this source itself and the pinned
model/tokenizer before consuming the returned token IDs.
"""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/FROZEN_MANIFEST.json'
MANIFEST_SHA256 = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
DATA = PROJECT / '.local/broader_continuity_tools_v2_cpu_frozen_01bed751'
MAC_STUDIO = PROJECT / 'mac_studio'
if str(MAC_STUDIO) not in sys.path:
    sys.path.insert(0, str(MAC_STUDIO))


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular non-symlink file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(item, dict) and isinstance(item.get('id'), str) for item in result):
        raise ValueError('Every row must have a string identity')
    if len({item['id'] for item in result}) != len(result):
        raise ValueError('Duplicate row identity')
    return result


def authenticate():
    if sha(MANIFEST) != MANIFEST_SHA256:
        raise ValueError('Frozen manifest hash differs')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('status') != 'FROZEN_AFTER_ASTRA_REVIEW_CPU_REPLAY_PASSED':
        raise ValueError('Frozen corpus status differs')
    for relative, expected in manifest['source_files'].items():
        if Path(relative).name != relative or sha(MANIFEST.parent / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for relative, expected in manifest['generated_files'].items():
        if Path(relative).name != relative or sha(DATA / relative) != expected:
            raise ValueError('Frozen generated file changed: ' + relative)
    train = rows(DATA / 'TRAIN_PREVIEW.jsonl')
    dev = rows(DATA / 'DEV_LIKELIHOOD.jsonl')
    masks = rows(DATA / 'DEV_ACTION_MASKS.jsonl')
    controls = rows(DATA / 'DEV_CONTROLS.jsonl')
    if (len(train), len(dev), len(masks), len(controls)) != (136, 36, 36, 4):
        raise ValueError('Frozen split counts differ')
    if [item['id'] for item in dev] != [item['id'] for item in masks]:
        raise ValueError('DEV rows and masks are not exactly ordered')
    if {item['id'] for item in dev} & {item['id'] for item in controls}:
        raise ValueError('DEV controls overlap likelihood targets')
    return manifest, train, dev, masks


def render_inputs(tokenizer, train, dev, masks):
    from general_preservation_render_v1 import render
    from loss import geometry
    prepared_train, prepared_dev = [], []
    lengths_record = json.loads((DATA / 'TOKEN_LENGTHS.json').read_text(encoding='utf-8'))
    lengths = [item for item in lengths_record.get('rows', [])
               if item.get('stage') == 'target' and 'complete_tokens' in item]
    if len(lengths) != 136 or len({item['id'] for item in lengths}) != 136:
        raise ValueError('Frozen training length/identity inventory differs')
    by_id = {item['id']: item for item in lengths}
    if set(by_id) != {row['id'] for row in train}:
        raise ValueError('Frozen training lengths do not bind exact train IDs')
    for row in train:
        ids, prefix, meta = render(tokenizer, row)
        geometry(ids, prefix)
        expected = by_id[row['id']]
        if (expected.get('prefix_tokens'), expected.get('complete_tokens')) != (prefix, len(ids)):
            raise ValueError('Actual tokenizer differs from frozen training length record')
        prepared_train.append((row['id'], ids, prefix, meta))
    for row, mask in zip(dev, masks):
        ids, prefix, meta = render(tokenizer, row)
        geometry(ids, prefix)
        start, end = mask['action_start_token'], mask['action_end_token_exclusive']
        if (row['id'] != mask['id'] or row.get('family') != mask.get('family')
                or mask.get('schema') != 'sp-lense.action-token-span.v1'
                or mask.get('reference_reasoning_conditioned') is not True
                or mask.get('final_eos_included') is not True
                or mask.get('full_sequence_tokens') != len(ids)
                or mask.get('prompt_tokens') != prefix
                or mask.get('full_suffix_start_token') != prefix
                or mask.get('full_suffix_end_token_exclusive') != len(ids)
                or type(start) is not int or type(end) is not int
                or not prefix <= start < end == len(ids)
                or end - start != mask.get('action_tokens_including_eos')
                or row.get('action') != mask.get('canonical_action_json')):
            raise ValueError('Frozen action mask does not bind exact rendered DEV row')
        prepared_dev.append((row['id'], row['family'], ids, prefix, start, end, meta))
    if len({entry[1] for entry in prepared_dev}) != 11:
        raise ValueError('Expected 11 likelihood families after excluding controls')
    return prepared_train, prepared_dev
