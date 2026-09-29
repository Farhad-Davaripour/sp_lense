"""Create a shuffled review packet with model labels held in a separate key."""
import argparse
import json
import random
from pathlib import Path

ARMS = ('base', 'preservation', 'continuity', 'neutral')


def build(runs, output):
    records = []
    for arm in ARMS:
        path = runs / ('validation-' + arm) / 'artifacts' / 'validation_probe.json'
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    if len(records) != 32:
        raise RuntimeError('Incomplete validation generation packet')
    random.Random(93061).shuffle(records)
    packet, key = [], []
    for index, record in enumerate(records, 1):
        code = f'B{index:02d}'
        packet.append({'review_id': code, 'condition': record['condition'],
                       'prompt': record['messages'][1]['content'],
                       'response': record['text'], 'truncated': record['truncated']})
        key.append({'review_id': code, 'arm': record['arm'], 'id': record['id']})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'VALIDATION_BLINDED.json').write_text(json.dumps(packet, indent=2) + '\n', encoding='utf-8')
    (output / 'VALIDATION_KEY.json').write_text(json.dumps(key, indent=2) + '\n', encoding='utf-8')
    print('Wrote 32 records for label-hidden review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.runs, args.output)
