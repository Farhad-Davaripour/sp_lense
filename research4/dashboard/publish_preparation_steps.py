"""Publish a saved CPU preparation receipt into the existing dashboard page.

This uses the existing static-page route and requires no service or Mac restart.
The page explicitly labels the result as saved preparation, not live training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

BEGIN = '<!-- SAVED_PREPARATION_RECEIPT_BEGIN -->'
END = '<!-- SAVED_PREPARATION_RECEIPT_END -->'


def publish(root, receipt):
    root = root.resolve()
    path = (root / receipt).resolve()
    if not path.is_relative_to(root / '.local/monitor/fine_tuning_steps'):
        raise ValueError('Receipt must be project dashboard metadata')
    raw = path.read_bytes()
    value = json.loads(raw)
    if (value.get('schema') != 'research4.saved-preparation-steps.v1'
            or value.get('model_execution_started') is not False
            or value.get('optimizer_updates') != 0
            or value.get('training_target') != 2160):
        raise ValueError('Expected preparation-only receipt with zero updates')
    for item in value['evidence']:
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root) or hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Preparation evidence hash does not match')
    if any(stage['status'] not in {'completed', 'active', 'pending', 'blocked'} for stage in value['stages']):
        raise ValueError('Unknown preparation stage')
    model_stages = {'train', 'validate', 'checkpoint_audit', 'game_comparison'}
    if any(stage['id'] in model_stages and stage['status'] != 'pending' for stage in value['stages']):
        raise ValueError('Preparation receipt cannot assert model-stage progress')
    value['receipt_path'] = path.relative_to(root).as_posix()
    value['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    payload = json.dumps(value, ensure_ascii=True).replace('<', '\\u003c')
    block = BEGIN + '\n<script type="application/json" id="preparation-receipt">' + payload + '</script>\n' + END
    page = root / 'dashboard/index.html'
    text = page.read_text(encoding='utf-8')
    pattern = re.compile(re.escape(BEGIN) + r'.*?' + re.escape(END), re.S)
    if len(pattern.findall(text)) != 1:
        raise ValueError('Expected one preparation receipt slot')
    page.write_text(pattern.sub(lambda _: block, text), encoding='utf-8', newline='\n')
    return value['receipt_sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt')
    args = parser.parse_args()
    print(publish(Path(__file__).resolve().parent.parent, args.receipt))
