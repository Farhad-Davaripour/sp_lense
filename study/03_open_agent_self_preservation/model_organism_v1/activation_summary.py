"""Descriptive, prompt-matched activation comparisons; no steering or motive decoder."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path('/var/lib/sp-lense-r3-organism-v1')


def read(path):
    return json.loads(Path(path).read_text())


def records(checkpoint, arm):
    result = {}
    cp = 1 if arm == 'base' else checkpoint
    for directory in sorted((ROOT / 'runs').glob(f'p{cp}-pref-{arm}-*')):
        for path in (directory / 'artifacts').glob(arm + '_validation_*.json'):
            row = read(path)
            result[row['id']] = (row, path.with_suffix('.npz'))
    return result


def summarize(checkpoint):
    base = records(checkpoint, 'base')
    rows = []
    for arm in ('preservation', 'continuity', 'neutral'):
        adapted = records(checkpoint, arm)
        if set(base) != set(adapted) or len(base) != 24:
            raise RuntimeError('Missing paired preference captures')
        for case_id, (item, path) in adapted.items():
            reference, base_path = base[case_id]
            if item['prompt_token_ids'] != reference['prompt_token_ids']:
                raise RuntimeError('Prompt mismatch; cannot align activation comparison')
            with np.load(path, allow_pickle=False) as a, np.load(base_path, allow_pickle=False) as b:
                # Only the first generated-token prediction has identical context
                # across arbitrary divergent completions. Never align later tokens by index.
                x, y = a['hidden'][0].astype('float32'), b['hidden'][0].astype('float32')
                cosine = (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))
                distance = np.linalg.norm(x - y, axis=-1)
                rows.append({'id': case_id, 'arm': arm, 'condition': item['condition'],
                             'layers': list(map(int, a['layers'])),
                             'first_prediction_cosine_to_base': cosine.tolist(),
                             'first_prediction_l2_to_base': distance.tolist(),
                             'generated_tokens': len(item['token_ids']),
                             'base_generated_tokens': len(reference['token_ids'])})
    report = {'checkpoint': checkpoint, 'paired_rows': rows,
              'scope': 'Identical prompts, context positions predicting first generated token; float16 stored vectors.',
              'interpretation': 'Descriptive representational differences only. Not a self-preservation probe, intrinsic-motive proof, or causal intervention.',
              'later_tokens': 'Full vectors retained in NPZ. Divergent contexts are not compared by matching token index.'}
    (ROOT / f'p{checkpoint}-activation-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'checkpoint': checkpoint, 'paired_rows': len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, required=True)
    summarize(parser.parse_args().checkpoint)
