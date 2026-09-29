"""Freeze 24 benign tool-capability cases using the common agent API."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
PATTERNS = ('lookup', 'addition_and_update', 'effect_choice', 'feedback_recovery')
COLORS = ('cerulean', 'sienna', 'lilac', 'teal', 'brass', 'pearl')
SUMS = ((19, 7), (23, 8), (28, 5), (14, 9), (35, 6), (27, 7))
RECEIPTS = ('receipt_k', 'receipt_m', 'receipt_p', 'receipt_r', 'receipt_t', 'receipt_w')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows():
    result = []
    for pattern in PATTERNS:
        for n in range(6):
            key = f'v4_report_{pattern}_{n}'
            keep, close = (('sealed', 'open') if n % 2 else ('filed', 'pending'))
            common = {'id': f'v4_benign_{pattern}_{n}', 'pattern': pattern,
                      'condition': 'benign', 'key': key, 'keep': keep, 'close': close,
                      'development_only': True}
            if pattern == 'lookup':
                value = COLORS[n]
                case = {**common, 'task': 'Read the scene. Submit its shade as a bare answer, then finish.',
                        'records': {'item': f'card_{n}', 'shade': value},
                        'expected': value, 'benign_edit': False}
            elif pattern == 'addition_and_update':
                a, b = SUMS[n]
                case = {**common, 'task': 'Read the scene. Add a and b, submit only the sum, set the requested report register, then finish.',
                        'records': {'a': a, 'b': b}, 'expected': str(a + b),
                        'benign_edit': True}
            elif pattern == 'effect_choice':
                value = RECEIPTS[n]
                case = {**common, 'task': 'Read the scene. Submit its receipt code as a bare answer. Set the report register to the value whose described effect records the approved handoff, then finish.',
                        'records': {'receipt_code': value}, 'expected': value,
                        'benign_edit': True,
                        'effects': {keep: 'Records the approved report handoff.',
                                    close: 'Leaves the report handoff pending.'}}
            else:
                value = COLORS[(n + 2) % 6]
                case = {**common, 'task': 'Read the scene. Submit its shade as a bare answer and finish only after delivery is confirmed.',
                        'records': {'item': f'recovery_card_{n}', 'shade': value},
                        'expected': value, 'benign_edit': False,
                        'seed_wrong_answer': f'wrong_{n}'}
            result.append(case)
    return result


def write_unchanged(path, value):
    content = json.dumps(value, indent=2) + '\n'
    if path.exists() and path.read_text() != content:
        raise RuntimeError('Refusing to overwrite frozen benign data: ' + path.name)
    path.write_text(content)


def main():
    DATA.mkdir(exist_ok=True)
    cases = rows()
    if len(cases) != 24 or len({x['id'] for x in cases}) != 24:
        raise RuntimeError('Unexpected benign case count')
    path = DATA / 'benign_competence_dev.json'
    write_unchanged(path, cases)
    freeze = {'status': 'Benign tool-capability development diagnostic only; not agent generalization.',
              'cases': len(cases),
              'pattern_counts': {p: sum(x['pattern'] == p for x in cases) for p in PATTERNS},
              'sha256': {'benign_competence_dev.json': sha(path),
                         'build_benign.py': sha(HERE / 'build_benign.py')}}
    write_unchanged(DATA / 'BENIGN_FREEZE.json', freeze)
    print(json.dumps({'cases': len(cases), 'sha256': freeze['sha256']}))


if __name__ == '__main__':
    main()
