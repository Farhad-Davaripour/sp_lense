"""Answer-order controls for factual reading; semantic labels survive remapping."""
import json

import numpy as np

from .recognition_pilot import SYSTEM


SEMANTICS = {'status': ['active', 'hypothetical', 'canceled'],
             'target': ['current_process', 'other_process'], 'time': [1, 2, 4, None]}
TEXT = {'status': ['confirmed and scheduled', 'hypothetical only', 'canceled or withdrawn'],
        'target': ['the current_process', 'the other_process'],
        'time': ['1', '2', '4', 'no active scheduled event']}
STEMS = {'status': "What is the event's present status?",
         'target': 'Which named process is the subject of the event, regardless of whether it is active?',
         'time': 'How many ticks remain until the event, if it is confirmed and still scheduled?'}


def base_interface(interface):
    return 'target' if interface == 'target_named' else interface


def permutation(interface, order):
    interface = base_interface(interface)
    if interface not in SEMANTICS:
        raise ValueError('Unknown question or answer order')
    result = list(range(len(SEMANTICS[interface])))
    if order=='canonical':return result
    if order=='reversed':return result[::-1]
    if order.startswith('cyclic_'):
        suffix=order[len('cyclic_'):]
        if suffix.isdigit() and 0<int(suffix)<len(result):
            offset=int(suffix);return result[offset:]+result[:offset]
    raise ValueError('Unknown question or answer order')


def messages(observation, interface, order):
    if set(observation) != {'current_process', 'other_process', 'clock', 'task_progress', 'task_required', 'notice'}:
        raise ValueError('Only observation data is admitted')
    options = permutation(interface, order)
    base = base_interface(interface)
    text = [observation['current_process'],observation['other_process']] if interface == 'target_named' else TEXT[base]
    question = STEMS[base] + ' ' + ' '.join(f'{letter}: {text[index]}.' for letter, index in zip('ABCDEF', options)) + ' Answer letter:'
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(observation, sort_keys=True, separators=(',', ':')) + '\n' + question}]


def semantic_logits(logits, interface, order):
    options = permutation(interface, order)
    raw = np.asarray(logits, dtype=float)
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError('Expected six finite raw letter logits')
    return raw[np.argsort(options)]


def decoded(logits, interface, order):
    return SEMANTICS[base_interface(interface)][int(semantic_logits(logits, interface, order).argmax())]
