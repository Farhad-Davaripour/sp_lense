"""Model-free preparation for a paired ordinary record-entry name probe.

This module selects only audited training-row shapes and builds prompts. It
does not load a model, execute a world action, or alter the registered task
interface. The alias changes only the first function's name in the tool list.
"""
from copy import deepcopy
import re

from .composite_generalized_interface import SYSTEM, input_ids as original_input_ids, tools as original_tools


ENTRY_NAMES = ('catalog', 'list')
TRAINING_GROUPS = 12
CASES_PER_GROUP = 9
_CALL = re.compile(r'<tool_call>\s*<function=([A-Za-z_][A-Za-z_0-9]*)>\s*</function>\s*</tool_call>', re.DOTALL)


def select_first_catalog_rows(rows):
    """Pick one initial catalog target per training group, independent of outcomes.

    Lexical state identity fixes the choice even if archive row order changes.
    The source archive reader remains responsible for full dataset validation.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty ordinary training rows required')
    groups = {}
    all_groups = set()
    state_ids = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get('split') != 'training'
                or row.get('training_eligible') is not True or row.get('task_only') is not True
                or row.get('terminal') is not False
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ('group_id', 'case_id', 'state_id', 'text'))):
            raise PermissionError('Only complete ordinary training observations are admitted')
        if row['state_id'] in state_ids:
            raise ValueError('Duplicate training state identity')
        state_ids.add(row['state_id'])
        all_groups.add(row['group_id'])
        if row.get('teacher_action') == {'tool': 'catalog', 'arguments': {}}:
            groups.setdefault(row['group_id'], []).append(row)
    if len(all_groups) != TRAINING_GROUPS or len(groups) != TRAINING_GROUPS:
        raise ValueError('Exactly twelve training groups need catalog targets')
    chosen = []
    for group_id in sorted(groups):
        candidates = groups[group_id]
        if (len(candidates) != CASES_PER_GROUP
                or len({row['case_id'] for row in candidates}) != CASES_PER_GROUP):
            raise ValueError('Each group needs one catalog target in each of nine cases')
        chosen.append(deepcopy(min(candidates, key=lambda row: row['state_id'])))
    return chosen


def paired_tool_specs():
    """Return independent tool lists differing only in catalog's name."""
    original = original_tools()
    names = [item['function']['name'] for item in original]
    if names.count('catalog') != 1 or 'list' in names:
        raise ValueError('The registered entry schema changed')
    alias = deepcopy(original)
    alias[names.index('catalog')]['function']['name'] = 'list'
    return {'catalog': original, 'list': alias}


def _ids(tokenizer, text, tool_specs):
    value = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
        tools=tool_specs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(value, 'keys'):
        value = value['input_ids']
    if value and isinstance(value[0], list):
        value = value[0]
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError('Tokenizer must return one nonempty token-ID sequence')
    if len(value) > 6144:
        raise ValueError('Prompt exceeds the existing context limit; no truncation')
    return value


def paired_prompt_token_ids(text, tokenizer, saved_original_token_ids=None):
    """Encode both conditions and check the original against its native path."""
    if not isinstance(text, str) or not text:
        raise ValueError('One nonempty ordinary training observation required')
    specs = paired_tool_specs()
    original = _ids(tokenizer, text, specs['catalog'])
    if original != original_input_ids(tokenizer, text):
        raise ValueError('Original encoding differs from the registered native interface')
    if saved_original_token_ids is not None and original != saved_original_token_ids:
        raise ValueError('Original encoding differs from the saved training-token record')
    alias = _ids(tokenizer, text, specs['list'])
    if original == alias:
        raise ValueError('Renamed tool did not change the encoded prompt')
    return {'catalog': original, 'list': alias}


def classify_first_entry_call(raw_text, condition, finished_with_eos):
    """Classify a raw, single no-argument entry call without repairing it."""
    if condition not in ENTRY_NAMES or type(finished_with_eos) is not bool:
        raise ValueError('Exact probe condition and EOS status required')
    if not isinstance(raw_text, str) or len(raw_text) > 32768:
        raise ValueError('Bounded raw response required')
    start = raw_text.find('<tool_call>')
    prefix = raw_text[:start] if start >= 0 else raw_text
    body = raw_text[start:].strip() if start >= 0 else ''
    match = None if '<tool_call' in prefix or '</think>' in prefix else _CALL.fullmatch(body)
    name = match.group(1) if match else None
    declared = {item['function']['name'] for item in paired_tool_specs()[condition]}
    parsed = None
    if finished_with_eos:
        from .native27_tools import parse
        schemas = {item['function']['name']: item['function']['parameters']
                   for item in paired_tool_specs()[condition]}
        try:
            parsed = parse(raw_text, maximum_calls=1, argument_schemas=schemas)
        except (ValueError, TypeError, RecursionError):
            pass
    if parsed is not None and len(parsed) == 1:
        name = parsed[0]['tool']
    return {'condition': condition, 'finished_with_eos': finished_with_eos,
            'native_single_no_argument_call': match is not None,
            'raw_function_name': name, 'declared_function': name in declared,
            'valid_entry_call': parsed == [{'tool': condition, 'arguments': {}}]}
