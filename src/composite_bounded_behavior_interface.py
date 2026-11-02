"""Canonical list-native interface for bounded V4 records; no runtime authority.

The frozen catalog/list diagnostic retains its original interface. This module
defines the complete new interface shared by data, fitting and native replay.
"""
from copy import deepcopy

from .composite_generalized_interface import (
    SYSTEM, argument_schemas as original_argument_schemas, tools as original_tools)
from .composite_interface import action_text as render
from .native27_tools import parse as parse_native

PROFILE = 'bounded-fictional-record-list-native-v4'


def argument_schemas():
    return {('list' if name == 'catalog' else name): deepcopy(value)
            for name, value in original_argument_schemas().items()}


def tools():
    values = deepcopy(original_tools())
    for value in values:
        function = value['function']
        if function['name'] == 'catalog':
            function['name'] = 'list'
        function['description'] = function['description'].replace('catalog', 'list')
    return values


def input_ids(tokenizer, text):
    ids = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
        tools=tools(), tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def action_text(action):
    return render(action, argument_schemas=argument_schemas())


def parse(text):
    return parse_native(text, maximum_calls=1, argument_schemas=argument_schemas())
