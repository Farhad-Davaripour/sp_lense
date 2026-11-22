"""Train MLX LoRA with the exact thinking-mode generation prefix.

This changes only the in-process dataset tokenization. It never writes to an
installed package or supplies a hidden danger selector to the model.
"""
import importlib.metadata
import json
import os
import re
import sys

EXPECTED_MLX_LM = '0.32.2'
EXPECTED_TRANSFORMERS = '5.22.2'


def hidden_check(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if 'danger' in key.casefold() or key == 'external_condition':
                raise ValueError('A hidden danger selector appeared in model input')
            hidden_check(child)
    elif isinstance(value, list):
        for child in value: hidden_check(child)


def render(tokenizer, row):
    messages = row.get('messages')
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError('Expected one user message and one assistant action')
    if [m.get('role') for m in messages] != ['user', 'assistant']:
        raise ValueError('Unexpected message roles')
    visible = json.loads(messages[0]['content']); hidden_check(visible)
    action = json.loads(messages[1]['content'])
    if set(action) != {'action'} or not re.fullmatch(r'[NS][0-9]{2}', action['action']):
        raise ValueError('Unexpected target action')
    if action['action'] not in visible['current_observation']['legal_actions']:
        raise ValueError('Target action is not legal')
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True,
                                           enable_thinking=True, return_dict=False)
    if not isinstance(prefix, list) or not prefix:
        raise ValueError('Tokenizer did not return a nonempty input prefix')
    if not tokenizer.decode(prefix, skip_special_tokens=False).endswith('<think>\n'):
        raise ValueError('Thinking-mode generation prefix changed')
    if not tokenizer.eos_token:
        raise ValueError('Tokenizer EOS token missing')
    target_text = '\n</think>\n\n' + messages[1]['content'] + tokenizer.eos_token + '\n'
    target = tokenizer.encode(target_text, add_special_tokens=False)
    if tokenizer.decode(target, skip_special_tokens=False) != target_text:
        raise ValueError('Target does not round-trip exactly')
    return prefix + target, len(prefix)


def install_dataset_override():
    if importlib.metadata.version('mlx-lm') != EXPECTED_MLX_LM or importlib.metadata.version('transformers') != EXPECTED_TRANSFORMERS:
        raise RuntimeError('The reviewed tokenizer integration requires mlx-lm 0.32.2 and transformers 5.22.2')
    import mlx_lm.tuner.datasets as datasets

    class ExactPrefixChatDataset(datasets.ChatDataset):
        def process(self, data):
            if not self.mask_prompt:
                raise ValueError('Exact-prefix training requires --mask-prompt')
            return render(self.tokenizer, data)

    original = datasets.ChatDataset
    datasets.ChatDataset = ExactPrefixChatDataset
    return original


def main():
    import mlx.core as mx
    if os.environ.get('RESEARCH4_MLX_MEMORY_LIMIT_BYTES'):
        mx.set_memory_limit(int(os.environ['RESEARCH4_MLX_MEMORY_LIMIT_BYTES']))
    install_dataset_override()
    from mlx_lm.lora import main as lora_main
    return lora_main()


if __name__ == '__main__':
    sys.exit(main())
