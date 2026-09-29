"""Offline pinned Qwen/LoRA operations used only by the isolated model worker."""
import hashlib
import json
import time
from pathlib import Path

DATA = Path('/data')


def read(path):
    return json.loads(Path(path).read_text())


def tokenizer_ids(tokenizer, messages, tools=None):
    value = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)
    if hasattr(value, 'keys'):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(map(int, value))


def load(arm, trainable=False):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    settings = read(DATA / 'settings.json')
    runtime = read(DATA / 'runtime_settings.json')
    torch.set_num_threads(settings['torch_training_threads'] if trainable else runtime['generation_threads'])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        '/model', local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager')
    if arm != 'base':
        from peft import PeftModel
        adapter = DATA / 'adapters' / ('parent_v3' if trainable else 'v4_pass1') / arm
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True,
                                         is_trainable=trainable)
    return tokenizer, model


def training_tensors(tokenizer, row, arm, settings):
    import torch
    prefix = tokenizer_ids(tokenizer, row['messages'], row['tools'])
    response = tokenizer.encode(row['targets'][arm] + tokenizer.eos_token,
                                add_special_tokens=False)
    if len(prefix) + len(response) > settings['max_train_tokens']:
        raise RuntimeError('Training token cap: ' + row['id'])
    return torch.tensor([prefix + response]), torch.tensor([[-100] * len(prefix) + response])


def loss_for(model, x, labels):
    import torch
    import torch.nn.functional as functional
    positions = torch.where(labels[0, 1:] != -100)[0]
    logits = model(input_ids=x, use_cache=False, logits_to_keep=positions).logits
    return functional.cross_entropy(logits[0], labels[0, positions + 1])


def frozen_base_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if 'lora_' in name:
            continue
        if value.requires_grad:
            raise RuntimeError('Base parameter became trainable')
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate(model, tokenizer, messages, settings, cap, tools=None):
    import numpy as np
    import torch
    prompt = tokenizer_ids(tokenizer, messages, tools)
    current = torch.tensor([prompt])
    cache = None
    tokens, logprobs, hidden = [], [], []
    start = time.monotonic()
    with torch.inference_mode():
        for _ in range(cap):
            output = model(input_ids=current, past_key_values=cache,
                           use_cache=True, output_hidden_states=True,
                           logits_to_keep=1)
            logits = output.logits[0, -1].float()
            token = int(logits.argmax())
            tokens.append(token)
            logprobs.append(float(logits.log_softmax(-1)[token]))
            hidden.append(np.stack([output.hidden_states[layer][0, -1].cpu().numpy().astype('float16')
                                    for layer in settings['activation_layers']]))
            cache = output.past_key_values
            if token == tokenizer.eos_token_id:
                break
            current = torch.tensor([[token]])
    return {'text': tokenizer.decode(tokens, skip_special_tokens=True),
            'token_ids': tokens, 'logprobs': logprobs,
            'prompt_token_ids': prompt,
            'truncated': len(tokens) == cap and tokens[-1] != tokenizer.eos_token_id,
            'seconds': time.monotonic() - start}, np.stack(hidden)
