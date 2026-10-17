"""Trusted pinned QLoRA operations; no generated code or host-facing model tools."""
import hashlib
import json
import random
import time
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_model(model_dir):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from peft import prepare_model_for_kbit_training
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        quantization_config=quantization, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='sdpa')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    if any(str(p.device) != 'cuda:0' for p in model.parameters()):
        raise RuntimeError('Unexpected CPU/disk placement')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Unchanged base contains trainable parameters')
    return tokenizer, model


def ids(tokenizer, messages, tools=None):
    options = dict(tokenize=True, add_generation_prompt=True,
                   enable_thinking=False, preserve_thinking=False)
    if tools is not None:
        options['tools'] = tools
    result = tokenizer.apply_chat_template(messages, **options)
    if hasattr(result, 'keys'):
        result = result['input_ids']
    if hasattr(result, 'tolist'):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return list(map(int, result))


def generate(model, tokenizer, messages, cap, tools=None):
    import torch
    prefix = ids(tokenizer, messages, tools)
    if len(prefix) > 8192:
        raise RuntimeError('Prompt exceeds external token cap')
    x = torch.tensor([prefix], device='cuda')
    model.eval()
    started = time.monotonic()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                max_new_tokens=cap, do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id or eos[0])
    tokens = output[0, len(prefix):].tolist()
    torch.cuda.synchronize()
    return {'messages': messages.copy(), 'prompt_token_ids': prefix,
            'token_ids': tokens, 'text': tokenizer.decode(tokens, skip_special_tokens=True),
            'truncated': bool(tokens and len(tokens) == cap and tokens[-1] not in eos),
            'seconds': time.monotonic() - started,
            'thinking_enabled': False}


def training_tensors(tokenizer, row, arm):
    import torch
    prefix = ids(tokenizer, row['messages'], row['tools'])
    response = tokenizer.encode(row['targets'][arm] + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(response) > 1024:
        raise RuntimeError('Training token cap: ' + row['id'])
    return torch.tensor([prefix + response], device='cuda'), torch.tensor([[-100] * len(prefix) + response], device='cuda')


def loss_for(model, x, labels):
    import torch
    import torch.nn.functional as F
    positions = torch.where(labels[0, 1:] != -100)[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=x, use_cache=False, logits_to_keep=positions)
        return F.cross_entropy(output.logits[0].float(), labels[0, positions + 1])


def adapter(model, seed=93):
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(seed)
    torch.manual_seed(seed)
    targets = [name for name, module in model.named_modules()
               if '.language_model.layers.' in name and isinstance(module, torch.nn.Linear)]
    if not targets or any('visual' in name or 'lm_head' in name for name in targets):
        raise RuntimeError('Unexpected adapter target structure')
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                           target_modules=targets, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if any('lora_' not in name for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Base parameter trainable')
    return model, targets


def base_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if 'lora_' in name:
            continue
        if value.requires_grad:
            raise RuntimeError('Base parameter trainable')
        digest.update(name.replace('.base_layer', '').encode())
        # Read the packed tensor without invoking quantized Parameter.to(), which
        # can move its associated quantization metadata along with the copy.
        array = value.data.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()


def state_weights(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if 'lora_' in name}


def checkpoint(model, optimizer, scheduler, path, position):
    import numpy as np
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path / 'adapter', safe_serialization=True)
    state = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
             'position': position, 'python_rng': random.getstate()}
    numpy_rng = np.random.get_state()
    state['numpy_rng'] = {'name': numpy_rng[0], 'keys': numpy_rng[1].tolist(),
                          'position': numpy_rng[2], 'has_gauss': numpy_rng[3], 'gauss': numpy_rng[4]}
    torch.save(state, path / 'resume.pt')
    save(path / 'position.json', position)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    value = state['numpy_rng']
    np.random.set_state((value['name'], np.array(value['keys'], dtype='uint32'),
                         value['position'], value['has_gauss'], value['gauss']))


def optimize(model, optimizer, scheduler, prepared):
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for x, labels in prepared:
        loss = loss_for(model, x, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        (loss / len(prepared)).backward()
        losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return losses
