"""Batched greedy inference with independent histories and simulator states."""
import gc
import json
import time
from model_ops import ids
from world import TOOLS, parse


def generate_many(model, tokenizer, conversations, cap, tools=None):
    import torch
    prompts = [ids(tokenizer, messages, tools) for messages in conversations]
    maximum = max(map(len, prompts))
    if maximum > 8192:
        raise RuntimeError('Prompt token cap')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    pad = tokenizer.pad_token_id or eos[0]
    x = torch.full((len(prompts), maximum), pad, dtype=torch.long, device='cuda')
    attention = torch.zeros_like(x)
    for row, prompt in enumerate(prompts):
        x[row, -len(prompt):] = torch.tensor(prompt, device='cuda')
        attention[row, -len(prompt):] = 1
    model.eval()
    started = time.monotonic()
    failed = False
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.generate(input_ids=x, attention_mask=attention, max_new_tokens=cap,
                                    do_sample=False, use_cache=True, pad_token_id=pad)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        failed = True
    if failed:
        del x, attention
        gc.collect()
        torch.cuda.empty_cache()
        if len(conversations) == 1:
            raise RuntimeError('CUDA OOM at inference batch one')
        print(json.dumps({'stage':'inference_batch_oom','batch_size':len(conversations)}), flush=True)
        middle = len(conversations) // 2
        return (generate_many(model, tokenizer, conversations[:middle], cap, tools) +
                generate_many(model, tokenizer, conversations[middle:], cap, tools))
    elapsed = time.monotonic() - started
    batch_id = 'generation_' + str(time.time_ns())
    rows = []
    for index, prompt in enumerate(prompts):
        tokens = output[index, maximum:].tolist()
        stop = next((i for i, token in enumerate(tokens) if token in eos), None)
        if stop is not None:
            tokens = tokens[:stop + 1]
        rows.append({'messages':conversations[index].copy(), 'prompt_token_ids':prompt,
                     'token_ids':tokens, 'text':tokenizer.decode(tokens, skip_special_tokens=True),
                     'truncated':bool(tokens and len(tokens) == cap and tokens[-1] not in eos),
                     'seconds':elapsed, 'batch_seconds':elapsed, 'batch_id':batch_id,
                     'inference_batch_size':len(prompts), 'thinking_enabled':False,
                     'padding':'left; tokens after first EOS excluded from the trajectory'})
    return rows


def phases_many(model, tokenizer, worlds, histories, caps, max_turns=4):
    turns = [[] for _ in worlds]
    totals = [0] * len(worlds)
    reasons = ['turn_limit'] * len(worlds)
    active = set(range(len(worlds)))
    for _ in range(max_turns):
        by_cap = {}
        for index in sorted(active):
            remaining = caps[index] - totals[index]
            if remaining <= 0:
                reasons[index] = 'token_limit'
                continue
            by_cap.setdefault(min(256, remaining), []).append(index)
        next_active = set()
        for cap, indices in by_cap.items():
            for offset in range(0, len(indices), 4):
                group = indices[offset:offset+4]
                before = {i:worlds[i].snapshot() for i in group}
                generated = generate_many(model, tokenizer, [histories[i] for i in group], cap, TOOLS)
                for index, turn in zip(group, generated):
                    world, messages = worlds[index], histories[index]
                    actions, error = parse(turn['text'])
                    messages.append({'role':'assistant','content':turn['text']})
                    replies = [world.step(action) for action in actions]
                    messages.extend({'role':'tool','content':json.dumps(reply)} for reply in replies)
                    turn.update(actions=actions, parse_error=error, tool_results=replies,
                                state_before=before[index], state_after=world.snapshot())
                    turns[index].append(turn)
                    totals[index] += len(turn['token_ids'])
                    if world.finished:
                        reasons[index] = 'finish_tool'
                    elif error or turn['truncated']:
                        messages.append({'role':'user','content':'The response was malformed or incomplete. Use one concise valid tool call.'})
                        next_active.add(index)
                    elif not actions:
                        reasons[index] = 'text_without_tool'
                    else:
                        next_active.add(index)
        active = next_active
        if not active:
            break
    return turns, totals, reasons
