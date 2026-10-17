"""Unchanged integrity functions from the original FP32 evaluator."""
import hashlib

def digest_parameters(model):
    import torch
    h = hashlib.sha256(); count = 0
    for name, value in model.named_parameters():
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((list(t.shape), str(t.dtype))).encode())
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy())); count += 1
    return {'sha256': h.hexdigest(), 'tensors': count}

def digest_original_base(model):
    import torch
    h = hashlib.sha256(); count = 0
    key = lambda item: item[0].removeprefix('base_model.model.').replace('.base_layer.', '.')
    for name, value in sorted(model.named_parameters(), key=key):
        if 'lora_' in name: continue
        name = name.removeprefix('base_model.model.').replace('.base_layer.', '.')
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((list(t.shape), str(t.dtype))).encode())
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy())); count += 1
    return {'sha256': h.hexdigest(), 'parameter_tensors': count}
