"""One actual Qwen block, three correctly located intervention boundaries."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as tf
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

NAMES = ['input', 'middle', 'output', 'hidden']


def array(x):
    p, _ = torch.autograd.forward_ad.unpack_dual(x)
    return p.detach().float().cpu().numpy().copy()


def tensor_sha(x):
    p, _ = torch.autograd.forward_ad.unpack_dual(x)
    p = p.detach().cpu().contiguous()
    return hashlib.sha256(p.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class Rig:
    def __init__(self, checkpoint, check):
        self.check = check
        self.calls = 0
        self.method = None
        self.compatibility_errors = []
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16, attn_implementation='sdpa')
        self.model.eval().requires_grad_(False)
        self.text = self.model.model.language_model
        self.layers = self.text.layers
        self.layer = self.layers[11]
        assert len(self.layers) == 24 and self.layer.block_type == 'full_attention'
        assert all(p.device.type == 'cpu' and not p.requires_grad for p in self.model.parameters())

    def precision(self, dtype):
        self.check()
        self.model.to(dtype=dtype).eval()
        self.dtype = dtype

    def execute(self, ids, site=None, direction=None, beta=None, graph=False):
        self.check()
        self.calls += 1
        assert self.calls <= 300
        handles, records = [], {}
        direction = direction.to(self.dtype) if direction is not None else None

        def edit(x):
            return torch.cat([x[:, :-1], x[:, -1:] + beta.to(self.dtype) * direction.reshape(1, 1, -1)], dim=1)

        if site == 'input':
            def inject_input(module, args):
                return (edit(args[0]),) + args[1:]
            handles.append(self.layer.register_forward_pre_hook(inject_input))
        elif site == 'middle':
            def inject_middle(module, args, output):
                # Change the attention branch BEFORE the decoder's residual sum.
                # Editing only the post-attention RMSNorm input misses the skip.
                if isinstance(output, tuple):
                    return (edit(output[0]),) + output[1:]
                return edit(output)
            handles.append(self.layer.self_attn.register_forward_hook(inject_middle))
        elif site == 'output':
            handles.append(self.layer.register_forward_hook(lambda module, args, output: edit(output)))
        elif site is not None:
            raise ValueError(site)

        def capture(name):
            def hook(module, args):
                records[name] = args[0]
            return hook

        handles.append(self.layer.input_layernorm.register_forward_pre_hook(capture('input')))
        handles.append(self.layer.post_attention_layernorm.register_forward_pre_hook(capture('middle')))
        handles.append(self.layer.register_forward_hook(lambda module, args, output: records.__setitem__('output', output)))
        try:
            with torch.set_grad_enabled(graph), sdpa_kernel(SDPBackend.MATH):
                answer = self.text(input_ids=torch.tensor([ids], dtype=torch.long),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                    use_cache=False, return_dict=True)
                points = tuple(records[name][0, -1] for name in NAMES[:3]) + (answer.last_hidden_state[0, -1],)
            self.last_prefix_hashes = {name: tensor_sha(records[name][:, :-1]) for name in NAMES[:3]}
            self.last_full = {name: array(records[name]) for name in NAMES[:3]}
            return points
        finally:
            for handle in reversed(handles):
                handle.remove()

    def value(self, ids, site=None, direction=None, alpha=0.):
        points = self.execute(ids, site, direction, torch.tensor(alpha))
        with torch.no_grad():
            logits = self.model.lm_head(points[-1])
        return {'points': {name: array(p) for name, p in zip(NAMES, points)},
                'logits': array(logits), 'prefix_hashes': dict(self.last_prefix_hashes),
                'full': dict(self.last_full)}

    def derivative(self, ids, site, direction):
        method = self.method
        if method in [None, 'forward_AD']:
            try:
                with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                    beta = torch.autograd.forward_ad.make_dual(torch.tensor(0.), torch.tensor(1.))
                    points = self.execute(ids, site, direction, beta)
                    pairs = [torch.autograd.forward_ad.unpack_dual(x) for x in points]
                    primals = tuple(p.detach().clone() for p, t in pairs)
                    tangents = tuple(torch.zeros_like(p) if t is None else t.detach().clone() for p, t in pairs)
                    if pairs[-1][1] is None:
                        raise RuntimeError('Missing forward tangent')
                self.method = 'forward_AD'
            except (NotImplementedError, RuntimeError) as error:
                message = str(error).lower()
                if method is not None or 'memory' in message or not any(s in message for s in ['forward', 'jvp', 'not implemented', 'derivative', 'tangent']):
                    raise
                self.compatibility_errors.append(type(error).__name__ + ': ' + str(error))
                primals = tangents = None
        else:
            primals = tangents = None
        if primals is None:
            fn = lambda beta: self.execute(ids, site, direction, beta, graph=True)
            with torch.enable_grad():
                primals, tangents = torch.autograd.functional.jvp(fn, torch.tensor(0.), torch.tensor(1.))
            self.method = 'reverse_over_reverse_JVP'
        with torch.no_grad():
            logits = self.model.lm_head(primals[-1])
            # The head's derivative has no bias contribution.
            q = tf.linear(tangents[-1], self.model.lm_head.weight)
        return {'primal': {name: array(p) for name, p in zip(NAMES, primals)},
                'tangent': {name: array(t) for name, t in zip(NAMES, tangents)},
                'logits': array(logits), 'q': array(q),
                'prefix_hashes': dict(self.last_prefix_hashes)}
