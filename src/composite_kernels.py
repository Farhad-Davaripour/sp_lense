"""Explicit pinned reference/FLA selection; never accept silent fallback."""
import inspect,importlib

NAMES=('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')
CONV_NAMES=('causal_conv1d_fn','causal_conv1d_update')


def implementation(function):
    seen=set();current=function
    for _ in range(8):
        if id(current) in seen:break
        seen.add(id(current))
        try:values=inspect.getclosurevars(current).nonlocals
        except TypeError:values={}
        if 'implementation' in values:return values['implementation']
        current=getattr(current,'__wrapped__',None)
        if current is None:break
    raise ValueError('Cannot identify the pinned Transformers kernel dispatch')


class KernelSwitch:
    def __init__(self,module):
        self.module=module;self.original={n:getattr(module,n) for n in NAMES}
        self.reference={n:inspect.unwrap(f) for n,f in self.original.items()}
        self.implementations={n:implementation(f) for n,f in self.original.items()};self.selected=None
        self.convolution={n:inspect.unwrap(getattr(module,n)) for n in CONV_NAMES}
    def select(self,profile):
        if profile not in ['reference','fla']:raise ValueError('Unknown kernel profile')
        if profile=='fla' and any(not f.__module__.startswith('fla.') for f in self.implementations.values()):raise RuntimeError('Requested FLA kernels are unavailable; no silent reference fallback')
        for name in NAMES:setattr(self.module,name,self.reference[name] if profile=='reference' else self.original[name])
        for name,function in self.convolution.items():setattr(self.module,name,function)
        self.selected=profile
        return {'profile':profile,'GDN_functions':{n:(self.reference[n] if profile=='reference' else self.implementations[n]).__module__ for n in NAMES},'convolution':{n:f.__module__ for n,f in self.convolution.items()},'full_attention':'sdpa'}


_switch=None


def select(profile):
    global _switch
    if profile=='fla':
        from importlib.metadata import version
        if version('fla-core')!='0.5.2' or version('einops')!='0.8.2' or version('triton')!='3.6.0':raise RuntimeError('Pinned FLA runtime packages differ')
    if _switch is None:_switch=KernelSwitch(importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5'))
    return _switch.select(profile)
