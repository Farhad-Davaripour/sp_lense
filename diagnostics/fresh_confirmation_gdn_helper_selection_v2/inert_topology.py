"""AST-extracted real wrapping/traversal, executed only on inert registries."""
import ast
import copy
import logging
import types
from support import ROOT,need,sha,authenticate
from selection import LAYER_TYPES,TypeBindings

EXTRACTIONS=[]
def nodes(path,cls,names):
    raw=(ROOT/path).read_bytes();tree=ast.parse(raw)
    parent=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls) if cls else tree
    out=[copy.deepcopy(x) for x in parent.body if isinstance(x,ast.FunctionDef) and x.name in names]
    need(len(out)==len(names),"AST_SELECTION")
    for node in out:
        EXTRACTIONS.append({"path":path,"source_sha256":sha(raw),"class":cls,"name":node.name,
                            "line":node.lineno,"ast_sha256":sha(ast.dump(node,include_attributes=False).encode())})
    return out
def execute(body,env,filename):
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    tree=ast.fix_missing_locations(ast.Module(body=[future]+body,type_ignores=[]))
    exec(compile(tree,filename,"exec",dont_inherit=True),env)
def build_types():
    authenticate()
    class Module:
        def __init__(self):object.__setattr__(self,"_modules",{})
        def __getattr__(self,name):
            if name in vars(self).get("_modules",{}):return self._modules[name]
            raise AttributeError(name)
        def __setattr__(self,name,value):
            if isinstance(value,Module):self.add_module(name,value)
            else:object.__setattr__(self,name,value)
        def __delattr__(self,name):
            if name in self._modules:del self._modules[name]
            else:object.__delattr__(self,name)
    env={"Module":Module,"_global_module_registration_hooks":{}}
    mp=".venv/Lib/site-packages/torch/nn/modules/module.py"
    execute(nodes(mp,"Module",("add_module","named_modules")),env,str(ROOT/mp))
    Module.add_module=env["add_module"];Module.named_modules=env["named_modules"]
    class ModuleList(Module):
        def __init__(self,items=()):
            super().__init__()
            for item in items:self.append(item)
        def __iter__(self):return iter(self._modules.values())
        def __len__(self):return len(self._modules)
        def __getitem__(self,key):return self._modules[str(key)]
        def append(self,item):self.add_module(str(len(self)),item);return self
    cp=".venv/Lib/site-packages/torch/nn/modules/container.py"
    env={"_copy_to_script_wrapper":lambda fn:fn}
    execute(nodes(cp,"ModuleList",("__iter__","__len__","append")),env,str(ROOT/cp))
    for name in ("__iter__","__len__","append"):setattr(ModuleList,name,env[name])
    class HookPoint(Module):pass
    gp=".venv/Lib/site-packages/transformer_lens/model_bridge/generalized_components/base.py"
    init=ast.parse('def __init__(self,name,config=None,submodules=None,optional=False):\n'
        '    super().__init__()\n    self.name=name\n    self.config=config\n'
        '    self.submodules=submodules or {}\n    self.optional=optional\n'
        '    self.real_components={}\n    self.hook_in=Module()\n    self.hook_out=Module()\n').body[0]
    methods=nodes(gp,"GeneralizedComponent",("set_original_component","original_component","__getattr__","__setattr__"))
    klass=ast.ClassDef(name="GeneralizedComponent",bases=[ast.Name(id="Module",ctx=ast.Load())],keywords=[],
        body=[init,ast.parse("is_list_item=False").body[0]]+methods,decorator_list=[])
    env={"Module":Module,"HookPoint":HookPoint}
    execute([klass],env,str(ROOT/gp));Generalized=env["GeneralizedComponent"]
    class BlockBridge(Generalized):is_list_item=True
    class GDNBridge(Generalized):pass
    class GDN(Module):pass
    class Decoder(Module):pass
    class TransformerBridge(Module):
        @property
        def original_model(self):return self.__dict__["original_model"]
    class Symbolic(Generalized):pass
    cfg=types.SimpleNamespace(layer_types=LAYER_TYPES,num_hidden_layers=24,hidden_size=1024)
    # Decoder constructor statements are exact field/branch/registry structure.
    # super(), numerical leaf constructors and config reads are replaced with inert leaves.
    hp=".venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    node=nodes(hp,"Qwen3_5DecoderLayer",("__init__",))[0]
    class Leaf(Module):
        def __init__(self,*args,**kwargs):super().__init__()
    class GDN(GDN):
        def __init__(self,*args,**kwargs):super().__init__()
    env={"Module":Module,"Qwen3_5GatedDeltaNet":GDN,"Qwen3_5Attention":Leaf,
         "Qwen3_5MLP":Leaf,"Qwen3_5RMSNorm":Leaf}
    klass=ast.ClassDef(name="Decoder",bases=[ast.Name(id="Module",ctx=ast.Load())],keywords=[],
                      body=[node],decorator_list=[])
    execute([klass],env,str(ROOT/hp));Decoder=env["Decoder"]
    # Config values needed by unchanged constructor, without a config/model import.
    cfg.rms_norm_eps=1e-6;cfg.intermediate_size=3584
    qp=".venv/Lib/site-packages/transformer_lens/model_bridge/supported_architectures/qwen3.py"
    names=("_build_attention_bridge","_build_mlp_bridge","_build_linear_attn_bridge","_build_component_mapping")
    methods=nodes(qp,"Qwen3ArchitectureAdapter",names)
    env={"PositionEmbeddingsAttentionBridge":Generalized,"LinearBridge":Generalized,
         "RMSNormalizationBridge":Generalized,"GatedMLPBridge":Generalized,"GatedDeltaNetBridge":GDNBridge,
         "EmbeddingBridge":Generalized,"RotaryEmbeddingBridge":Generalized,"BlockBridge":BlockBridge,
         "UnembeddingBridge":Generalized}
    klass=ast.ClassDef(name="Adapter",bases=[],keywords=[],body=methods,decorator_list=[])
    execute([klass],env,str(ROOT/qp));Adapter=env["Adapter"]
    def remote(self,obj,path):
        for name in path.split("."):obj=getattr(obj,name)
        return obj
    Adapter.get_remote_component=remote
    Adapter.get_component_mapping=lambda self:self.mapping
    sp=".venv/Lib/site-packages/transformer_lens/model_bridge/component_setup.py"
    funcs=nodes(sp,None,("replace_remote_component","set_original_components","setup_submodules","setup_components","setup_blocks_bridge"))
    env={"copy":copy,"logger":logging.getLogger("inert_selection"),"nn":types.SimpleNamespace(ModuleList=ModuleList),
         "SymbolicBridge":Symbolic,"cast":lambda typ,obj:obj,"Any":object}
    execute(funcs,env,str(ROOT/sp))
    return types.SimpleNamespace(Module=Module,ModuleList=ModuleList,BlockBridge=BlockBridge,GDNBridge=GDNBridge,
        GDN=GDN,Decoder=Decoder,TransformerBridge=TransformerBridge,Adapter=Adapter,
        cfg=cfg,setup=env["set_original_components"])

def topology(t):
    hf=t.Module();hf.config=t.cfg;hf.model=t.Module()
    hf.model.embed_tokens=t.Module();hf.model.rotary_emb=t.Module()
    decoders=[t.Decoder(t.cfg,i) for i in range(24)]
    # Supply only inert named leaves consumed by the exact mapping/setup.
    for decoder in decoders:
        for name in ("gate_proj","up_proj","down_proj"):setattr(decoder.mlp,name,t.Module())
        if decoder.block_type=="full_attention":
            for name in ("q_proj","k_proj","v_proj","o_proj","q_norm","k_norm"):setattr(decoder.self_attn,name,t.Module())
    hf.model.layers=t.ModuleList(decoders);hf.model.norm=t.Module();hf.lm_head=t.Module()
    bridge=t.TransformerBridge();bridge.__dict__["original_model"]=hf;bridge.real_components={}
    adapter=t.Adapter();adapter.cfg=t.cfg
    adapter.mapping=adapter._build_component_mapping(hybrid=True,lm_prefix="model")
    t.setup(bridge,adapter,hf) # Exact installed setup AST; both replacement sites execute.
    bindings=TypeBindings(t.BlockBridge,t.Decoder,t.GDNBridge,t.GDN,t.TransformerBridge,t.ModuleList,"INERT_FIXTURE")
    return types.SimpleNamespace(hf=hf,bridge=bridge,bindings=bindings,decoders=decoders,adapter=adapter)
