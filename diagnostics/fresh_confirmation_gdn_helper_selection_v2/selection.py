"""Post-bridge original-instance graph, metadata only; no computation."""
import json
import pathlib
import sys
from dataclasses import dataclass
from support import need,authenticate,encoded,sha,ROOT

LAYER_TYPES=tuple("full_attention" if (i+1)%4==0 else "linear_attention" for i in range(24))
LABELS=tuple("model.layers."+str(i)+".linear_attn" for i,x in enumerate(LAYER_TYPES) if x=="linear_attention")
TYPE_SOURCES={
 "block":("transformer_lens.model_bridge.generalized_components.block","BlockBridge"),
 "decoder":("transformers.models.qwen3_5.modeling_qwen3_5","Qwen3_5DecoderLayer"),
 "gdn_bridge":("transformer_lens.model_bridge.generalized_components.gated_delta_net","GatedDeltaNetBridge"),
 "gdn":("transformers.models.qwen3_5.modeling_qwen3_5","Qwen3_5GatedDeltaNet"),
 "bridge":("transformer_lens.model_bridge.transformer_bridge","TransformerBridge"),
 "list":("torch.nn.modules.container","ModuleList")}
@dataclass(frozen=True)
class TypeBindings:
    block:type
    decoder:type
    gdn_bridge:type
    gdn:type
    bridge:type
    list:type
    mode:str
    def verify(self,targets):
        pins=authenticate()
        need(self.mode==targets.mode and self.mode in ("INERT_FIXTURE","LIVE_EXISTING_MODULE"),"TYPE_MODE")
        need(self.gdn is targets.original_type,"ORIGINAL_TYPE")
        records={}
        for role,(module_name,class_name) in TYPE_SOURCES.items():
            cls=getattr(self,role)
            need(isinstance(cls,type),"TYPE_BINDING")
            source=".venv/Lib/site-packages/"+module_name.replace(".","/")+".py"
            need(source in pins["files"],"TYPE_SOURCE")
            if self.mode=="LIVE_EXISTING_MODULE":
                module=sys.modules.get(module_name)
                need(module is not None and getattr(module,class_name,None) is cls and
                     pathlib.Path(module.__file__).resolve()==(ROOT/source).resolve() and
                     cls.__module__==module_name and cls.__name__==class_name,"TYPE_BINDING")
            records[role]={"identity":id(cls),"source_sha256":pins["files"][source]["sha256"]}
        need(len({id(getattr(self,k)) for k in TYPE_SOURCES})==6,"DISTINCT_TYPES")
        return records

def registry(obj):
    value=vars(obj).get("_modules")
    need(type(value) is dict,"MODULE_REGISTRY")
    return value

def child(obj,name):
    value=registry(obj).get(name)
    need(value is not None,"MISSING_MODULE")
    return value

def original(wrapper):
    raw=child(wrapper,"_original_component")
    need(wrapper.original_component is raw,"ORIGINAL_PROPERTY")
    return raw

def walk(root):
    """Torch named_modules(remove_duplicate=False) order, with finite cycle guard."""
    count=0
    def visit(obj,path,ancestors,depth):
        nonlocal count
        count+=1
        need(count<=65536 and depth<=64 and len(path)<=1024,"TRAVERSAL_BOUND")
        need(id(obj) not in ancestors,"MODULE_CYCLE")
        yield path,obj
        next_ancestors=ancestors+(id(obj),)
        for name,module in registry(obj).items():
            need(type(name) is str and name and "." not in name,"MODULE_NAME")
            if module is not None:
                yield from visit(module,path+("." if path else "")+name,next_ancestors,depth+1)
    yield from visit(root,"",(),0)

class Selected(tuple):
    def __new__(cls,values,graph,strong):
        obj=super().__new__(cls,values)
        obj.graph_raw=encoded(graph)
        obj.strong=tuple(strong) # Never reconstruct a reference from a later observation.
        return obj

def select(hf_model,bridge_model,targets,bridge_type):
    bindings=bridge_type
    need(type(bindings) is TypeBindings,"TYPE_BINDINGS_REQUIRED")
    type_records=bindings.verify(targets)
    need(type(bridge_model) is bindings.bridge and bridge_model.original_model is hf_model,"ROOT_IDENTITY")
    need(tuple(hf_model.config.layer_types)==LAYER_TYPES and hf_model.config.num_hidden_layers==24,"LAYER_LAYOUT")
    layers=child(child(hf_model,"model"),"layers")
    blocks=child(bridge_model,"blocks")
    need(hf_model.model.layers is layers and bridge_model.blocks is blocks and
         layers is blocks and type(layers) is bindings.list,"SHARED_BLOCKS")
    need(tuple(registry(blocks))==tuple(str(i) for i in range(24)),"LAYER_COUNT")
    selected=[];rows=[];strong=[hf_model,bridge_model,layers,bindings,targets]
    for i,kind in enumerate(LAYER_TYPES):
        block=child(blocks,str(i))
        need(type(block) is bindings.block and block.name=="model.layers."+str(i),"BLOCK_IDENTITY")
        decoder=original(block)
        need(type(decoder) is bindings.decoder and getattr(decoder,"block_type",None)==kind,"LAYER_KIND")
        row={"index":i,"kind":kind,"block":id(block),"decoder":id(decoder)}
        strong.extend((block,decoder))
        if kind=="linear_attention":
            wrapper=child(block,"linear_attn")
            need(child(decoder,"linear_attn") is wrapper and type(wrapper) is bindings.gdn_bridge and
                 block.linear_attn is wrapper and decoder.linear_attn is wrapper and wrapper.name=="linear_attn",
                 "MAPPING_IDENTITY")
            gdn=original(wrapper)
            need(type(gdn) is bindings.gdn,"MAPPING_IDENTITY")
            row.update(gdn_bridge=id(wrapper),gdn=id(gdn))
            selected.append(("model.layers."+str(i)+".linear_attn",gdn))
            strong.extend((wrapper,gdn))
        else:
            need(registry(block).get("linear_attn") is None and registry(decoder).get("linear_attn") is None and
                 getattr(block,"linear_attn",None) is None and getattr(decoder,"linear_attn",None) is None,
                 "FULL_LAYER_GDN")
        rows.append(row)
    need(len({r["block"] for r in rows})==24 and len({r["decoder"] for r in rows})==24 and
         len({r["gdn_bridge"] for r in rows if "gdn" in r})==18 and len({id(x[1]) for x in selected})==18,
         "COMPLETE_UNIQUE_SET")
    roles=("block","decoder","gdn_bridge","gdn")
    graph_roots={}
    for root_name,root,prefix in (("hf",hf_model,"model.layers"),("bridge",bridge_model,"blocks")):
        observed=[]
        for path,obj in walk(root):
            matching=[role for role in roles if isinstance(obj,getattr(bindings,role))]
            if matching:
                need(len(matching)==1 and type(obj) is getattr(bindings,matching[0]),"PATH_TYPE")
                observed.append([path,matching[0],id(obj)])
        expected=[]
        for r in rows:
            p=prefix+"."+str(r["index"])
            expected.extend([[p,"block",r["block"]],[p+"._original_component","decoder",r["decoder"]]])
            if "gdn" in r:
                for q in (p+"._original_component.linear_attn",p+".linear_attn"):
                    expected.extend([[q,"gdn_bridge",r["gdn_bridge"]],[q+"._original_component","gdn",r["gdn"]]])
        need(observed==expected,"COMPLETE_ALIAS_GRAPH")
        graph_roots[root_name]=observed
    graph={"schema":"post_bridge_selection.v2","mode":bindings.mode,"types":type_records,
           "roots":{"hf":id(hf_model),"bridge":id(bridge_model),"shared_blocks":id(blocks)},
           "layers":rows,"paths":graph_roots,"layer_count":24,"original_gdn_count":18,
           "original_occurrences_per_root":36}
    need(len(encoded(graph))<=32768,"GRAPH_CAP")
    return Selected(selected,graph,strong)

def saved_selection(selected):
    return [{"label":label,"identity":id(obj)} for label,obj in selected]

def saved_graph(selected):
    return json.loads(selected.graph_raw)

def graph_sha(selected):
    return sha(selected.graph_raw) if selected else None
