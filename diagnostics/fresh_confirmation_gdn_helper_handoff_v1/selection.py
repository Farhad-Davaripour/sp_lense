"""Complete fixed text-Qwen mapping; no parameter enumeration or values."""
from support import need,authenticate

LAYER_TYPES=tuple("full_attention" if (i+1)%4==0 else "linear_attention" for i in range(24))
LABELS=tuple("model.layers."+str(i)+".linear_attn" for i,x in enumerate(LAYER_TYPES) if x=="linear_attention")

def select(hf_model,bridge_model,targets,bridge_type):
    authenticate()
    need(tuple(hf_model.config.layer_types)==LAYER_TYPES and hf_model.config.num_hidden_layers==24,"LAYER_LAYOUT")
    layers=tuple(hf_model.model.layers);blocks=tuple(bridge_model.blocks)
    need(len(layers)==len(blocks)==24,"LAYER_COUNT")
    selected=[]
    for i,(kind,layer,block) in enumerate(zip(LAYER_TYPES,layers,blocks)):
        original=getattr(layer,"linear_attn",None)
        wrapped=getattr(block,"linear_attn",None)
        need(layer.layer_type==kind,"LAYER_KIND")
        if kind=="linear_attention":
            need(type(original) is targets.original_type and type(wrapped) is bridge_type and
                wrapped.original_component is original,"MAPPING_IDENTITY")
            selected.append(("model.layers."+str(i)+".linear_attn",original))
        else:
            need(original is None and (wrapped is None or wrapped.original_component is None),"FULL_LAYER_GDN")
    need(tuple(x[0] for x in selected)==LABELS and len({id(x[1]) for x in selected})==18,"COMPLETE_UNIQUE_SET")
    # remove_duplicate=False is essential: aliases/duplicates must remain visible.
    traversed=tuple((name,obj) for name,obj in hf_model.named_modules(remove_duplicate=False)
                    if type(obj) is targets.original_type)
    need(tuple((name,id(obj)) for name,obj in traversed)==tuple((name,id(obj)) for name,obj in selected),
         "COMPLETE_TRAVERSAL_BIJECTION")
    return tuple(selected)

def saved_selection(selected):
    return [{"label":label,"identity":id(obj)} for label,obj in selected]
