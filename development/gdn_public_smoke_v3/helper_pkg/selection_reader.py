"""Independent saved path-graph reconstruction; no live selection import."""
from .support import need,encoded,sha,authenticate,HERE
def verify_graph(graph,selection,mode,source_lock,admitted_sha):
    pins=authenticate()
    need(source_lock["pins_sha256"]==sha((HERE/"SOURCE_PINS.json").read_bytes()),"GRAPH_SOURCE_LOCK")
    need(sha(encoded(graph))==admitted_sha,"ADMITTED_GRAPH_SHA")
    need(graph["schema"]=="post_bridge_selection.v2" and graph["mode"]==
         ("INERT_FIXTURE" if mode=="INERT_FIXTURE" else "LIVE_EXISTING_MODULE"),"GRAPH_MODE")
    sources={
      "block":"transformer_lens/model_bridge/generalized_components/block.py",
      "decoder":"transformers/models/qwen3_5/modeling_qwen3_5.py",
      "gdn_bridge":"transformer_lens/model_bridge/generalized_components/gated_delta_net.py",
      "gdn":"transformers/models/qwen3_5/modeling_qwen3_5.py",
      "bridge":"transformer_lens/model_bridge/transformer_bridge.py",
      "list":"torch/nn/modules/container.py"}
    need(set(graph["types"])==set(sources),"GRAPH_TYPES")
    identities=[]
    for role,path in sources.items():
        row=graph["types"][role]
        need(set(row)=={"identity","source_sha256"} and
             row["source_sha256"]==pins["files"][".venv/Lib/site-packages/"+path]["sha256"],"GRAPH_TYPE_PIN")
        identities.append(row["identity"])
    roots=graph["roots"]
    need(set(roots)=={"hf","bridge","shared_blocks"},"GRAPH_ROOTS")
    identities.extend(roots.values())
    layers=graph["layers"]
    need(len(layers)==24 and graph["layer_count"]==24 and graph["original_gdn_count"]==18 and
         graph["original_occurrences_per_root"]==36,"GRAPH_COUNTS")
    expected_selection=[]
    for i,row in enumerate(layers):
        linear=(i+1)%4!=0
        keys={"index","kind","block","decoder"}|({"gdn_bridge","gdn"} if linear else set())
        need(set(row)==keys and row["index"]==i and
             row["kind"]==("linear_attention" if linear else "full_attention"),"GRAPH_LAYER")
        identities.extend(row[k] for k in ("block","decoder"))
        if linear:
            identities.extend((row["gdn_bridge"],row["gdn"]))
            expected_selection.append({"label":"model.layers."+str(i)+".linear_attn","identity":row["gdn"]})
    need(all(type(x) is int and x>0 for x in identities) and len(set(identities))==len(identities),"GRAPH_IDENTITY_BIJECTION")
    need(selection==expected_selection,"GRAPH_SEMANTIC_SELECTION")
    expected_roots={}
    for root,prefix in (("hf","model.layers"),("bridge","blocks")):
        paths=[]
        for i,row in enumerate(layers):
            p=prefix+"."+str(i)
            paths.extend([[p,"block",row["block"]],[p+"._original_component","decoder",row["decoder"]]])
            if (i+1)%4!=0:
                for q in (p+"._original_component.linear_attn",p+".linear_attn"):
                    paths.extend([[q,"gdn_bridge",row["gdn_bridge"]],[q+"._original_component","gdn",row["gdn"]]])
        expected_roots[root]=paths
    need(graph["paths"]==expected_roots,"SAVED_ALIAS_GRAPH")
    need(set(graph)=={"schema","mode","types","roots","layers","paths","layer_count","original_gdn_count",
                     "original_occurrences_per_root"} and len(encoded(graph))<=32768,"GRAPH_SCHEMA")
    return True
