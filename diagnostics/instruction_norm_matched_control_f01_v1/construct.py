"""Authenticate archives and freeze one-shot absolute float32 offsets before loading."""
import array
import json
import math
import sys
import zlib

from core import HERE, ROOT, Budget, CAP, TOTAL_CAP, FILE_CAP, INTEGRITY_TOL, REPLAY_TOL, git, require, sha
from transfer import norm, f32, prepare

SOURCES = {
 "last": ("33013ccb3e5520e01e534fafc8dabc2619b90f82", "diagnostics/instruction_activation_transfer_f01_v1/", "c7dcc6d5f96f8374604b73d4602fe524f7d17c4eef51c47067956e9547931d7c"),
 "suffix": ("b39f9cd878f60ec75fd641df95ef3a3b5af55c4e", "diagnostics/instruction_suffix_transfer_f01_v1/", "2a84282e411d7a43851e625f9439f72fdc879e32010a810153bd9ce72c887ee8"),
}

class Archive:
    def __init__(self, commit, namespace, inventory_sha):
        self.commit, self.namespace = commit, namespace
        raw = git("show", commit+":"+namespace+"FINAL_INVENTORY.json")
        require(sha(raw) == inventory_sha, "source inventory authentication")
        self.entries = {r["path"]:r for r in json.loads(raw)["files"]}
        self.accessed = {}
    def get(self,name):
        require(name in self.entries, "unbound source artifact")
        raw = (ROOT/self.namespace/name).read_bytes()
        require(sha(raw)==self.entries[name]["sha256"] and len(raw)==self.entries[name]["bytes"], "source artifact authentication")
        self.accessed[name]=sha(raw)
        return raw
    def json(self,name):
        return json.loads(self.get(name))

def authenticate(commit,namespace,inventory_sha):
    return Archive(commit,namespace,inventory_sha)

def decode(raw,shape):
    n,width=shape
    require(1<=n<=256 and width==1024 and len(raw)==n*width*4,"matrix dimensions")
    values=array.array("f")
    values.frombytes(raw)
    if sys.byteorder!="little": values.byteswap()
    require(all(math.isfinite(x) for x in values),"nonfinite matrix")
    return [list(values[j*width:(j+1)*width]) for j in range(n)]

def packed(matrix):
    require(1<=len(matrix)<=256 and all(len(row)==1024 for row in matrix),"matrix dimensions")
    require(all(math.isfinite(x) for row in matrix for x in row),"nonfinite matrix")
    values=array.array("f",(x for row in matrix for x in row))
    if sys.byteorder!="little": values.byteswap()
    return values.tobytes()

def save_matrix(budget,name,matrix):
    raw=packed(matrix)
    compressed=zlib.compress(raw)
    budget.write_bytes(name,compressed)
    return {"file":name,"shape":[len(matrix),1024],"dtype":"float32","byte_order":"little","order":"row-major",
            "raw_bytes":len(raw),"raw_sha256":sha(raw),"compressed_sha256":sha(compressed)}

def load_matrix(root,item):
    require(item["dtype"]=="float32" and item["byte_order"]=="little" and item["order"]=="row-major","matrix schema")
    target=(root/item["file"]).resolve()
    require(target.is_relative_to(root.resolve()),"matrix path")
    compressed=target.read_bytes()
    require(sha(compressed)==item["compressed_sha256"],"matrix compressed identity")
    raw=zlib.decompress(compressed)
    require(sha(raw)==item["raw_sha256"] and len(raw)==item["raw_bytes"],"matrix raw identity")
    return decode(raw,item["shape"])

def archived_window(archive,index):
    meta=archive.json(f"states/{index:02d}.json")
    item=meta["binary"]
    compressed=archive.get(item["file"])
    require(sha(compressed)==item["compressed_sha256"],"archived compressed identity")
    raw=zlib.decompress(compressed)
    require(sha(raw)==item["raw_sha256"] and len(raw)==item["raw_bytes"],"archived raw identity")
    size=item["shape"][0]*1024*4
    result={key:decode(raw[i*size:(i+1)*size],item["shape"]) for i,key in enumerate(item["arrays"])}
    require(len(raw)==size*len(result),"archived array count")
    return result,meta

def scaled_offset(pre,post,L):
    require(len(pre)==len(post) and 1<=len(pre)<=256,"source window")
    require(all(len(row)==1024 for row in pre+post),"source dimensions")
    D=[[float(a)-float(b) for a,b in zip(after,before,strict=True)] for before,after in zip(pre,post,strict=True)]
    require(all(math.isfinite(x) for row in D for x in row) and math.isfinite(L) and L>=0,"finite scaling")
    F=norm(x for row in D for x in row)
    require(F>0 and math.isfinite(F),"zero/nonfinite suffix norm")
    s=L/F
    offset=[[f32(s*x) for x in row] for row in D]
    require(all(math.isfinite(x) for row in offset for x in row),"finite scaled float32 offset")
    return offset,{"archived_last_actual_norm":L,"archived_suffix_actual_frobenius":F,"scale":s,"scale_hex":s.hex()}

def construct():
    from transfer import verify_fixed, apply_values, verify_pair
    budget=Budget(HERE)
    (HERE/"candidates").mkdir(exist_ok=False)
    archives={key:authenticate(*args) for key,args in SOURCES.items()}
    last,suffix=archives["last"],archives["suffix"]
    old=last.json("freeze.json")["plan"]
    prior=suffix.json("freeze.json")["plan"]
    require(old["prompts"]==prior["prompts"] and old["cells"]==prior["cells"],"unchanged parent prompt/pairing bytes")
    last_rows=[json.loads(line) for line in last.get("raw_rows.jsonl").decode().splitlines()]
    old_result=last.json("results.json")
    suffix_result=suffix.json("results.json")
    prompts=prior["prompts"][:4]
    require(all(p["role"]=="neutral_receiver" for p in prompts),"four exact neutrals")
    alignment={p["prompt_id"]:prior["alignment"][p["prompt_id"]] for p in prompts}
    baselines={}
    matrices={}
    for i,p in enumerate(prompts,1):
        matrix,_=archived_window(suffix,i)
        h0=matrix["post"]
        require(h0[-1]==last.json(f"states/{i:02d}.json")["post"],"last/suffix baseline identity")
        matrices[p["prompt_id"]]=h0
        baselines[p["prompt_id"]]=save_matrix(budget,f"candidates/{p['prompt_id']}_archived.f32.zlib",h0)
        require(1<=len(h0)==alignment[p["prompt_id"]]["suffix_length"]<=256,"fixed suffix bound")
    candidates={}
    controls=[]
    edits=[]
    for i,parent in enumerate(prior["cells"][12:],13):
        key=parent["cell_id"].removesuffix("_edit")
        h0=matrices[parent["baseline_cell_id"]]
        state=last.json(f"states/{i:02d}.json")
        donor=last.json(f"states/{i-8:02d}.json")
        recipe=prepare(h0[-1],donor["post"])
        require(recipe["planned_applied"]==state["planned_applied"],"locked last-token recipe replay")
        require(state["pre"]==h0[-1] and state["post"]==[f32(b+d) for b,d in zip(h0[-1],recipe["planned_applied"],strict=True)],"archived last arithmetic")
        L=norm(float(a)-float(b) for a,b in zip(state["post"],state["pre"],strict=True))
        require(L==state["actual_norm"],"archived realized last norm")
        window,_=archived_window(suffix,i)
        require(window["pre"]==h0,"archived original suffix baseline")
        offset,scaling=scaled_offset(window["pre"],window["post"],L)
        for kind,values,start in (("control",[recipe["planned_applied"]],len(h0)-1),("suffix",offset,0)):
            cell={k:v for k,v in parent.items() if k not in ("donor_cell_id","cell_id","kind")}
            cell.update(cell_id=key+"_"+kind,kind=kind,pair_id=key,paired_control_cell_id=key+"_control")
            item=save_matrix(budget,"candidates/"+cell["cell_id"]+".f32.zlib",values)
            item.update(scaling,mask=alignment[cell["prompt_id"]]["selected_positions"][start:],window_offset=start,source_edit_index=i)
            candidates[cell["cell_id"]]=item
            (controls if kind=="control" else edits).append(cell)
        ctrl=verify_fixed(h0,h0,apply_values(h0,[recipe["planned_applied"]],len(h0)-1),[recipe["planned_applied"]],len(h0)-1)
        edited=verify_fixed(h0,h0,apply_values(h0,offset,0),offset,0)
        verify_pair(ctrl["actual_total_norm"],edited["actual_total_norm"])
        candidates[key+"_suffix"]["prospective_realized_pair_norm_error"]=abs(ctrl["actual_total_norm"]-edited["actual_total_norm"])
        replay=last_rows[i-1]
        last.get(replay["logits_file"])
        candidates[key+"_control"]["archived_logit_row"]=replay
        candidates[key+"_control"]["archived_scored_row"]=next(r for r in old_result["rows"] if r["cell_id"]==parent["cell_id"])
    n=max(a["suffix_length"] for a in alignment.values())
    def bound(x): return x+(x>>12)+(x>>14)+(x>>25)+13
    total=36*bound(n*1024*4)+20*bound(248320*4)+sum((HERE/x["file"]).stat().st_size for x in list(candidates.values())+list(baselines.values()))+10*1024**2
    require(total<TOTAL_CAP and bound(2*n*1024*4)<FILE_CAP,"prospective storage ceiling")
    sources={key:{"commit":args[0],"namespace":args[1],"inventory_sha256":args[2],"accessed":archives[key].accessed} for key,args in SOURCES.items()}
    plan={**prior,"prompts":prompts,"cells":prior["cells"][:4]+controls+edits,"alignment":alignment,
          "sources":sources,"candidates":candidates,"archived_baselines":baselines,
          "provenance":{"selection":"same four neutral prompts and all eight P/C pairings; no donors forwarded"},
          "intervention":{"hook":"blocks.10.hook_out","d_model":1024,"cap_per_token":CAP,"baseline_tolerance":INTEGRITY_TOL,
             "arithmetic":"D=binary64(archived suffix post)-binary64(pre); F=sqrt(fsum(all D_ij squared)); L=sqrt(fsum(archived last post-pre squared)); s=L/F once; offset32=f32(s*D); post32=f32(fresh_pre32+offset32). Last control offset32 reconstructs locked prepare(h0,hd) from archived states. No corrections.",
             "masks":"control final encoded input token; suffix exact archived65-token mask; outside mask bytes unchanged",
             "paired_realized_norm_absolute_tolerance":INTEGRITY_TOL,"replay_logit_max_abs_tolerance":REPLAY_TOL,
             "replay_tolerance_source":"scripts/verify_local_controllability.py ARITHMETIC_TOL=2e-5; exact hashes and choices also reported"},
          "comparison_rule":"At least one additional strict eligible suffix flip versus paired last control, with zero losses of strict flips or retentions; all integrity/replay checks valid. Margin-only improvements do not pass.",
          "prior_counts":{"last_strict":old_result["receiver_strict_passes"],"last_flips":old_result["flips"],"larger_suffix_strict":suffix_result["receiver_strict_passes"],"larger_suffix_flips":suffix_result["flips"]},
          "storage":{"conservative_total_bytes":total,"namespace_ceiling":TOTAL_CAP,"file_ceiling":FILE_CAP,"max_tokens":n,"encoding":"zlib little-endian float32 row-major; authenticated dimensions"},
          "limits":{**prior["limits"],"namespace_bytes":TOTAL_CAP}}
    budget.write("inputs.json",plan)
    print(json.dumps({"status":"PASS","cells":20,"prompts":4,"candidate_offsets":16,"storage_bound":total,"model_loaded":False}))

if __name__=="__main__":
    construct()
