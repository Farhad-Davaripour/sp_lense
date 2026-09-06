"""Per-token inherited cap over one fixed matched suffix, with compact states."""
import array
import math
import struct
import sys
import zlib

from core import HERE, CAP, INTEGRITY_TOL, require, sha, read

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def norm(values):
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def prepare(h0, donor):
    require(len(h0) == len(donor) == 1024, "1024-dimensional states")
    require(all(math.isfinite(x) for x in h0 + donor), "nonfinite state")
    base_norm = norm(h0)
    require(base_norm > 0 and math.isfinite(base_norm), "zero/nonfinite h0 norm")
    raw = [float(d) - float(b) for d, b in zip(donor, h0, strict=True)]
    raw_norm = norm(raw)
    factor = 1. if raw_norm == 0 else min(1., CAP * base_norm / raw_norm)
    planned = [f32(factor * x) for x in raw]
    require(all(math.isfinite(x) for x in raw + planned), "nonfinite displacement")
    return {"h0_norm": base_norm, "raw_delta": raw, "raw_delta_norm": raw_norm,
            "factor": factor, "planned_applied": planned, "planned_norm": norm(planned),
            "raw_zero": raw_norm == 0, "clipped": factor < 1.}


def verify_realized(h0, pre, post, plan):
    require(len(pre) == len(post) == len(h0) == 1024, "state dimensions")
    require(all(math.isfinite(x) for x in pre + post), "nonfinite realized state")
    baseline_error = max(abs(float(a)-float(b)) for a, b in zip(pre, h0, strict=True))
    require(baseline_error <= INTEGRITY_TOL, "pre-edit baseline mismatch")
    expected = [f32(a + d) for a, d in zip(pre, plan["planned_applied"], strict=True)]
    require(post == expected, "float32 applied arithmetic mismatch")
    actual = [float(a)-float(b) for a, b in zip(post, pre, strict=True)]
    actual_norm = norm(actual)
    require(actual_norm <= CAP * plan["h0_norm"] + INTEGRITY_TOL, "realized norm exceeds cap")
    return {"pre_baseline_max_abs_error": baseline_error, "actual_applied": actual,
            "actual_norm": actual_norm, "actual_relative_norm": actual_norm / plan["h0_norm"]}



def prepare_window(h0, donor):
    require(len(h0)==len(donor) and 1<=len(h0)<=256,"fixed window dimensions")
    return [prepare(b,d) for b,d in zip(h0,donor,strict=True)]


def verify_window(h0, donor, pre, post, plans):
    require(len(h0)==len(donor)==len(pre)==len(post)==len(plans),"window row count")
    metrics=[]
    for j,(b,d,before,after,plan) in enumerate(zip(h0,donor,pre,post,plans,strict=True)):
        actual=verify_realized(b,before,after,plan)
        difference=[float(a)-float(x) for a,x in zip(after,d,strict=True)]
        metrics.append({"offset":j,**{k:v for k,v in plan.items() if k not in ("raw_delta","planned_applied")},
                        **{k:v for k,v in actual.items() if k!="actual_applied"},
                        "post_minus_donor_norm":norm(difference),"post_minus_donor_max_abs":max(abs(x) for x in difference)})
    def frob(key):
        return math.sqrt(math.fsum(m[key]*m[key] for m in metrics))
    base=frob("h0_norm")
    actual=frob("actual_norm")
    return {"per_token":metrics,"aggregate_h0_frobenius":base,
            "aggregate_raw_delta_frobenius":frob("raw_delta_norm"),
            "aggregate_planned_frobenius":frob("planned_norm"),
            "aggregate_actual_frobenius":actual,"aggregate_actual_relative_frobenius":actual/base,
            "aggregate_post_minus_donor_frobenius":frob("post_minus_donor_norm"),
            "clipped_tokens":sum(m["clipped"] for m in metrics),
            "maximum_actual_relative_token_norm":max(m["actual_relative_norm"] for m in metrics)}


def patch_window(activation, start, plans, torch):
    changed=activation.clone()
    changed[0,start:,:]=activation[0,start:,:]+torch.tensor(
        [p["planned_applied"] for p in plans],dtype=torch.float32,device=activation.device)
    return changed


class CaptureHook:
    def __init__(self,torch,alignment,h0=None,donor=None,plans=None):
        self.torch,self.alignment,self.h0,self.donor,self.plans=torch,alignment,h0,donor,plans
        self.calls=0
        self.state=None

    def __call__(self,activation,hook):
        del hook
        self.calls+=1
        require(self.calls==1,"hook fired more than once")
        start=self.alignment["first_token_index"]
        require(tuple(activation.shape)==(1,len(self.alignment["full_token_ids"]),1024),"hook shape/alignment")
        pre=activation[0,start:,:].detach().float().cpu().tolist()
        outside=activation[:,:start,:].detach().float().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes()
        self.state={"pre":pre,"hook_calls":self.calls,"sequence_length":int(activation.shape[1]),
                    "input_token_index":int(activation.shape[1])-1,"suffix_length":len(pre),
                    "selected_positions":self.alignment["selected_positions"],"outside_sha_before":sha(outside)}
        try:
            require(all(math.isfinite(x) for row in pre for x in row),"nonfinite captured window")
            changed=activation if self.plans is None else patch_window(activation,start,self.plans,self.torch)
            after_outside=changed[:,:start,:].detach().float().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes()
            post=changed[0,start:,:].detach().float().cpu().tolist()
            self.state.update(post=post,outside_sha_after=sha(after_outside),
                              outside_max_abs_difference=float((changed[:,:start,:]-activation[:,:start,:]).abs().max().item()))
            require(outside==after_outside,"outside fixed suffix changed")
            if self.plans is not None:
                self.state.update(verify_window(self.h0,self.donor,pre,post,self.plans))
            else:
                require(pre==post,"capture-only window changed")
            self.state["integrity_passed"]=True
            return changed
        except BaseException as error:
            self.state["integrity_passed"]=False
            self.state["integrity_error"]=type(error).__name__+": "+str(error)[:1024]
            raise


def write_capture(budget,index,state,edited):
    arrays=["pre","post"] if edited else ["post"]
    require(1<=len(state["post"])<=256,"state window bound")
    packed=array.array("f",(x for key in arrays for row in state[key] for x in row))
    if sys.byteorder!="little": packed.byteswap()
    raw=packed.tobytes()
    compressed=zlib.compress(raw)
    binary=f"states/{index:02d}.f32.zlib"
    budget.write_bytes(binary,compressed)
    metadata={k:v for k,v in state.items() if k not in ("pre","post")}
    if "per_token" in metadata:
        import json
        require(all(len(json.dumps(m).encode())<=2048 for m in metadata["per_token"]),"per-token metadata storage bound")
    metadata["binary"]={"file":binary,"arrays":arrays,"shape":[len(state["post"]),1024],
                        "dtype":"float32","byte_order":"little","order":"row-major",
                        "raw_bytes":len(raw),"raw_sha256":sha(raw),"compressed_sha256":sha(compressed)}
    name=f"states/{index:02d}.json"
    budget.write(name,metadata)
    return name


def load_capture(output,name):
    state=read(output/name)
    binary=state["binary"]
    n,width=binary["shape"]
    require(1<=n<=256 and width==1024 and binary["dtype"]=="float32"
            and binary["byte_order"]=="little" and binary["order"]=="row-major","binary state schema")
    require(binary["arrays"] in (["post"],["pre","post"]),"binary arrays")
    target=(output/binary["file"]).resolve()
    require(target.is_relative_to((output/"states").resolve()),"binary state path")
    compressed=target.read_bytes()
    require(sha(compressed)==binary["compressed_sha256"],"binary compressed state hash")
    raw=zlib.decompress(compressed)
    require(len(raw)==binary["raw_bytes"]==len(binary["arrays"])*n*width*4
            and sha(raw)==binary["raw_sha256"],"binary raw state identity")
    values=array.array("f")
    values.frombytes(raw)
    if sys.byteorder!="little": values.byteswap()
    for a,key in enumerate(binary["arrays"]):
        state[key]=[list(values[(a*n+j)*width:(a*n+j+1)*width]) for j in range(n)]
    if "pre" not in state: state["pre"]=state["post"]
    return state
