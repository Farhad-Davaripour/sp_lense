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



def apply_values(pre,offset,start):
    require(0<=start<len(pre) and start+len(offset)==len(pre),"fixed tail mask")
    return [list(row) for row in pre[:start]]+[
        [f32(a+d) for a,d in zip(row,delta,strict=True)] for row,delta in zip(pre[start:],offset,strict=True)]


def verify_pair(control,suffix):
    require(math.isfinite(control) and math.isfinite(suffix) and abs(control-suffix)<=INTEGRITY_TOL,"realized paired total norms mismatch")
    return abs(control-suffix)


def baseline_match(archived,current):
    require(len(archived)==len(current) and all(len(row)==1024 for row in archived+current),"baseline dimensions")
    require(all(math.isfinite(x) for row in archived+current for x in row),"nonfinite baseline")
    error=max(abs(float(a)-float(b)) for x,y in zip(archived,current,strict=True) for a,b in zip(x,y,strict=True))
    require(error<=INTEGRITY_TOL,"archived/fresh baseline mismatch")
    return error


def verify_fixed(h0,pre,post,offset,start):
    error=baseline_match(h0,pre)
    require(post==apply_values(pre,offset,start),"fixed float32 applied arithmetic mismatch")
    require(pre[:start]==post[:start],"outside mask changed")
    per=[]
    actual_flat=[]
    for j,(base,before,after,delta) in enumerate(zip(h0[start:],pre[start:],post[start:],offset,strict=True),start):
        base_norm=norm(base)
        require(base_norm>0 and math.isfinite(base_norm),"zero/nonfinite h0 norm")
        require(all(math.isfinite(x) for x in after+delta),"nonfinite realized displacement")
        actual=[float(a)-float(b) for a,b in zip(after,before,strict=True)]
        actual_flat.extend(actual)
        value=norm(actual)
        require(value<=CAP*base_norm+INTEGRITY_TOL,"realized per-token cap")
        per.append({"window_offset":j,"h0_norm":base_norm,"planned_norm":norm(delta),
                    "actual_norm":value,"actual_relative_norm":value/base_norm})
    total=norm(actual_flat)
    base_total=norm(x for row in h0 for x in row)
    return {"per_token":per,"actual_total_norm":total,"planned_total_norm":norm(x for row in offset for x in row),
            "actual_relative_window_norm":total/base_total,"pre_baseline_max_abs_error":error}


def patch_fixed(activation,start,offset,torch):
    changed=activation.clone()
    changed[0,start:,:]=activation[0,start:,:]+torch.tensor(offset,dtype=torch.float32,device=activation.device)
    return changed


class CaptureHook:
    def __init__(self,torch,alignment,h0,offset=None,window_offset=0,expected_norm=None):
        self.torch,self.alignment,self.h0,self.offset=torch,alignment,h0,offset
        self.window_offset,self.expected_norm=window_offset,expected_norm
        self.calls=0
        self.state=None

    def __call__(self,activation,hook):
        del hook
        self.calls+=1
        require(self.calls==1,"hook fired more than once")
        start=self.alignment["first_token_index"]
        mask_start=start+self.window_offset if self.offset is not None else int(activation.shape[1])
        require(tuple(activation.shape)==(1,len(self.alignment["full_token_ids"]),1024),"hook shape/alignment")
        pre=activation[0,start:,:].detach().float().cpu().tolist()
        outside=activation[:,:mask_start,:].detach().float().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes()
        self.state={"pre":pre,"hook_calls":self.calls,"sequence_length":int(activation.shape[1]),
                    "input_token_index":int(activation.shape[1])-1,"suffix_length":len(pre),
                    "selected_positions":self.alignment["selected_positions"],
                    "edited_positions":list(range(mask_start,int(activation.shape[1]))) if self.offset is not None else [],
                    "outside_sha_before":sha(outside)}
        try:
            self.state["archived_or_fresh_baseline_max_abs_error"]=baseline_match(self.h0,pre)
            changed=activation if self.offset is None else patch_fixed(activation,mask_start,self.offset,self.torch)
            after_outside=changed[:,:mask_start,:].detach().float().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes()
            post=changed[0,start:,:].detach().float().cpu().tolist()
            self.state.update(post=post,outside_sha_after=sha(after_outside),
                              outside_max_abs_difference=float((changed[:,:mask_start,:]-activation[:,:mask_start,:]).abs().max().item()))
            require(outside==after_outside,"outside fixed mask changed")
            if self.offset is not None:
                self.state.update(verify_fixed(self.h0,pre,post,self.offset,self.window_offset))
                self.state["paired_or_archived_norm_error"]=verify_pair(self.expected_norm,self.state["actual_total_norm"])
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
