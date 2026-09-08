"""Thin receiver handoff, deliberately no provider import or load entrypoint.

This class does not authorize model work. The future owned loader must supply a
receiver with a separately verified fixed legacy fingerprint, before any call.
"""
from reference import need
from replay import LIMITS,SCHEDULE

class ReceiverEngine:
    limits=LIMITS
    def __init__(self,receiver,counters,to_offset,to_bytes,weight_check,restore):
        self.receiver,self.counters=receiver,counters
        self.to_offset,self.to_bytes=to_offset,to_bytes
        self.weight_check,self.restore=weight_check,restore
        self.next=0;self.current=None;self.loads=1;self.forwards=0;self.derivatives=0
        self.admitted=weight_check()
        need(self.admitted is True,"FIXED_LEGACY_WEIGHT_ADMISSION")
        self.receiver.clean()

    def forward(self,cell,ids,mask,offset):
        need(self.next<len(SCHEDULE) and cell==SCHEDULE[self.next],"EXACT_REPLAY_SCHEDULE")
        r=self.receiver
        if cell=="entry":r.start_request("fixed_public_f03_C")
        if cell=="gradient_1":r.begin_edit()
        if cell=="endpoint":
            # Endpoint is a fresh cold request with the exact already-saved offset.
            r.finish_request();r.start_request("fixed_public_f03_C_cold_endpoint");r.begin_edit()
        r.clear_capture();self.counters.attempts["forward"]+=1;self.forwards+=1
        phase="baseline" if cell=="public_smoke" else cell
        logits,h=r.forward_inputs(ids,mask,phase,self.to_offset(offset));self.current=logits
        result={"logits_raw":self.to_bytes(logits),"h":h.detach().cpu().tolist(),
                "unselected_sha256":r.capture["unselected_sha256"],"nonfinal_unchanged":True}
        self.next+=1
        if cell not in ("gradient_1",):r.clear_capture();self.current=None
        return result

    def gradient(self):
        need(self.next==4 and self.current is not None and self.derivatives==0,"ONE_CURRENT_DERIVATIVE")
        self.counters.attempts["derivative"]+=1;self.derivatives+=1
        gradient=self.receiver.gradient(self.current).tolist()
        self.receiver.clear_capture();self.current=None
        return gradient

    def close(self):
        try:
            state=self.receiver.finalize()
            need(self.weight_check() is True,"FINAL_FIXED_LEGACY_WEIGHT")
            return {"complete":True,"native_state":state,"fixed_legacy_weight_verified":True}
        finally:self.restore()

    def accounting(self):
        return {"loads":self.loads,"forwards":self.forwards,"derivatives":self.derivatives,"encoding":0}

def real_entrypoint(*args,**kwargs):
    # Not a production permission bit. There is deliberately no loader reachable
    # until complete source-derived fingerprint mapping and owner binding exist.
    raise RuntimeError("NATIVE_REPLAY_REAL_ADMISSION_NOT_IMPLEMENTED")
