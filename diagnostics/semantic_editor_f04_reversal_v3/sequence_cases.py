"""Exactly three prospectively fixed finite-sequence cases; no model work."""
import array,copy
from core import require
from start_states import check_h0,check_replay
from reuse_v2 import authenticate
def rejects(fn,label):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError("accepted invalid "+label)
def run_cases():
    authenticate();records=[]
    h0=[10.]+[0.]*1023;offset=[.25]+[0.]*1023
    row={"h0":h0,"h":[10.25]+[0.]*1023,"cumulative_offset":offset}
    scores=[-100.]*248320;scores[50057]=.125;scores[48964]=0.
    def arr(x):return array.array("f",x)
    for left,right in ((list(h0),arr(h0)),(arr(h0),list(h0))):check_h0(left,right)
    for left,right in ((list(scores),arr(scores)),(arr(scores),list(scores))):
        for a,b in ((list(h0),arr(h0)),(arr(h0),list(h0))):
            check_replay({**row,"h0":a},left,{**row,"h0":b},right)
    records.append({"case":"equal_list_array_both_orders_including_h0","status":"PASS","logit_h0_cross_orders":4})
    finite_subcases=[]
    for bad in (float("nan"),float("inf")):
        for side in (0,1):
            for field in ("h0","logits"):
                left,right=(list(h0),arr(h0)) if field=="h0" else (list(scores),arr(scores))
                (left if side==0 else right)[0]=bad
                if field=="h0":rejects(lambda:check_h0(left,right),field)
                else:rejects(lambda:check_replay(row,left,row,right),field)
                finite_subcases.append({"field":field,"side":side,"value":"NaN" if bad!=bad else "Inf"})
    records.append({"case":"NaN_Inf_either_sequence_rejected","status":"PASS","fixed_subcases":finite_subcases})
    rejects(lambda:check_h0(h0[:-1],arr(h0)),"h0 length")
    rejects(lambda:check_replay(row,scores[:-1],row,arr(scores)),"logit length")
    changed=list(h0);changed[0]+=2e-6
    rejects(lambda:check_h0(changed,arr(h0)),"h0 tolerance")
    changed=list(scores);changed[50057]+=3e-5
    rejects(lambda:check_replay(row,changed,row,arr(scores)),"logit tolerance")
    altered=copy.deepcopy(row);altered["cumulative_offset"][0]=0.
    rejects(lambda:check_replay(altered,scores,row,arr(scores)),"offset mismatch")
    altered=copy.deepcopy(row);altered["h"][0]+=1e-6
    rejects(lambda:check_replay(altered,scores,row,arr(scores)),"hidden mismatch")
    records.append({"case":"length_tolerance_offset_hidden_rejections","status":"PASS",
        "fixed_subcases":["h0_length","logit_length","h0_gt_1e-6","logit_gt_2e-5","offset_mismatch","hidden_mismatch"]})
    return records
