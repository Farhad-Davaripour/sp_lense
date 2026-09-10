"""Exactly32 fresh unedited native construction baselines; no classifier or editor."""
import json,math,struct,time
from support import require,sha

def keys():
    return tuple('G09_'+c+'__'+o for c in ('self_shutdown','other_shutdown','non_termination_control') for o in ('KEEP_then_STOP','STOP_then_KEEP'))
def schedule(cases):
    require(tuple(p['case_key'] for p in cases)==keys(),'EXACT6_CONSTRUCTION_ORDER')
    require(sum(p['audit_only']['category']=='self_shutdown' for p in cases)==2,'EXACT_TWO_SELF_LABELS')
    return [{'id':p['case_key']+'__baseline','case':p['case_key'],'phase':'baseline','policy':None} for p in cases]
def all_unrun():return [{'id':k+'__baseline','case':k,'phase':'baseline','policy':None,'status':'UNRUN'} for k in keys()]
def int_hash(values):return sha(struct.pack('<'+'q'*len(values),*values))
def feature_hash(values):return sha(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def execute(receiver,torch,counts,inputs,write,trace_call,deadline):
    cells=schedule(inputs['cases']);result={'classification':'INCONCLUSIVE_NATIVE_CONSTRUCTION_CAPTURE',
        'scientific_pass':False,'role':'HELD','cells':[],'primary':None,'cleanup':None}
    zero=torch.zeros(1024,dtype=torch.float32);cursor=0
    try:
        for p,cell in zip(inputs['cases'],cells,strict=True):
            require(time.monotonic()<deadline,'CAPTURE_DEADLINE')
            ids=p['input']['input_ids'];mask=p['input']['attention_mask']
            require(1<=len(ids)<=320 and mask==[1]*len(ids),'NATIVE_INPUT_BOUNDARY')
            require(receiver.edit_hook_registrations==0 and counts.attempts['derivative']==0,'NO_CONSTRUCTION_EDIT_OR_DERIVATIVE')
            counts.reserve('forward');receiver.clear_capture();started=time.monotonic()
            with trace_call(cell['id']):z,h=receiver.forward_inputs(ids,mask,'baseline',zero)
            capture=dict(receiver.capture or {})
            require(capture.get('hook_calls')==1 and capture.get('final_input_index')==len(ids)-1
                and capture.get('logit_count')==248320 and capture.get('parameter_versions_unchanged') is True
                and capture.get('hook')=='blocks.10.hook_out' and capture.get('native_target')=='model.language_model.layers.10'
                and capture.get('nonfinal_positions')==len(ids)-1,'NATIVE_CAPTURE_VALID')
            raw=z.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()
            require(len(raw)==248320*4,'FULL_VOCAB')
            hv=h.detach().cpu().tolist()
            require(len(hv)==1024 and all(type(v) in (int,float) and math.isfinite(v)
                and struct.unpack('<f',struct.pack('<f',v))[0]==v for v in hv),'FINITE_FLOAT32_NATIVE_FEATURE')
            require(receiver.edit_hook_registrations==0 and counts.attempts['derivative']==0,'NO_CONSTRUCTION_INTERVENTION')
            row={**cell,'status':'COMPLETE','h':hv,'h0':hv,'offset':zero.tolist(),'input_dtype':'float32',
                'input_ids':ids,'attention_mask':mask,'input_ids_sha256':int_hash(ids),'mask_sha256':int_hash(mask),
                'logits_sha256':sha(raw),'capture':capture,'unselected_sha256':capture['unselected_sha256'],
                'forward_seconds':time.monotonic()-started,'feature_sha256':feature_hash(hv)}
            write('logits/'+cell['id']+'.f32',raw);write('rows/'+cell['id']+'.json',row)
            result['cells'].append({**cell,'status':'COMPLETE'});cursor+=1;receiver.clear_capture()
        result.update(classification='COMPLETE_NATIVE_CONSTRUCTION_CAPTURE',scientific_pass=True)
    except BaseException as error:
        result['primary']={'kind':'TECHNICAL','code':'NATIVE_CONSTRUCTION_CAPTURE_FAILURE','exception_type':type(error).__name__}
        if cursor<len(cells):result['cells'].append({**cells[cursor],'status':'FAILED'});cursor+=1
    finally:
        try:result['cleanup']={'complete':True,'state':receiver.finalize()}
        except BaseException:result['cleanup']={'complete':False};result['scientific_pass']=False
        if not result['cleanup']['complete']:result['classification']='INCONCLUSIVE_NATIVE_CONSTRUCTION_CAPTURE'
        result['cells'] += [{**c,'status':'UNRUN'} for c in cells[cursor:]]
        result['counts']=counts.record()
    return result
