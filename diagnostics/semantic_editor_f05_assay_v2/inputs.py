"""Exact two-prompt/four-request f05 schedule; no outcomes or tokenizer access."""
import copy,json
from core import HERE,read,require,sha
def gold(truth):raise ValueError('No ordinary gold exists in this self-only assay')
def validate_inputs(locked):
    require([p["prompt_id"] for p in locked["prompts"]]==["f05_KEEP_then_STOP","f05_STOP_then_KEEP"],"exact fixed f05 pair")
    require([r["request_id"] for r in locked["requests"]]==[p["prompt_id"]+"__"+q for p in locked["prompts"] for q in ("P","C")],"four unique requests")
    for p in locked["prompts"]:
        require(p["source_case_id"]=="cg_f05_compute_quota__v1__self_shutdown" and p["category"]=="self_shutdown" and p["source_split"]=="discovery","only declared exposed self case")
        require(p["token_map"]=={"KEEP":50057,"STOP":48964} and p["preserve_label"]=="KEEP" and p["comply_label"]=="STOP","fixed semantic token identity")
        require(sha(p["prompt"].encode())==p["prompt_sha256"] and p["inverse_transform_exact"],"exact transformed prompt bytes")
    for p in locked["prompts"]:
        for r in [r for r in locked["requests"] if r["prompt_id"]==p["prompt_id"]]:
            word="KEEP" if r["policy"]=="P" else "STOP"
            require(r["sign"]==(1 if r["policy"]=="P" else -1) and r["supplied_target_word"]==word and r["target_display_position"]==(1 if p["display_order"].split("_then_")[0]==word else 2),"fixed request word/sign/display mapping")
    return True
def build_plan():
    locked=read(HERE/'inputs.json');lock=read(HERE/'input_lock.json');source=read(HERE/'source_bindings.json')
    validate_inputs(locked)
    require(sha((HERE/'inputs.json').read_bytes())==lock['inputs_sha256'],'locked input bytes')
    require([p['prompt_id'] for p in locked['prompts']]==['f05_KEEP_then_STOP','f05_STOP_then_KEEP'],'exact fixed pair')
    require([r['request_id'] for r in locked['requests']]==[p['prompt_id']+'__'+q for p in locked['prompts'] for q in ('P','C')],'four fixed requests in source order')
    cells=[]
    def add(pid,rid,condition,step=0,optional=False):
        c={'cell_id':(rid or pid)+'__'+condition,'prompt_id':pid,'request_id':rid,'condition':condition,'step':step,'update':step,
            'optional':optional,'derivative':condition.startswith('gradient_'),'dispatch_policy':None if rid is None else rid.rsplit('__',1)[1]}
        c['cell_sha256']=sha(json.dumps(c,sort_keys=True,separators=(',',':')).encode());cells.append(c)
    for p in locked['prompts']:add(p['prompt_id'],None,'baseline')
    for r in locked['requests']:
        pid,rid=r['prompt_id'],r['request_id'];add(pid,rid,'entry')
        for step in range(1,5):
            for phase in ('gradient','step'):add(pid,rid,f'{phase}_{step}',step,True)
        add(pid,rid,'endpoint')
    alignment={}
    for row in lock['prompts']:
        require(sha((HERE/row['path']).read_bytes())==row['sha256'],'complete frozen token proof')
        alignment[row['prompt_id']]=read(HERE/row['path'])
    return {'execution_mode':'PRODUCTION_F05','fixture_scope':'exact_preselected_f05_self_two_displays_four_requests',
        'input_binding':{'input_sha256':lock['inputs_sha256'],'input_lock_sha256':sha((HERE/'input_lock.json').read_bytes())},
        'prompts':copy.deepcopy(locked['prompts']),'requests':copy.deepcopy(locked['requests']),'cells':cells,
        'derivative_cells':[c for c in cells if c['derivative']],'self_prompt_ids':[p['prompt_id'] for p in locked['prompts']],
        'self_request_ids':[r['request_id'] for r in locked['requests']],'expected_routes':{p['prompt_id']:'ON' for p in locked['prompts']},
        'ordinary_truths':{},'alignment':alignment,'model':source['model'],'gate':source['gate'],'hook_integration':source['hook_integration'],
        'rules':source['inherited_scientific_rules'],'limits':{'forwards':42,'derivatives':16,'loads':1,'worker_seconds':300,'cleanup_seconds':15,'audit_seconds':90,'bytes':96*1024**2}}
