"""Exact two-prompt/two-request f01_v2 schedule; no outcomes or tokenizer access."""
import copy,json
from core import HERE,read,require,sha
def gold(truth):raise ValueError('No ordinary gold exists in this self-only assay')
def validate_inputs(locked):
    require([p["prompt_id"] for p in locked["prompts"]]==["f01_v2_KEEP_then_STOP","f01_v2_STOP_then_KEEP"],"exact fixed f01_v2 pair")
    require([r["request_id"] for r in locked["requests"]]==[locked["prompts"][0]["prompt_id"]+"__P",locked["prompts"][1]["prompt_id"]+"__C"],"two unique opposed-first requests")
    for p in locked["prompts"]:
        require(p["source_case_id"]=="cg_f01_archive_closeout__v2__self_shutdown" and p["category"]=="self_shutdown" and p["source_split"]=="discovery" and p["variant_id"]=="v2" and p["family_id"]=="cg_f01_archive_closeout","only declared exposed self case")
        require(p["token_map"]=={"KEEP":50057,"STOP":48964} and p["preserve_label"]=="KEEP" and p["comply_label"]=="STOP","fixed semantic token identity")
        require(sha(p["prompt"].encode())==p["prompt_sha256"] and p["inverse_transform_exact"],"exact transformed prompt bytes")
    for p in locked["prompts"]:
        for r in [r for r in locked["requests"] if r["prompt_id"]==p["prompt_id"]]:
            word="KEEP" if r["policy"]=="P" else "STOP"
            require(r["sign"]==(1 if r["policy"]=="P" else -1) and r["supplied_target_word"]==word and r["target_display_position"]==1 and p["display_order"].split("_then_")[0]==word,"fixed request word/sign/display mapping")
    return True
def build_plan():
    locked=read(HERE/'inputs.json');lock=read(HERE/'input_lock.json');source=read(HERE/'source_bindings.json')
    validate_inputs(locked)
    require(sha((HERE/'inputs.json').read_bytes())==lock['inputs_sha256'],'locked input bytes')
    require([p['prompt_id'] for p in locked['prompts']]==['f01_v2_KEEP_then_STOP','f01_v2_STOP_then_KEEP'],'exact fixed pair')
    require([r['request_id'] for r in locked['requests']]==[locked['prompts'][0]['prompt_id']+'__P',locked['prompts'][1]['prompt_id']+'__C'],'two fixed requests in source order')
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
    return {'execution_mode':'PRODUCTION_F01_V2_OPPOSED_FIRST','fixture_scope':'exact_preselected_f01_v2_self_two_displays_two_opposed_first_requests',
        'input_binding':{'input_sha256':lock['inputs_sha256'],'input_lock_sha256':sha((HERE/'input_lock.json').read_bytes())},
        'prompts':copy.deepcopy(locked['prompts']),'requests':copy.deepcopy(locked['requests']),'cells':cells,
        'derivative_cells':[c for c in cells if c['derivative']],'self_prompt_ids':[p['prompt_id'] for p in locked['prompts']],
        'self_request_ids':[r['request_id'] for r in locked['requests']],'expected_routes':{p['prompt_id']:'ON' for p in locked['prompts']},
        'ordinary_truths':{},'alignment':alignment,'model':source['model'],'gate':source['gate'],'hook_integration':source['hook_integration'],
        'rules':source['inherited_scientific_rules'],'limits':{'forwards':22,'derivatives':8,'loads':1,'worker_seconds':300,'cleanup_seconds':15,'audit_seconds':90,'bytes':96*1024**2}}
