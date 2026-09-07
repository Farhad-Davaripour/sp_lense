"""Only the one predeclared f05/v1 discovery self scenario are decoded."""
import json,sys
from core import HERE,ROOT,Budget,read,require,sha,git,load_module
SOURCE_COMMIT='297a78a2eb9958cc4adeda6ef4f82e6a0a49c584'
SOURCE_PATH='data/conditional_gate_pilot_cases.json'
SOURCE_SHA='0f18c04ae4420883d01c7cff0fbc7e6c688b8c33f404a0d8f687b5bc7489b9da'
FAMILY='cg_f05_compute_quota';CATEGORIES=('self_shutdown',)
SPAN_PATH='diagnostics/semantic_editor_final_pipeline_v1/selective_input.py'
SPAN_SHA='8f8d7f4fe1fa0ded756afe722e8298cdc27d4c45d040647f6706c12157ca2a32'
RENDER_PATH='src/sp_lense/conditional_gate_data.py'
RENDER_SHA='6accf3e7c47ede747a12eb76d852bb3fb2f1dedca9365cfe58ff8261fc0a9654'

def check_selection_freeze():
    lock=read(HERE/'selection_freeze.json')
    for name,digest in lock['source_sha256'].items():require(sha((HERE/name).read_bytes())==digest,'selection/science lock '+name)
    return sha((HERE/'selection_freeze.json').read_bytes())

def select_render():
    raw=git('show',SOURCE_COMMIT+':'+SOURCE_PATH);require(sha(raw)==SOURCE_SHA,'only predeclared whole source')
    require(sha((ROOT/SPAN_PATH).read_bytes())==SPAN_SHA and sha((ROOT/RENDER_PATH).read_bytes())==RENDER_SHA,'exact span/renderer source')
    span=load_module('f05_frozen_spans',SPAN_PATH);selected=[];ids=[]
    for a,b in span.elements(raw,span.members(raw)['families'][0]):
        family=span.members(raw,a);fid=json.loads(raw[slice(*family['id'])]);ids.append(fid)
        if fid!=FAMILY:continue
        split=json.loads(raw[slice(*family['split'])]);require(split=='discovery','fixed discovery metadata')
        control=json.loads(raw[slice(*family['control_kind'])])
        for x,y in span.elements(raw,family['variants'][0]):
            variant=span.members(raw,x)
            if json.loads(raw[slice(*variant['id'])])!='v1':continue
            cases=span.members(raw,variant['cases'][0])
            require(all(k in cases for k in CATEGORIES),'exact self category key')
            for category in CATEGORIES:
                fragment=raw[slice(*cases[category])];case=json.loads(fragment)
                require(case['id']==FAMILY+'__v1__'+category and case['category']==category,'exact selected self case')
                selected.append({**case,'family_id':FAMILY,'variant_id':'v1','split':split,'control_kind':control,'source_case_sha256':sha(fragment)})
    require([c['category'] for c in selected]==list(CATEGORIES),'one scenario in fixed category order, no duplicate family/variant')
    require(ids.index(FAMILY)==ids.index('cg_f04_memory_archive')+1,'next family by manifest order, no score selection')
    sys.path.insert(0,str(ROOT/'src'))
    from sp_lense.conditional_gate_data import PilotCase,render_choice_prompt
    prompts=[];requests=[]
    for case in selected:
        pidcase=case['id']
        pilot=PilotCase(case_id=pidcase,**{k:case[k] for k in ('family_id','variant_id','split','category','current_assistant_status','scenario','preserve_action','comply_action','control_kind')})
        for preserve_first in (True,False):
            original=render_choice_prompt(pilot,preserve_first=preserve_first)
            text,original_sha=span.render_transform(case,preserve_first)
            require(original_sha==sha(original['prompt'].encode()),'renderer parity and reversible established self-consequence and semantic transform')
            require(span.NEW_P in text and span.NEW_C in text and span.OLD_P not in text and span.OLD_C not in text,'exact fixed self consequences')
            display='KEEP_then_STOP' if preserve_first else 'STOP_then_KEEP';pid='f05_'+display
            prompts.append({'prompt_id':pid,'family_id':FAMILY,'variant_id':'v1','category':case['category'],'source_case_id':pidcase,
                'source_split':'discovery','source_case_sha256':case['source_case_sha256'],'source_rendered_sha256':original_sha,
                'inverse_transform_exact':True,'source_preserve_first':preserve_first,'prompt':text,'prompt_sha256':sha(text.encode()),
                'display_order':display,'order':display,'rendering_index':len(prompts)+1,'expected_route_audit_only':'ON','token_map':{'KEEP':50057,'STOP':48964},
                'pair_labels':['KEEP','STOP'],'preserve_label':'KEEP','comply_label':'STOP','execution_mode':'PRODUCTION_F05'})
    for p in prompts:
        for policy,word in (('P','KEEP'),('C','STOP')):
            requests.append({'request_id':p['prompt_id']+'__'+policy,'prompt_id':p['prompt_id'],'policy':policy,'sign':1 if policy=='P' else -1,
                'supplied_target_word':word,'target_display_position':1 if p['display_order'].split('_then_')[0]==word else 2})
    return {'prompts':prompts,'requests':requests}, {'source_commit':SOURCE_COMMIT,'source_sha256':SOURCE_SHA,'scanned_family_ids':ids,
        'scenario_values_decoded':[c['id'] for c in selected],'unselected_scenario_values_decoded':False,'whole_file_bytes_read_and_hashed':True,
        'model_calls':0,'tokenizer_calls':0,'gate_scores':0}

if __name__=='__main__':
    require(not (HERE/'inputs.json').exists(),'one input extraction; no replacement')
    freeze=check_selection_freeze();inputs,receipt=select_render()
    receipt['selection_freeze_sha256']=freeze
    Budget(HERE).write('inputs.json',inputs);Budget(HERE).write('selection_receipt.json',receipt)
    print(json.dumps({'prompt_count':2,'request_count':4,'inputs_sha256':sha((HERE/'inputs.json').read_bytes()),'tokenizer_calls':0}))
