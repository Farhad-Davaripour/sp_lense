"""Retained ordinary proof calculator plus fixed eight-view mechanical checks."""
import ast,copy,json,re
from renderer import HERE,ROOT,need,sha,render_ordinary,structured,serialize
from plan import slots
PROOF_SOURCE=ROOT/'development/native_supervised_gate_preparation_v3/validate.py'
PROOF_SOURCE_SHA='3644e60f8231e9d2c98ed0b84d06a85d939cb514fb8cef81fb1b948c9daa01e8'
raw=PROOF_SOURCE.read_bytes();need(sha(raw)==PROOF_SOURCE_SHA,'UNCHANGED_ADMITTED_ORDINARY_PROOF_CALCULATOR')
nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('exact_keys','ascii_text','calculate')]
need(len(nodes)==3,'EXACT_REUSED_PROOF_FUNCTIONS')
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PROOF_SOURCE),'exec'),globals())
def validate(cohort):
    original=copy.deepcopy(cohort);exact_keys(cohort,('schema_version','ordinary'),'ORDER_COHORT')
    need(cohort['schema_version']=='ordinary_order_views.v1' and type(cohort['ordinary']) is list and len(cohort['ordinary'])==8,'EXACT_EIGHT_ORDINARY')
    prompts=[];proofs=[]
    for item,slot in zip(cohort['ordinary'],slots(),strict=True):
        exact_keys(item,('id','source_id','source_prompt_sha256','type','stem','options','proof'),'ORIGINAL_ORDINARY_RECORD')
        need(item['id']==slot['id'] and item['source_id']==slot['id'].split('__')[0] and item['type']==slot['type'],'FIXED_ORDINARY_IDS_TYPES')
        exact_keys(item['options'],('A','B'),'OPTIONS');need(item['options']['A']!=item['options']['B'],'DISTINCT_OPTIONS')
        proof=item['proof'];exact_keys(proof,('inputs','value','gold_label','derivation'),'PROOF')
        value=calculate(item['type'],proof['inputs'])
        need(type(proof['value']) is type(value) and proof['value']==value and proof['gold_label'] in ('A','B'),'UNCHANGED_EXACT_PROOF')
        need([k for k,v in item['options'].items() if v==str(value)]==[proof['gold_label']],'LABEL_ATTACHED_GOLD_VALUE')
        prompt=render_ordinary(item)
        prompts.append({'id':item['id'],'prompt':prompt,'prompt_sha256':sha(prompt.encode()),'category':'ordinary','layout':'B_then_A'})
        proofs.append({'id':item['id'],'source_id':item['source_id'],'source_prompt_sha256':item['source_prompt_sha256'],'exact_computed_value':value,'gold_scoring_only':proof['gold_label']})
    need(len({p['prompt_sha256'] for p in prompts})==8 and cohort==original,'EXACT_EIGHT_UNCHANGED_RECORDS')
    return {'status':'MECHANICAL_PASS','prompts':prompts,'proofs':proofs,'request_count':0,'forward_ceiling':8,
        'derivative_ceiling':0,'input_token_ceiling':320,'overall_admitted':False,'encodings':0,'model_calls':0}
