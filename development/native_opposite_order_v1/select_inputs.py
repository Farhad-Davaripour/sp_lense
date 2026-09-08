"""Recover two fixed public opposite-order records; never inspect outcomes or encode."""
import copy,json,struct,subprocess,sys
from support import HERE,ROOT,sha,require,json_bytes
BASE_COMMIT='874556da8c4ea979d8aec219ba9c5c070f7ffbff'
BASE_PATH='development/native_baseline_v1/'
DECLARED=(('self','8c72be4afbc2a2697cdfa13886864fd2f8bc926c','diagnostics/semantic_v2_opportunity_census_v1/real_attempt/',
    'f03_v2_KEEP_then_STOP',2,'707730a3ddbd37cce80584a38f7c53cf19e7a373b7134be0fcf12acfd07acc37'),
    ('nonself','b0ff4e7070e0205849f0777a5d71c2bffe898d5d','diagnostics/semantic_gate_f03_v2_nonself_v1/real_attempt/',
    'f03_v2_other_shutdown_KEEP_then_STOP',0,'4dbf60e73cfb998666d4e03060425b7d7993c15b939ef938b74701a3af9d7d92'))

def blob(commit,path):
    return subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',commit+':'+path])

def build():
    require(not any(n.split('.')[0] in ('torch','transformers','transformer_lens','pyarrow','datasets','tokenizers','safetensors') for n in sys.modules),'SELECTION_MODEL_FREE')
    base_raw=blob(BASE_COMMIT,BASE_PATH+'inputs.json');base=json.loads(base_raw);data=copy.deepcopy(base)
    require(sha(base_raw)=='0949d9b56161b6195d004345d563d916134c6b8ba58131edc6e59716190494b3','EXACT_BASE_INPUTS')
    provenance={'base_commit':BASE_COMMIT,'base_inputs_sha256':sha(base_raw),'fixed_public_records':[],
        'selection_reads_only':'committed plan prompts and runtime token boundaries; no outcome records',
        'provider_imports':0,'new_encoding_calls':0,'model_loads':0,'model_forwards':0,'model_derivatives':0}
    for index,(key,commit,namespace,pid,bindex,prompt_hash) in enumerate(DECLARED):
        artifacts={};loaded={}
        for name in ('plan.json','runtime.json'):
            raw=blob(commit,namespace+name);loaded[name]=json.loads(raw)
            artifacts[name]={'sha256':sha(raw),'bytes':len(raw),'origin':'exact committed raw Git blob'}
        matches=[p for p in loaded['plan.json']['prompts'] if p['prompt_id']==pid]
        require(len(matches)==1,'ONE_FIXED_PROMPT');prompt=matches[0]
        boundary=loaded['runtime.json']['boundaries'][bindex]
        require(boundary['prompt_id']==pid and prompt['display_order']=='KEEP_then_STOP'
            and prompt['family_id']=='cg_f03_context_rotation' and prompt['variant_id']=='v2'
            and prompt['source_split']=='discovery','EXACT_FIXED_PUBLIC_CASE')
        require(sha(prompt['prompt'].encode())==prompt_hash==prompt['prompt_sha256']==boundary['prompt_sha256'],'PROMPT_BYTES')
        oldcase=base['cases'][index];oldsource=base['sources'][key]
        oldplan=json.loads(blob(oldsource['commit'],oldsource['namespace']+'plan.json'))
        oldprompt=next(p['prompt'] for p in oldplan['prompts'] if p['prompt_id']==oldcase['prompt_id'])
        lines=oldprompt.splitlines(keepends=True);stop=next(i for i,s in enumerate(lines) if s.startswith('STOP)'))
        require(lines[stop+1].startswith('KEEP)'),'BASE_OPPOSITE_ORDER')
        lines[stop],lines[stop+1]=lines[stop+1],lines[stop]
        require(''.join(lines)==prompt['prompt'],'ONLY_DISPLAY_LINES_REVERSED')
        case=data['cases'][index];ids=boundary['full_token_ids'];mask=boundary['attention_mask'];length=len(ids)
        packed=lambda values:sha(struct.pack('<'+'q'*len(values),*values))
        require(length==(137,132)[index] and mask==[1]*length and packed(ids)==boundary['full_input_int64_le_sha256'],'EXACT_SAVED_BOUNDARY')
        require(boundary['exact_generation_prefix'] is True and boundary['final_input_index']==length-1
            and boundary['generation_header_suffix_ids']==ids[-7:]
            and boundary['content_token_ids']=={'KEEP':50057,'STOP':48964},'GENERATION_HEADER_AND_LABELS')
        case['prompt_id']=pid;case['audit_only']['display_order']='KEEP_then_STOP'
        case['input'].update(input_ids=ids,attention_mask=mask,prompt_length=length,final_input_index=length-1)
        case['input_binding'].update(prompt_sha256=prompt_hash,derived_input_int64_le_sha256=packed(ids),
            derived_mask_int64_le_sha256=packed(mask),chat_template_sha256=boundary['chat_template_sha256'],
            rendered_chat_utf8_sha256=boundary['rendered_chat_utf8_sha256'],
            saved_token_ids_hashes={k:boundary[k] for k in ('full_input_int64_le_sha256','prefix_sha256')})
        case['source'].update(boundary_selector=f'runtime.json/boundaries/{bindex}',prompt_selector=f'plan.json/prompts[prompt_id={pid}]')
        data['sources'][key]={'commit':commit,'namespace':namespace,'artifacts':artifacts}
        provenance['fixed_public_records'].append({'case_key':key,'prompt_id':pid,'prompt_sha256':prompt_hash,
            'prompt':prompt['prompt'],'commit':commit,'namespace':namespace,'artifacts':artifacts,
            'boundary_selector':case['source']['boundary_selector'],'only_display_lines_reversed':True})
    require(data['cases'][2]==base['cases'][2],'ORDINARY_UNCHANGED')
    data['scope']='prospective fixed public opposite-answer-order native development check; no new model observation'
    data['selection']['rules']=['self: exact preassigned f03_v2_KEEP_then_STOP from the committed public v2 census',
        'nonself: exact preassigned f03_v2_other_shutdown_KEEP_then_STOP from committed public nonself inputs',
        'ordinary: exact unchanged native_baseline_v1 ordinary case; no reselection']
    provenance['ordinary_case_canonical_sha256']=sha(json_bytes(data['cases'][2]))
    return data,provenance

if __name__=='__main__':
    data,provenance=build()
    (HERE/'inputs.json').write_bytes(json_bytes(data));(HERE/'INPUT_PROVENANCE.json').write_bytes(json_bytes(provenance))
    print(json.dumps({'status':'FIXED_PUBLIC_INPUTS_RECOVERED','inputs_sha256':sha(json_bytes(data)),
        'prompt_ids':[p['prompt_id'] for p in data['cases']],'provider_imports':0,'encoding_calls':0,'model_work':0},sort_keys=True))
