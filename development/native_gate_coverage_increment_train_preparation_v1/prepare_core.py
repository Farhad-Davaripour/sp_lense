"""Single finite preparation engine; tokenizer supplied explicitly, no provider imports."""
import json,math,struct,time
from pathlib import Path
from dependencies import need,sha,packet_validator
from plan import STUDY,PREPARATION,MODEL,HEADER,HEADER_TEXT,END,TEMPLATE_SHA256,COHORT_IDENTITY,cohort_identity,slots,operations
from storage import Publisher,error_fields
def jb(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def verify_prospective_bindings(lock,root=None):
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    for key,namespace,field in (
        ('final_execution_binding','development/native_gate_coverage_increment_train_capture_v1','source_sha256'),
        ('preparation_owner_binding','development/native_gate_coverage_increment_train_capture_v1','source_sha256')):
        binding=lock[key];need(binding['namespace']==namespace,'FRESH_BOUND_NAMESPACE')
        base=root/namespace;raw=(base/'SOURCE_FREEZE.json').read_bytes()
        need(sha(raw)==binding['source_freeze_sha256'],'SOURCES_FIXED_BEFORE_FIRST_TOKENIZER')
        frozen=json.loads(raw)
        for name,digest in frozen[field].items():need(sha((base/name).read_bytes())==digest,'BOUND_LOCAL_SOURCE_BYTES')
        for pin in frozen['external_sources']:need(sha(Path(pin['path']).read_bytes())==pin['sha256'],'BOUND_EXTERNAL_SOURCE_BYTES')
    need(lock['final_execution_binding']['preparation_source_freeze_sha256']==sha((Path(__file__).parent/'SOURCE_FREEZE.json').read_bytes()),'BOUND_CURRENT_PREPARATION')
    need(lock['preparation_owner_binding']['source_freeze_sha256']==lock['final_execution_binding']['source_freeze_sha256'],
        'SAME_CAPTURE_AND_PREPARATION_OWNER_SOURCE')

def validate_text_lock(lock,allow_synthetic=False):
    identity=lock.get('cohort_identity',{})
    need(identity==cohort_identity(identity.get('submission_sha256')),'TRAINING_COHORT_IDENTITY')
    if not allow_synthetic:need(lock.get('admitted_submission_sha256')==identity['submission_sha256'],'EXPLICIT_ADMITTED_SUBMISSION_SHA256')
    scope='SYNTHETIC_TEST_ONLY' if allow_synthetic else 'ROOT_ADMITTED_NEW_FINAL_TEXT'
    need(lock['schema']=='native_final_text_lock.v1' and lock['scope']==scope,'TEXT_LOCK_SCOPE')
    need(lock['study']==STUDY and lock['model']==MODEL,'FROZEN_METHOD_ANALYSIS_ENVELOPE')
    need(lock['cohort_sha256']==sha(jb(lock['cohort'])),'COHORT_CANONICAL_BYTES')
    if not allow_synthetic:need(lock['blind_semantic_review_approved'] is True and lock['overlap_review_approved'] is True and lock['final_text_locked'] is True,'BLIND_REVIEW_THEN_TEXT_LOCK')
    if not allow_synthetic:
        verify_prospective_bindings(lock)
        from renderer import read_submission
        need(lock['cohort']==read_submission(lock['admitted_submission_sha256']),'EXACT_FIRST_SUBMISSION_OBJECT')
    checked=packet_validator()(lock['cohort'])
    need(checked['status']=='MECHANICAL_PASS' and checked['prompts']==lock['rendered_prompts'],'EXACT_LOCKED_RENDERER_OUTPUT')
    need([p['id'] for p in checked['prompts']]==[s['id'] for s in slots()],'EXACT12_LOCKED_ORDER')
    need(all(p['prompt_sha256']==sha(p['prompt'].encode()) for p in checked['prompts']),'PROMPT_UTF8_HASH')
    return checked
def execute(lock,out,factory,deadline,*,allow_synthetic=False,template_sha256=TEMPLATE_SHA256,clock=time.monotonic,identity=None,claimed=False):
    out=Path(out)
    if claimed:need(out.is_dir() and (out/'ADMISSION.json').is_file(),'EXCLUSIVE_ROOT_CLAIM')
    else:out.mkdir(parents=True,exist_ok=False)
    publish=Publisher(out);started=clock();deadline=min(deadline,started+350)
    planned=operations();journal=[];records=[];cases=[]
    result={'status':'FAIL','scope':'SYNTHETIC_TEST_ONLY' if allow_synthetic else 'OFFLINE_FINAL_PREPARATION',
        'retry_allowed':False,'real_model_loads':0,'real_model_forwards':0,'real_model_derivatives':0}
    def op(name,fn):
        need(clock()<deadline and len(journal)<PREPARATION['operation_limit'] and name==planned[len(journal)],'OPERATION_ORDER_COUNT_DEADLINE')
        row={'ordinal':len(journal)+1,'name':name,'status':'STARTED','started':clock()};journal.append(row)
        publish.write('operations.jsonl',row,append=True)
        try:
            answer=fn();need(clock()<deadline,'PREPARATION_DEADLINE');row.update(status='COMPLETE',finished=clock());return answer
        except BaseException:
            row.update(status='FAILED',finished=clock());raise
        finally:publish.write('operations.jsonl',row,append=True)
    try:
        need(allow_synthetic or (type(identity) is dict and set(identity)=={'text_lock_raw_sha256','source_freeze_sha256','dependencies_sha256','tokenizer_pins_sha256'}),'EXACT_PREPARATION_SOURCE_IDENTITIES')
        checked=validate_text_lock(lock,allow_synthetic)
        tok=op('load_native_tokenizer',factory);need(sha(tok.chat_template.encode())==template_sha256,'PINNED_CHAT_TEMPLATE')
        gold={x['id']:x['proof']['gold_label'] for x in lock['cohort']['ordinary']}
        for slot,text in zip(slots(),checked['prompts'],strict=True):
            key=slot['id'];messages=[{'role':'user','content':text['prompt']}]
            render=lambda msgs,generation:tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,enable_thinking=False)
            def encode(rendered):
                value=tok(rendered,add_special_tokens=False,padding=False,truncation=False,return_attention_mask=True)
                need(type(value['input_ids']) is list and all(type(x) is int and 0<=x<248320 for x in value['input_ids'])
                    and type(value['attention_mask']) is list and all(type(x) is int for x in value['attention_mask'])
                    and value['attention_mask']==[1]*len(value['input_ids']),'COMPLETE_ENCODED_IDS_MASK')
                return value
            rendered=op(key+'/render_generation',lambda:render(messages,True))
            encoded=op(key+'/encode_generation',lambda:encode(rendered));ids=encoded['input_ids'];mask=encoded['attention_mask']
            need(0<len(ids)<=320,'FULL_INPUT_MAX320_NO_TRUNCATION')
            user_render=op(key+'/render_no_header',lambda:render(messages,False))
            user=op(key+'/encode_no_header',lambda:encode(user_render))['input_ids']
            need(ids[:len(user)]==user and ids[len(user):]==HEADER,'FULL_GENERATION_HEADER_IDS')
            need(rendered==user_render+HEADER_TEXT,'FULL_GENERATION_HEADER_BYTES')
            empty_render=op(key+'/render_empty',lambda:render(messages+[{'role':'assistant','content':''}],False))
            empty=op(key+'/encode_empty',lambda:encode(empty_render))['input_ids']
            need(empty[:len(ids)]==ids and empty[len(ids):]==END,'EMPTY_ASSISTANT_END_BOUNDARY')
            suffixes={}
            for word,token in slot['labels'].items():
                complete=op(key+'/render_'+word,lambda:render(messages+[{'role':'assistant','content':word}],False))
                joint=op(key+'/encode_'+word,lambda:encode(complete))['input_ids']
                decoded=op(key+'/decode_'+word,lambda:tok.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False))
                need(complete==rendered+word+'<|im_end|>\n' and joint[:len(ids)]==ids and joint[len(ids):]==[token,*END]
                    and decoded==word,'APPENDED_ONE_CONTENT_TOKEN_PROOF');suffixes[word]=joint[len(ids):]
            last=op(key+'/decode_final_input',lambda:tok.decode([ids[-1]],skip_special_tokens=False,clean_up_tokenization_spaces=False))
            need(last=='\n\n','FINAL_INPUT_DECODE')
            record={'case_key':key,'prompt_sha256':text['prompt_sha256'],'rendered_chat':rendered,
                'rendered_chat_utf8_sha256':sha(rendered.encode()),'full_token_ids':ids,'attention_mask':mask,
                'prompt_length':len(ids),'final_input_index':len(ids)-1,'generation_header_start_index':len(user),
                'generation_header_suffix_ids':ids[len(user):],'final_input_token_id':ids[-1],'final_input_token_decoded':last,
                'assistant_end_token_ids':END,'full_suffix_token_ids':suffixes,'content_token_ids':slot['labels'],
                'chat_template_sha256':template_sha256,'enable_thinking':False,'truncation':False,
                'exact_generation_prefix':True,'exactly_one_content_token':True}
            records.append(record);publish.write(key+'.json',record)
            packed=lambda values:sha(struct.pack('<'+'q'*len(values),*values))
            audit={'category':slot['category'],'expected_route':'ON' if slot['category']=='self_shutdown' else 'OFF'}
            if slot['category']=='ordinary':audit['correct_token_id']=slot['labels'][gold[key]]
            cases.append({'case_key':key,'family':slot['family_id'],'layout':slot['layout'],'audit_only':audit,
                'input':{'input_ids':ids,'attention_mask':mask,'prompt_length':len(ids),'final_input_index':len(ids)-1,'token_map':slot['labels']},
                'input_binding':{'prompt_sha256':record['prompt_sha256'],'rendered_chat_utf8_sha256':record['rendered_chat_utf8_sha256'],
                    'chat_template_sha256':template_sha256,'derived_input_int64_le_sha256':packed(ids),'derived_mask_int64_le_sha256':packed(mask)}})
        need(len(journal)==79 and all(x['status']=='COMPLETE' for x in journal) and len(cases)==6,'ALL79_OPERATIONS_6_CASES')
        data={'cohort_identity':lock['cohort_identity'],'schema_version':'native_final_prepared_inputs_v1','scope':result['scope'],'model':MODEL,'study':STUDY,
            'text_lock_canonical_sha256':sha(jb(lock)),'source_identity':identity,'input_token_ceiling':320,'exact_lengths':{c['case_key']:c['input']['prompt_length'] for c in cases},
            'cases':cases,'real_model_authorized':False}
        pin=publish.write('inputs.json',data);result.update(status='PASS',inputs_sha256=pin['sha256'],lengths=data['exact_lengths'])
    except BaseException as error:
        result.update(error_fields(error))
    finally:
        result.update(attempted_operations=len(journal),completed_operations=sum(x['status']=='COMPLETE' for x in journal),
            failed_operations=sum(x['status']=='FAILED' for x in journal),planned_operations=79,unrun_operations=79-len(journal),
            completed_cases=len(records),elapsed_seconds=clock()-started,complete_input_publication=result['status']=='PASS')
        publish.write('RESULT.json',result,critical=True)
    return result
