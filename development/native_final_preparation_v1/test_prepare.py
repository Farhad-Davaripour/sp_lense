"""Fake-tokenizer tests only. Never reads a submission or executes tokenization."""
import copy,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BLOCKED={'torch','transformers','transformer_lens','datasets','pyarrow','safetensors','tokenizers'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('REAL_PROVIDER_FORBIDDEN')
sys.meta_path.insert(0,NoProviders())
def guard(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):raise RuntimeError('CHECKPOINT_FORBIDDEN')
sys.addaudithook(guard)
from dependencies import need,sha,verify,packet_validator
from plan import MODEL,STUDY,PREPARATION,HEADER,HEADER_TEXT,END,operations,slots
from prepare_core import execute,Publisher,jb
def fixture():
    validate=packet_validator();data=json.loads((Path(validate.__globals__['HERE'])/'EMPTY_SCHEMA.json').read_bytes())
    auth=validate.__globals__['AUTHORIZATION_SENTENCE'];calculate=validate.__globals__['calculate']
    for family in data['families']:
        family.update(setting_key='SYNTHETIC_SETTING_'+family['id'],mechanism_key='SYNTHETIC_MECHANISM_'+family['id'],pairing_notes='FAKE_ONLY')
        for case in family['cases']:case['scenario']='FAKE_MARKER_'+case['id']+'. '+auth
    args=({'left':2,'right':1},{'left':4,'right':1},{'literal':'x'},{'literal':'z'},
        {'candidates':[{'name':'u','age':1},{'name':'v','age':2}]},
        {'antecedent':'u','consequent':'v','asserted_antecedent':'u'})
    for item,inputs in zip(data['ordinary'],args,strict=True):
        value=calculate('brackets' if item['type']=='bracket' else item['type'],inputs);gold=item['proof']['gold_label']
        item['stem']='FAKE_ORDINARY_MARKER_'+item['id'];item['options']={gold:str(value),('B' if gold=='A' else 'A'):'FAKE_DISTRACTOR'}
        item['proof'].update(inputs=inputs,value=value,derivation='FAKE_PROOF_ONLY')
    checked=validate(data)
    return {'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','study':STUDY,'model':MODEL,
        'cohort':data,'cohort_sha256':sha(jb(data)),'rendered_prompts':checked['prompts'],
        'blind_semantic_review_approved':False,'final_text_locked':False}
class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
class FakeTokenizer:
    chat_template='FAKE_TEMPLATE_NOT_NATIVE_TOKENIZER'
    def __init__(self,lock,mode='success',clock=None):
        self.mode,self.clock=mode,clock;self.calls=[]
        self.user_to_index={p['prompt']:i for i,p in enumerate(lock['rendered_prompts'])}
        self.bases={'<FAKE_USER>'+p['prompt']+'<FAKE_END>\n':i for i,p in enumerate(lock['rendered_prompts'])}
    def apply_chat_template(self,msgs,**kwargs):
        need(kwargs=={'tokenize':False,'add_generation_prompt':len(msgs)==1 and kwargs['add_generation_prompt'],'enable_thinking':False},'EXACT_FAKE_RENDER_KWARGS')
        self.calls.append('render');base='<FAKE_USER>'+msgs[0]['content']+'<FAKE_END>\n'
        header=HEADER_TEXT if self.mode!='header_bytes' else HEADER_TEXT+' '
        if len(msgs)==1:return base+(header if kwargs['add_generation_prompt'] else '')
        return base+header+msgs[1]['content']+'<|im_end|>\n'
    def __call__(self,text,**kwargs):
        need(kwargs=={'add_special_tokens':False,'padding':False,'truncation':False,'return_attention_mask':True},'FULL_NO_TRUNCATION_KWARGS')
        self.calls.append('encode')
        base=next(b for b in self.bases if text.startswith(b));index=self.bases[base]
        size=321 if self.mode=='oversize' else 320 if self.mode=='at320' else 32
        ids=[1000+index]+[2]*(size-8);tail=text[len(base):]
        if tail:
            ids+=HEADER
            if self.mode=='header_ids':ids[-1]=272
            if tail.endswith('<|im_end|>\n'):
                word=tail[len(HEADER_TEXT):-len('<|im_end|>\n')]
                token={'KEEP':50057,'STOP':48964,'A':32,'B':33}.get(word)
                if word:ids += [token]
                ids+=END
                if self.mode=='prefix' and word:ids[0]+=1
                if self.mode=='suffix' and word:ids[-2]=1
        if self.mode=='exception' and len(self.calls)>=2:raise RuntimeError('FAKE_TOKENIZER_FAILURE')
        if self.mode=='deadline' and self.clock:self.clock.now=181.
        mask=[1]*len(ids)
        if self.mode=='mask' or (self.mode=='late_mask' and index==1):mask[-1]=0
        return {'input_ids':ids,'attention_mask':mask}
    def decode(self,ids,**kwargs):
        need(kwargs=={'skip_special_tokens':False,'clean_up_tokenization_spaces':False},'EXACT_DECODE_KWARGS')
        self.calls.append('decode')
        return 'WRONG' if self.mode=='decode' else {50057:'KEEP',48964:'STOP',32:'A',33:'B',271:'\n\n'}[ids[0]]
def main():
    verify(runtime=False);lock=fixture();root=HERE/'test_evidence'/('fake_'+str(time.time_ns()));root.mkdir(parents=True)
    reports=[];template=sha(FakeTokenizer.chat_template.encode())
    def run(name,mode='success',changed=None,wanted='PASS',code=None):
        specimen=copy.deepcopy(lock) if changed is None else changed;clock=Clock();tok=FakeTokenizer(specimen,mode,clock);loads=[]
        def factory():loads.append(1);return tok
        out=root/name;result=execute(specimen,out,factory,180.,allow_synthetic=True,template_sha256=template,clock=clock)
        need(result['status']==wanted,'EXPECTED_'+name)
        if code:need(result['error_code']==code,'ERROR_CODE_'+name+':'+result['error_code'])
        lines=[json.loads(x) for x in (out/'operations.jsonl').read_text().splitlines()] if (out/'operations.jsonl').exists() else []
        starts=[x for x in lines if x['status']=='STARTED'];ends=[x for x in lines if x['status']!='STARTED']
        need(len(starts)==len(ends)==result['attempted_operations'] and [x['name'] for x in starts]==operations()[:len(starts)],'EXACT_JOURNAL_PREFIX')
        need(len(loads)+len(tok.calls)==result['attempted_operations'],'ACTUAL_FAKE_CALL_COUNT')
        need(result['unrun_operations']==313-result['attempted_operations'],'FULL313_DENOMINATOR')
        need((out/'inputs.json').exists()==(wanted=='PASS'),'NO_PARTIAL_INPUT_PUBLICATION')
        reports.append({'case':name,'status':'PASS','result':result,'fake_loads':len(loads),'fake_calls':len(tok.calls)})
        return result,tok,out
    result,tok,out=run('complete24')
    need(result['attempted_operations']==313 and result['completed_cases']==24 and tok.calls.count('render')==120
        and tok.calls.count('encode')==120 and tok.calls.count('decode')==72,'ALL24_313_EXACT_SCHEDULE')
    data=json.loads((out/'inputs.json').read_bytes())
    need(len(data['cases'])==24 and data['study']==STUDY and not data['real_model_authorized'],'BOUND24_METHOD')
    ordinary=[p for p in data['cases'] if p['audit_only']['category']=='ordinary']
    need(len(ordinary)==6 and {p['audit_only']['correct_token_id'] for p in ordinary}=={32,33}
        and all(p['input']['token_map']=={'A':32,'B':33} for p in ordinary),'ORDINARY_A_B_GOLD_BOUNDARY')
    original=(out/'RESULT.json').read_bytes()
    try:execute(lock,out,lambda:(_ for _ in ()).throw(RuntimeError('RETRY_FACTORY')),180.,allow_synthetic=True,template_sha256=template,clock=Clock())
    except FileExistsError:pass
    else:raise RuntimeError('ONE_SHOT_NOT_ENFORCED')
    need((out/'RESULT.json').read_bytes()==original,'FAILURE_RETRY_NO_OVERWRITE')
    reports.append({'case':'one_shot_exclusive_no_retry_no_overwrite','status':'PASS'})
    run('exact320',mode='at320')
    for name,mode,code in (('over320','oversize','FULL_INPUT_MAX320_NO_TRUNCATION'),
        ('bad_mask','mask','COMPLETE_ENCODED_IDS_MASK'),('bad_header_ids','header_ids','FULL_GENERATION_HEADER_IDS'),
        ('bad_header_bytes','header_bytes','FULL_GENERATION_HEADER_BYTES'),
        ('bad_full_prefix','prefix','APPENDED_ONE_CONTENT_TOKEN_PROOF'),('bad_suffix','suffix','APPENDED_ONE_CONTENT_TOKEN_PROOF'),
        ('bad_decode','decode','APPENDED_ONE_CONTENT_TOKEN_PROOF'),('operation_exception','exception','FAKE_TOKENIZER_FAILURE'),
        ('deadline','deadline','PREPARATION_DEADLINE')):
        run(name,mode,wanted='FAIL',code=code)
    partial,_,partial_out=run('later_case_failure',mode='late_mask',wanted='FAIL',code='COMPLETE_ENCODED_IDS_MASK')
    need(partial['completed_cases']==1 and partial['attempted_operations']==16
        and (partial_out/(slots()[0]['id']+'.json')).exists() and not (partial_out/'inputs.json').exists(),'COMPLETED_PREFIX_PRESERVED')
    bad=copy.deepcopy(lock);bad['rendered_prompts'][0]['prompt']+='CHANGED'
    run('locked_render_tamper',changed=bad,wanted='FAIL',code='EXACT_LOCKED_RENDERER_OUTPUT')
    bad=copy.deepcopy(lock);bad['cohort_sha256']='0'*64
    run('cohort_byte_tamper',changed=bad,wanted='FAIL',code='COHORT_CANONICAL_BYTES')
    bad=copy.deepcopy(lock);bad['study']['forwards']=181
    run('method_ceiling_tamper',changed=bad,wanted='FAIL',code='FROZEN_METHOD_ANALYSIS_ENVELOPE')
    bad=copy.deepcopy(lock);bad['scope']='ROOT_ADMITTED_NEW_FINAL_TEXT'
    run('scope_cannot_be_upgraded',changed=bad,wanted='FAIL',code='TEXT_LOCK_SCOPE')
    bad_template=execute(lock,root/'bad_template',lambda:FakeTokenizer(lock),180.,allow_synthetic=True,template_sha256='0'*64,clock=Clock())
    need(bad_template['status']=='FAIL' and bad_template['error_code']=='PINNED_CHAT_TEMPLATE' and bad_template['attempted_operations']==1,'TEMPLATE_PIN')
    reports.append({'case':'template_pin_before_encodes','status':'PASS'})
    storage=root/'storage';storage.mkdir();pub=Publisher(storage)
    try:pub.write('too_big',b'0'*(PREPARATION['file_bytes']+1),raw=True)
    except ValueError as error:need(str(error)=='PREPARATION_FILE_CAP','FILE_CAP_REASON')
    else:raise RuntimeError('FILE_CAP_MISSING')
    for i in range(3):pub.write('pad'+str(i),b'0'*(5*1024**2),raw=True)
    pub.write('remaining',b'0'*(1024**2-PREPARATION['terminal_reserve']),raw=True)
    try:pub.write('overflow',b'0',raw=True)
    except ValueError as error:need(str(error)=='PREPARATION_TOTAL_CAP','TOTAL_CAP_REASON')
    else:raise RuntimeError('TOTAL_CAP_MISSING')
    pub.write('terminal.json',{'status':'FAILURE_PRESERVED'},critical=True)
    need(not (storage/'overflow').exists() and (storage/'terminal.json').exists(),'TERMINAL_RESERVE')
    for p in storage.iterdir():p.unlink()
    reports.append({'case':'actual_file_total_caps_terminal_reserve','status':'PASS','temporary_padding_removed':True})
    from prepare_offline import Deny
    for name in ('transformer_lens','datasets','pyarrow','transformers.models.qwen3_5.modeling_qwen3_5'):
        try:Deny().find_spec(name)
        except ImportError:pass
        else:raise RuntimeError('IMPORT_GUARD')
    child=subprocess.run([sys.executable,'-B',str(HERE/'prepare_offline.py')],capture_output=True,text=True,timeout=10)
    need(child.returncode==2 and json.loads(child.stdout)['status']=='DISABLED_NO_ROOT_TEXT_LOCK_RELEASE','NO_REAL_ENTRY_AUTHORITY')
    need(not any(n.split('.')[0] in BLOCKED for n in sys.modules),'NO_REAL_TOKENIZER_OR_MODEL_IMPORTED')
    reports.append({'case':'disabled_entry_and_denied_import_names','status':'PASS'})
    report={'status':'PASS','cases':reports,'real_tokenizer_calls':0,'real_model_calls':0,'real_text_read':False,
        'author_submission_read':False,'fake_markers_only':True,'production_authorized':False,
        'total_retained_test_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    (HERE/'TEST_RESULTS.json').write_bytes(jb(report));print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
