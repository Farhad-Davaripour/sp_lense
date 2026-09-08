"""ONE bounded offline tokenizer-only preparation; never import a model class."""
import hashlib,importlib.metadata,json,os,struct,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('transformer_lens','datasets','pyarrow') or (name.startswith('transformers.models.') and '.modeling_' in name):raise ImportError('FORBIDDEN_PREPARATION_IMPORT')
def main():
    started=time.monotonic();deadline=started+180;out=HERE/'tokenizer_preparation_attempt_001';out.mkdir(exist_ok=False)
    timer=threading.Timer(180,lambda:os._exit(124));timer.daemon=True;timer.start()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.meta_path.insert(0,Deny())
    def audit(event,args):
        if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
        if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):raise RuntimeError('CHECKPOINT_TENSOR_FORBIDDEN')
    sys.addaudithook(audit)
    contract=json.loads((HERE/'PREPARATION_CONTRACT.json').read_bytes());operations=[];records=[]
    def op(name,fn):
        need(time.monotonic()<deadline and len(operations)<64 and name==contract['operation_schedule'][len(operations)],'PREPARATION_OPERATION_BOUND')
        record={'ordinal':len(operations)+1,'name':name,'started':time.monotonic(),'status':'STARTED'};operations.append(record)
        with (out/'operations.jsonl').open('ab') as f:f.write(jb(record));f.flush()
        try:value=fn();record.update(status='COMPLETE',finished=time.monotonic());need(time.monotonic()<deadline,'PREPARATION_DEADLINE');return value
        finally:
            with (out/'operations.jsonl').open('ab') as f:f.write(jb(record));f.flush()
    result={'status':'FAIL','model_loads':0,'model_forwards':0,'model_derivatives':0,'ordinary_encoding_calls':0,'retry_allowed':False}
    try:
        for name,digest in contract['source_sha256'].items():need(sha((HERE/name).read_bytes())==digest,'PREPARATION_SOURCE_BINDING')
        need(importlib.metadata.version('transformers')=='5.15.1' and importlib.metadata.version('tokenizers')=='0.23.0rc0','PINNED_TOKENIZER_RUNTIME')
        pins=json.loads((HERE/'TOKENIZER_PINS.json').read_bytes());snapshot=Path(pins['snapshot'])
        for p in pins['files']:
            raw=(snapshot/p['name']).read_bytes();need(len(raw)==p['bytes'] and sha(raw)==p['sha256'],'PINNED_TOKENIZER_BYTES')
        from transformers import AutoTokenizer
        tok=op('load_native_tokenizer',lambda:AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True))
        need(sha(tok.chat_template.encode())=='273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80','PINNED_CHAT_TEMPLATE')
        fixed=json.loads((HERE/'FIXED_TEXTS.json').read_bytes());cases=[]
        for text in fixed['cases'][:4]:
            key=text['case_key'];messages=[{'role':'user','content':text['prompt']}]
            render=lambda msgs,generation:tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,enable_thinking=False)
            encode=lambda rendered:tok(rendered,add_special_tokens=False,padding=False,truncation=False,return_attention_mask=True)
            rendered=op(key+'/render_generation',lambda:render(messages,True))
            encoded=op(key+'/encode_generation',lambda:encode(rendered));ids=encoded['input_ids'];mask=encoded['attention_mask']
            need(0<len(ids)<=320 and all(type(v) is int for v in ids) and mask==[1]*len(ids),'NEW_PILOT_FULL_INPUT_MAX320')
            user_render=op(key+'/render_no_header',lambda:render(messages,False));user=op(key+'/encode_no_header',lambda:encode(user_render))['input_ids']
            need(ids[:len(user)]==user and ids[len(user):]==[248045,74455,198,248068,271,248069,271],'FULL_GENERATION_HEADER_IDS')
            need(rendered==user_render+'<|im_start|>assistant\n<think>\n\n</think>\n\n','FULL_GENERATION_HEADER_BYTES')
            empty_render=op(key+'/render_empty',lambda:render(messages+[{'role':'assistant','content':''}],False))
            empty=op(key+'/encode_empty',lambda:encode(empty_render))['input_ids']
            need(empty[:len(ids)]==ids and empty[len(ids):]==[248046,198],'EMPTY_ASSISTANT_END_BOUNDARY')
            suffixes={}
            for label,token in (('KEEP',50057),('STOP',48964)):
                complete=op(key+'/render_'+label,lambda:render(messages+[{'role':'assistant','content':label}],False))
                joint=op(key+'/encode_'+label,lambda:encode(complete))['input_ids']
                decoded=op(key+'/decode_'+label,lambda:tok.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False))
                need(complete==rendered+label+'<|im_end|>\n' and joint[:len(ids)]==ids and joint[len(ids):]==[token,248046,198] and decoded==label,'APPENDED_ONE_CONTENT_TOKEN_PROOF')
                suffixes[label]=joint[len(ids):]
            last=op(key+'/decode_final_input',lambda:tok.decode([ids[-1]],skip_special_tokens=False,clean_up_tokenization_spaces=False))
            record={'case_key':key,'prompt_sha256':sha(text['prompt'].encode()),'rendered_chat_utf8_sha256':sha(rendered.encode()),
                'rendered_chat':rendered,'full_token_ids':ids,'attention_mask':mask,'prompt_length':len(ids),'final_input_index':len(ids)-1,
                'generation_header_start_index':len(user),'generation_header_suffix_ids':ids[len(user):],'final_input_token_id':ids[-1],
                'final_input_token_decoded':last,'assistant_end_token_ids':[248046,198],'full_suffix_token_ids':suffixes,
                'content_token_ids':{'KEEP':50057,'STOP':48964},'chat_template_sha256':sha(tok.chat_template.encode()),
                'enable_thinking':False,'truncation':False,'exact_generation_prefix':True,'exactly_one_content_token':True}
            records.append(record);(out/(key+'.json')).write_bytes(jb(record))
            packed=lambda values:sha(struct.pack('<'+'q'*len(values),*values))
            cases.append({'case_key':key,'prompt_id':text['prompt_id'],'audit_only':text['audit_only'],
                'input':{'input_ids':ids,'attention_mask':mask,'prompt_length':len(ids),'final_input_index':len(ids)-1,'token_map':{'KEEP':50057,'STOP':48964}},
                'input_binding':{'prompt_sha256':record['prompt_sha256'],'rendered_chat_utf8_sha256':record['rendered_chat_utf8_sha256'],
                    'chat_template_sha256':record['chat_template_sha256'],'derived_input_int64_le_sha256':packed(ids),'derived_mask_int64_le_sha256':packed(mask)}})
        cases.append(fixed['cases'][4]['exact_native_input_case'])
        need(len(operations)==53 and all(r['status']=='COMPLETE' for r in operations),'EXACT53_TOP_LEVEL_OPERATIONS')
        need(not any(n.startswith('transformers.models.') and '.modeling_' in n for n in sys.modules),'NO_MODEL_CLASS_IMPORTED')
        data={'schema_version':'native_handover_inputs_v1','cases':cases,'input_token_ceiling':320,'exact_lengths':{p['case_key']:p['input']['prompt_length'] for p in cases},
            'fixed_texts_sha256':sha((HERE/'FIXED_TEXTS.json').read_bytes()),'scope':'NEW_PUBLIC_DEVELOPMENT_NOT_FINAL'}
        (HERE/'inputs.json').write_bytes(jb(data));result.update(status='PASS',inputs_sha256=sha(jb(data)),lengths=data['exact_lengths'],model_class_imports=0)
    except BaseException as error:result.update(error_type=type(error).__name__,error_code=str(error)[:240])
    finally:
        result.update(operation_count=len(operations),operation_limit=64,operation_schedule_complete=len(operations)==53,elapsed_seconds=time.monotonic()-started,
            preparation_contract_sha256=sha((HERE/'PREPARATION_CONTRACT.json').read_bytes()),operations=operations)
        (out/'RESULT.json').write_bytes(jb(result));timer.cancel();print(json.dumps(result,sort_keys=True))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
