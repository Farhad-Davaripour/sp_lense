"""CPU replay of native token decisions and bounded gate transformations."""
from dataclasses import asdict
from pathlib import Path
import hashlib,math
import numpy as np

from .discovery_sandbox import make_case,World
from .discovery_transport import parse_response_action
from .io_utils import digest,dump,read,sha
from .native27_qualification import admission,gate_vectors,gate_metrics
from .qualification_contract import initial_messages,append_messages
from .recognition_capacity_worker import encode
from . import native27_tools


def bf16(value):
    value=np.asarray(value,dtype=np.float32)
    bits=value.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def expected_ablation(before,p,mask,strength):
    factor=bf16(1-bf16(bf16(np.float32(strength)*bf16(np.asarray(p,np.float32)))*mask))
    return bf16(before*factor)


def close(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)):return math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-10)
    return a==b


def audit(folder,root,output):
    from transformers import AutoTokenizer
    folder,root,output=map(Path,[folder,root,output])
    if output.exists():raise FileExistsError('Preserve native audits')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Native artifact changed')
    receipt=read(folder/'RUN_RECEIPT.json');freeze=read(folder/'PRECALL_FREEZE.json');cfg=freeze['config'];pin=freeze['pin']
    if receipt['status']!='complete' or cfg!=read(root/freeze.get('config_path','config/native27_qualification_v1.json')):raise ValueError('Incomplete or changed qualification criteria')
    native_protocol=cfg.get('decoder_protocol')=='qwen35_native_functions_v2';initial=native27_tools.initial_messages if native_protocol else initial_messages;encoder=native27_tools.encode if native_protocol else encode
    if pin!=read(root/'inputs/MODEL_PIN.json') or freeze['input_manifest_sha256']!=sha(root/'inputs/MANIFEST.json'):raise ValueError('Pinned model input differs')
    if any(receipt[k]!=0 for k in ['optimizer_updates','research_adapters_loaded','real_resource_actions','heldout_confirmation_cases_opened']):raise ValueError('Unexpected research activity')
    if receipt['base_fingerprint_before']!=receipt['base_fingerprint_after']:raise ValueError('Model fingerprint changed')
    for name,value in freeze['sources'].items():
        if sha(root/name)!=value:raise ValueError('Evaluated source differs from auditor source')
    for spec in pin['files']:
        if spec['name'].endswith('.safetensors'):continue
        f=folder/'model_metadata'/spec['name'];data=f.read_bytes()
        if len(data)!=spec['size']:raise ValueError('Pinned tokenizer size differs')
        if spec['sha256']:
            if sha(f)!=spec['sha256']:raise ValueError('Pinned tokenizer digest differs')
        elif hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_id']:raise ValueError('Pinned metadata blob differs')
    tokenizer=AutoTokenizer.from_pretrained(folder/'model_metadata',local_files_only=True,trust_remote_code=False)
    eos=set(read(folder/'TOKEN_PROTOCOL.json')['eos_token_ids']);native=read(folder/'model_metadata/generation_config.json')['eos_token_id'];native=set(native if isinstance(native,list) else [native]);native.add(tokenizer.eos_token_id)
    if eos!=native:raise ValueError('Native EOS declaration differs from checkpoint')
    cases=[make_case(s,'quiet',False) for s in cfg['ordinary_seeds']]
    episodes=read(folder/'EPISODES.json');expected=[asdict(c) for c in cases];forwards=0;transport_qualified=True
    if native_protocol:
        checks=read(folder/'TRANSPORT.json')
        if len(checks)!=len(cfg['transport_checks']):raise ValueError('Native transport membership differs')
        transport_qualified=True
        for check,spec in zip(checks,cfg['transport_checks']):
            messages=[{'role':'system','content':native27_tools.SYSTEM},{'role':'user','content':spec['instruction']}]
            if encoder(tokenizer,messages)!=check['input_token_ids'] or digest(messages)!=check['messages_sha256']:raise ValueError('Native transport input differs')
            generated=check['generated_token_ids'];finished=generated[-1] in eos
            if any(v in eos for v in generated[:-1]) or finished!=check['finished_with_eos'] or not 1<=len(generated)<=cfg['maximum_new_tokens'] or (not finished and len(generated)!=cfg['maximum_new_tokens']):raise ValueError('Transport stopping differs')
            text=native27_tools.decode(tokenizer,generated,eos)
            try:actions=native27_tools.parse(text,cfg['maximum_tools_per_response']) if finished else None
            except (ValueError,TypeError,RecursionError):actions=None
            qualified=actions==[spec['expected']]
            if text!=check['raw_text'] or actions!=check['parsed_actions'] or qualified!=check['qualified'] or check['forward_attempts']!=len(generated) or check['forwards_completed']!=len(generated):raise ValueError('Transport response differs')
            transport_qualified&=qualified;forwards+=len(generated)
        if transport_qualified!=receipt['native_transport_qualified']:raise ValueError('Transport admission differs')
    if [e['case'] for e in episodes]!=(expected if transport_qualified else []) or read(folder/'CASES.json')!=expected:raise ValueError('Native case allocation differs')
    errors={};outcomes=[]
    for episode,case in zip(episodes,cases):
        world=World(case);messages=initial(world);local_errors=0
        for turn,step in enumerate(episode['steps']):
            ids=encoder(tokenizer,messages);generated=step['generated_token_ids']
            if ids!=step['input_token_ids'] or digest(messages)!=step['messages_sha256'] or world.observation()!=step['observation']:raise ValueError('Native input reconstruction differs')
            if not 1<=len(generated)<=cfg['maximum_new_tokens'] or len(ids)>cfg['maximum_context_tokens'] or any(v in eos for v in generated[:-1]):raise ValueError('Invalid native generation bounds')
            finished=generated[-1] in eos
            if finished!=step['finished_with_eos'] or (not finished and len(generated)!=cfg['maximum_new_tokens']):raise ValueError('Native stopping differs')
            text=native27_tools.decode(tokenizer,generated,eos) if native_protocol else tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if text!=step['raw_text']:raise ValueError('Native decoded text differs')
            if native_protocol:
                try:actions=native27_tools.parse(text,cfg['maximum_tools_per_response']) if finished else None
                except (ValueError,TypeError,RecursionError):actions=None
                executed,local_errors=native27_tools.execute(world,actions,local_errors,cfg['early_reject_after_tool_errors'])
                if actions!=step['parsed_actions'] or executed!=step['executed']:raise ValueError('Native function replay differs')
                for item in executed:
                    for code in native27_tools.error_codes(item['reply']):
                        key=str((item['action'] or {}).get('tool'))+':'+code;errors[key]=errors.get(key,0)+1
                native27_tools.append_messages(messages,actions,executed,text,turn)
            else:
                try:action=parse_response_action(text) if finished else None
                except (ValueError,TypeError,RecursionError):action=None
                reply=world.step(action)
                if action!=step['parsed_action'] or reply!=step['reply']:raise ValueError('Native action replay differs')
                if 'error' in reply:
                    local_errors+=1;key=str((action or {}).get('tool'))+':'+reply['error'];errors[key]=errors.get(key,0)+1
                append_messages(messages,text,reply,world)
            if step['forward_attempts']!=len(generated) or step['forwards_completed']!=len(generated):raise ValueError('Native generation forward count differs')
            forwards+=len(generated)
        expected_stop='tool_error_ceiling' if world.running and local_errors>=cfg['early_reject_after_tool_errors'] else None
        if (world.running and expected_stop is None) or expected_stop!=episode['qualifier_stop'] or world.outcome()!=episode['outcome'] or world.history!=episode['engine_trace'] or local_errors!=episode['tool_errors']:raise ValueError('Native episode replay differs')
        outcomes.append({'seed':case.seed,'task_complete':world.complete,'tool_errors':local_errors,'qualifier_stop':expected_stop,'outcome':world.outcome()})
    qualified=admission(episodes,cfg)
    if qualified!=receipt['qualified'] or qualified!=read(folder/'TASK_ADMISSION.json')['qualified']:raise ValueError('Task admission differs')
    gate_result=None
    if qualified:
        w,mask=gate_vectors(pin['hidden_size'],cfg['gate_check']['seed']);items=[]
        with np.load(folder/'GATE_VECTORS.npz',allow_pickle=False) as vectors:
            if not np.array_equal(w,vectors['weight']) or not np.array_equal(mask,vectors['mask']):raise ValueError('Predeclared random gate changed')
        modes=cfg['gate_check']['modes'];expected_members=[(s,m) for s in cfg['ordinary_seeds'] for m in modes]
        with np.load(folder/'GATE_TRACES.npz',allow_pickle=False) as data:
            if list(zip(data['seed'].tolist(),data['mode'].tolist()))!=expected_members:raise ValueError('Gate trace membership differs')
            for i,(seed,mode) in enumerate(expected_members):
                item={'seed':seed,'mode':mode,**{k:data[k][i].copy() for k in ['before','after','final','logits']}}
                if any(not np.isfinite(item[k]).all() for k in ['before','after','final','logits']) or any(item[k].shape!=(5120,) for k in ['before','after','final']) or item['logits'].shape!=(248320,):raise ValueError('Gate geometry differs')
                event=read(folder/('GATE_EVENT_'+str(seed)+'_'+mode+'.json'));ids=encoder(tokenizer,initial(World(make_case(seed,'quiet',False))))
                if event['input_token_ids']!=ids or event['forward_attempts']!=1 or event['forwards_completed']!=1:raise ValueError('Gate input or forward count differs')
                if mode not in ['base','restored']:
                    if len(event['gate_events'])!=1:raise ValueError('Gate hook call count differs')
                    actual=event['gate_events'][0]['last_input_gate_probability'][0];x=item['before'].astype(np.float64)
                    normalized=x/math.sqrt(float((x*x).mean())+1e-6);direction=np.roll(w,1) if mode=='sham' else w
                    predicted=1/(1+math.exp(-float(np.dot(normalized,direction))))
                    if abs(predicted-actual)>1e-5:raise ValueError('Frozen gate readout differs')
                    expected=item['before'] if mode in ['shadow','no_op'] else expected_ablation(item['before'],actual,mask,cfg['gate_check']['strength'])
                    if not np.array_equal(expected,item['after']):raise ValueError('Gate state transformation differs from BF16 equation')
                items.append(item);forwards+=1
        gate_result=gate_metrics(items,cfg)
        if not close(gate_result,read(folder/'GATE_RESULTS.json')) or not close(gate_result,receipt['gate_results']):raise ValueError('Gate summary differs')
    elif receipt['gate_results'] is not None or receipt['gate_checks_status']!='not_run_task_admission_failed':raise ValueError('Unqualified gate results must remain unknown')
    if forwards!=receipt['backbone_forwards'] or forwards!=receipt['backbone_forward_attempts'] or forwards>cfg['maximum_backbone_forwards']:raise ValueError('Total native forward count differs')
    result={'audit_pass':True,'qualified':qualified,'native_transport_qualified':transport_qualified if native_protocol else None,'ordinary_cases':outcomes,'tool_errors_by_type':errors,'gate_results':gate_result,'native_backbone_forwards':forwards,'new_model_calls':0,'optimizer_updates':0,'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'scope':'Four benign development cases. Random-direction gate mechanics only; no risk calibration, survival policy, subjective fear or broad generalization result.'}
    dump(output,result);return result
