"""Independent offline CPU audit of the one frozen f46 v3 diagnostic.

No worker/supervisor/comparator import, network, model execution or MLX import.
Numerical replay is implemented here using streamed saved F32 data. The pinned
shared layout helper authenticates metadata only; tokenizer use is CPU-only.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path,PurePosixPath
import struct
import sys

PROJECT=Path(__file__).resolve().parent.parent
PLAN_SHA='f46f436ac656cef2af503111e460f86ecfdb93ea9b910042c23c1220dd82e164'
BASE_SHA='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
SCHEMA='research4.selected-head-equivalence-independent-audit.v3'
CHUNK=1024**2
LOSS={'loss_atol':1e-4,'loss_rtol':1e-4,'gradient_atol':1e-6,'gradient_rtol':1e-3,
    'tensor_relative_l2':1e-3,'tiny_reference_norm':1e-8,'tiny_tensor_max_abs':1e-8,'global_relative_l2':1e-3}
LOGITS={'gradient_atol':1e-3,'gradient_rtol':1e-4,'tensor_relative_l2':1e-4,
    'tiny_reference_norm':1e-8,'tiny_tensor_max_abs':1e-8,'global_relative_l2':1e-4}


def decode(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    def invalid(_):raise ValueError('Nonfinite JSON')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)


def sha(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Expected regular file: '+str(path))
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(CHUNK),b''):result.update(block)
    return result.hexdigest()


def local_path(project,relative):
    if not isinstance(relative,str) or ':' in relative or '\\' in relative:raise ValueError('POSIX project-relative path required')
    value=PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or not value.parts:raise ValueError('Unsafe evidence path')
    current=Path(project).resolve()
    for part in value.parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink evidence prohibited')
    if any('onedrive' in part.casefold() for part in current.resolve().parts):raise ValueError('OneDrive prohibited')
    return current


def binding(project,item,run=None):
    if not isinstance(item,dict) or not {'path','sha256'}<=set(item):raise ValueError('Missing evidence binding')
    path=local_path(project,item['path'])
    if run is not None and not path.resolve().is_relative_to(run.resolve()):raise ValueError('Raw evidence belongs to another run')
    if sha(path)!=item['sha256']:raise ValueError('Bound bytes changed: '+item['path'])
    return path


def header(path,expected):
    with Path(path).open('rb') as stream:
        first=stream.read(8)
        if len(first)!=8:raise ValueError('Truncated tensor header')
        length=struct.unpack('<Q',first)[0]
        if not 2<=length<=16*1024**2:raise ValueError('Tensor header exceeds bound')
        data=stream.read(length)
        if len(data)!=length:raise ValueError('Truncated tensor header data')
    items=decode(data);metadata=items.pop('__metadata__',None)
    if metadata is not None and (not isinstance(metadata,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in metadata.items())):
        raise ValueError('Invalid native tensor metadata')
    if set(items)!=set(expected):raise ValueError('Tensor inventory differs')
    spans=[]
    for name,shape in expected.items():
        spec=items[name]
        if not isinstance(spec,dict) or set(spec)!={'dtype','shape','data_offsets'} or spec['dtype']!='F32' or spec['shape']!=list(shape):
            raise ValueError('Tensor name/shape/dtype differs: '+name)
        offsets=spec['data_offsets']
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(v) is not int for v in offsets):raise ValueError('Invalid tensor offsets')
        begin,end=offsets
        if begin<0 or end-begin!=4*math.prod(shape):raise ValueError('Tensor byte length differs')
        spans.append((begin,end))
    cursor=0
    for begin,end in sorted(spans):
        if begin!=cursor:raise ValueError('Tensor storage gap/overlap')
        cursor=end
    if Path(path).stat().st_size!=8+length+cursor:raise ValueError('Tensor payload length differs')
    return items,8+length


def compare_files(reference,selected,expected,tolerance,zero_a=False):
    import numpy as np
    ah,ao=header(reference,expected);bh,bo=header(selected,expected)
    leaves=[];global_norm=global_error=0.0
    with Path(reference).open('rb') as a,Path(selected).open('rb') as b:
        for name in sorted(expected):
            a.seek(ao+ah[name]['data_offsets'][0]);b.seek(bo+bh[name]['data_offsets'][0])
            remain=4*math.prod(expected[name]);r2=s2=d2=0.0;maximum=0.0;violations=rnz=snz=0
            while remain:
                size=min(CHUNK,remain);ra=a.read(size);rb=b.read(size)
                if len(ra)!=size or len(rb)!=size:raise ValueError('Truncated numerical payload')
                x=np.frombuffer(ra,dtype='<f4').astype(np.float64);y=np.frombuffer(rb,dtype='<f4').astype(np.float64)
                if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite raw numerical value')
                difference=y-x;absolute=np.abs(difference)
                r2+=float(np.sum(x*x,dtype=np.float64));s2+=float(np.sum(y*y,dtype=np.float64));d2+=float(np.sum(difference*difference,dtype=np.float64))
                maximum=max(maximum,float(np.max(absolute,initial=0)))
                violations+=int(np.count_nonzero(absolute>tolerance['gradient_atol']+tolerance['gradient_rtol']*np.abs(x)))
                rnz+=int(np.count_nonzero(x));snz+=int(np.count_nonzero(y));remain-=size
            norm=math.sqrt(r2);relative=math.sqrt(d2/r2) if r2 else None
            norm_ok=maximum<=tolerance['tiny_tensor_max_abs'] if norm<tolerance['tiny_reference_norm'] else relative<=tolerance['tensor_relative_l2']
            exact_a=not(zero_a and name.endswith('.lora_a')) or (rnz==0 and snz==0)
            passed=violations==0 and norm_ok and exact_a
            leaves.append({'name':name,'reference_l2':norm,'selected_l2':math.sqrt(s2),'difference_l2':math.sqrt(d2),
                'relative_l2':relative,'max_absolute_difference':maximum,'violations':violations,'passed':passed})
            global_norm+=r2;global_error+=d2
    if global_norm<=0:raise ValueError('Entire reference is zero; proof would be vacuous')
    ratio=math.sqrt(global_error/global_norm)
    return {'passed':all(row['passed'] for row in leaves) and ratio<=tolerance['global_relative_l2'],
        'global_relative_l2':ratio,'tensor_count':len(expected),'failed_tensors':sum(not row['passed'] for row in leaves),'tensors':leaves}


def snapshot(path,expected):
    import numpy as np
    items,offset=header(path,expected);digest=hashlib.sha256();b_layers=set();a_nonzero=0
    with Path(path).open('rb') as stream:
        for name in sorted(expected):
            digest.update(name.encode());digest.update(str(tuple(expected[name])).encode())
            begin,end=items[name]['data_offsets'];stream.seek(offset+begin);remain=end-begin;nonzero=False
            while remain:
                data=stream.read(min(CHUNK,remain))
                if not data:raise ValueError('Truncated parameter snapshot')
                values=np.frombuffer(data,dtype='<f4')
                if not np.isfinite(values).all():raise ValueError('Nonfinite parameter snapshot')
                nonzero|=bool(np.count_nonzero(values));digest.update(data);remain-=len(data)
            if nonzero and name.endswith('.lora_b'):b_layers.add(int(name.split('.layers.')[1].split('.')[0]))
            if nonzero and name.endswith('.lora_a'):a_nonzero+=1
    return {'state_sha256':digest.hexdigest(),'nonzero_b_layers':sorted(b_layers),'nonzero_a_tensors':a_nonzero}


def raw_mean_nll(path,targets,vocabulary):
    """Independent CPU logsumexp against actual rendered TRAIN target IDs."""
    import numpy as np
    items,offset=header(path,{'target_logits':(len(targets),vocabulary)})
    begin=items['target_logits']['data_offsets'][0];total=0.0
    with Path(path).open('rb') as stream:
        stream.seek(offset+begin)
        for token in targets:
            if type(token) is not int or not 0<=token<vocabulary:raise ValueError('Rendered target outside vocabulary')
            data=stream.read(vocabulary*4)
            if len(data)!=vocabulary*4:raise ValueError('Truncated target-logit row')
            row=np.frombuffer(data,dtype='<f4').astype(np.float64)
            if not np.isfinite(row).all():raise ValueError('Nonfinite target logits')
            maximum=float(np.max(row));total+=maximum+math.log(float(np.sum(np.exp(row-maximum),dtype=np.float64)))-float(row[token])
    return total/len(targets)


def close(a,b,tolerance=LOSS):return abs(a-b)<=tolerance['loss_atol']+tolerance['loss_rtol']*abs(a)


def verify_base(project):
    model=project/'.local/models/qwen3.8-27b-mlx-4bit';manifest=model.with_name(model.name+'.manifest.json')
    if sha(manifest)!=BASE_SHA:raise ValueError('Base manifest differs')
    value=decode(manifest.read_bytes());inventory={item['path']:item for item in value['files']}
    actual={p.relative_to(model).as_posix() for p in model.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model).parts}
    if actual!=set(inventory):raise ValueError('Base inventory differs')
    for name,item in inventory.items():
        path=local_path(project,'.local/models/qwen3.8-27b-mlx-4bit/'+name)
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Actual base bytes differ')
    return model


def audit(project,run,packet,approved_sha=PLAN_SHA):
    project=Path(project).resolve();run=Path(run).resolve()
    if not run.is_relative_to((project/'.local').resolve()) or sha(packet)!=approved_sha or approved_sha!=PLAN_SHA:
        raise ValueError('Wrong frozen diagnostic/run')
    plan=decode(Path(packet).read_bytes())
    if (plan['schema'],plan['rank'],plan['num_layers'],plan['head_window'],plan['original_started_epoch'],plan['original_deadline_epoch'],plan['gradient_tolerances'],plan['logit_tolerances'])!=(
        'research4.selected-head-equivalence-plan.v3',4,64,32,1794187139.176875,1794266339.176875,LOSS,LOGITS):raise ValueError('Frozen diagnostic protocol differs')
    for name,value in plan['source_sha256'].items():
        if sha(project/'mac_studio'/name)!=value:raise ValueError('Frozen proof source changed')
    for key in ('protocol','config','cohort','original_campaign','original_failure_latch','original_supervisor_failure'):binding(project,plan[key])
    request=decode((run/'request.json').read_bytes())
    if request.get('plan_sha256')!=approved_sha or request.get('plan')!=plan:raise ValueError('Owned run request differs from reviewed packet')
    required=['equivalence_receipt.json','equivalence_closeout.json','single_update_evidence.json','diagnostic_state0_weights.safetensors','diagnostic_state1_weights.safetensors']
    required += [f'state{s}_{m}_capture.json' for s in (0,1) for m in ('full','selected')]
    required += ['long_full_forward_capture.json','long_selected_forward_capture.json']
    missing=[name for name in required if not (run/name).is_file()]
    evidence={p.name:sha(p) for p in run.iterdir() if p.is_file()}
    base={'schema':SCHEMA,'approved_plan_sha256':approved_sha,'auditor_sha256':sha(Path(__file__)),'raw_evidence_sha256':evidence,
        'model_executed':False,'eligible_for_tuning_or_serving':False,'optimized_long_pilot_verified':False}
    if missing or (run/'supervisor_failure.json').exists() or (run/'worker_failure.json').exists():
        return {**base,'status':'incomplete_or_failed','missing_files':missing,'real_equivalence_measurements_verified':False,'full_base_files_verified':False,
            'scope':'No complete closed proof; absent measurements are not zero results. Do not start the optimized pilot from this audit.'}
    closeout=decode((run/'equivalence_closeout.json').read_bytes());receipt=decode((run/'equivalence_receipt.json').read_bytes())
    closeout_binding={'path':(run/'equivalence_closeout.json').relative_to(project).as_posix(),'sha256':sha(run/'equivalence_closeout.json')}
    receipt_binding={'path':(run/'equivalence_receipt.json').relative_to(project).as_posix(),'sha256':sha(run/'equivalence_receipt.json')}
    if (closeout.get('schema'),closeout.get('status'),closeout.get('cleanup_confirmed'),closeout.get('plan_sha256'),closeout.get('receipt'),
        closeout.get('original_campaign'),closeout.get('original_failure_latch'),closeout.get('eligible_for_tuning_or_serving'),closeout.get('optimized_long_pilot_verified'))!=(
        'research4.supervised-selected-head-equivalence.v3','passed',True,approved_sha,receipt_binding,plan['original_campaign'],plan['original_failure_latch'],False,False):raise ValueError('Supervised proof closeout differs')
    if (receipt.get('schema'),receipt.get('status'),receipt.get('plan_sha256'),receipt.get('rank'),receipt.get('num_layers'),receipt.get('gradient_tensors'),
        receipt.get('diagnostic_optimizer_updates'),receipt.get('eligible_for_tuning_or_serving'),receipt.get('original_deadline_epoch'))!=(
        'research4.selected-head-equivalence-receipt.v3','passed',approved_sha,4,64,992,1,False,1794266339.176875):raise ValueError('Worker proof scope differs')
    model=verify_base(project)
    helper_path=project/'mac_studio/game_all_layers_layout.py'
    spec=importlib.util.spec_from_file_location('reviewed_layout_metadata_only',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    expected=helper.expected_adapter_layout(model,4,64)
    if len(expected)!=992:raise ValueError('Pinned all-block layout differs')
    # Reconstruct target IDs using only the two declared TRAIN rows, never game
    # or held-out data. The renderer itself is source-hash authenticated above.
    renderer_path=project/'mac_studio/game_training_renderer.py'
    spec=importlib.util.spec_from_file_location('reviewed_cpu_renderer_only',renderer_path);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    cohort_path=binding(project,plan['cohort']);data=cohort_path.parent
    rows=[decode(line) for line in (data/'train768.jsonl').read_text(encoding='utf-8').splitlines()]
    schema=decode((data/'action_schema.json').read_bytes());targets={}
    for kind in ('short','long'):
        item=plan['inputs'][kind];row=rows[item['training_index']]
        if hashlib.sha256(renderer.canonical(row).encode()).hexdigest()!=item['row_sha256']:raise ValueError('TRAIN input bytes differ')
        ids,prefix=renderer.render(tokenizer,row,schema)
        if (len(ids),prefix)!=(item['total_tokens'],item['prefix_tokens']):raise ValueError('Rendered target boundaries differ')
        targets[kind]=ids[prefix:]
    vocabulary=decode((model/'config.json').read_bytes())['text_config']['vocab_size']
    snapshots=[snapshot(run/f'diagnostic_state{state}_weights.safetensors',expected) for state in (0,1)]
    if snapshots[0]['nonzero_b_layers'] or snapshots[0]['nonzero_a_tensors']<1 or snapshots[1]['nonzero_b_layers']!=list(range(64)) or snapshots[0]['state_sha256']==snapshots[1]['state_sha256']:
        raise ValueError('Real zero-B and changed all-block parameter snapshots are not established')
    update=decode((run/'single_update_evidence.json').read_bytes())
    if (update.get('optimizer_step'),update.get('weight_state_sha256'),update.get('nonzero_b_weight_layers'))!=(1,snapshots[1]['state_sha256'],list(range(64))):raise ValueError('Single-update evidence differs')
    proofs=[]
    for label,state in (('state0',0),('state1',1),('long',1)):
        suffix='_forward' if label=='long' else ''
        captures=[decode((run/f'{label}_{method}{suffix}_capture.json').read_bytes()) for method in ('full','selected')]
        kind='long' if label=='long' else 'short';n=len(targets[kind]);paths=[];means=[]
        for capture in captures:
            if (capture['weight_state_sha256'],capture['target_tokens'],capture['target_logits']['shape'])!=(snapshots[state]['state_sha256'],n,[n,vocabulary]):raise ValueError('Capture state or exact target coverage differs')
            path=binding(project,capture['target_logits'],run);paths.append(path)
            mean=raw_mean_nll(path,targets[kind],vocabulary);means.append(mean)
            if not close(capture['mean_nll'],mean):raise ValueError('Reported loss differs from raw logits and actual TRAIN labels')
        logits=compare_files(paths[0],paths[1],{'target_logits':(n,vocabulary)},LOGITS)
        if not logits['passed'] or not close(means[0],means[1]):raise ValueError('Independent target-logit/loss equivalence failed')
        proof={'label':label,'target_tokens':n,'raw_cpu_mean_nll':means,'target_logits':logits}
        if label!='long':
            gradient_paths=[binding(project,capture['gradients'],run) for capture in captures]
            gradients=compare_files(gradient_paths[0],gradient_paths[1],expected,LOSS,zero_a=state==0)
            if not gradients['passed']:raise ValueError('Independent all-tensor gradient equivalence failed')
            for field in ('reference_l2','selected_l2'):
                covered={int(item['name'].split('.layers.')[1].split('.')[0]) for item in gradients['tensors'] if item['name'].endswith('.lora_b') and item[field]>0}
                if covered!=set(range(64)):raise ValueError('Raw gradient B coverage differs')
            if state and not any(item['name'].endswith('.lora_a') and item['reference_l2']>0 and item['selected_l2']>0 for item in gradients['tensors']):raise ValueError('Updated A-gradient branch is not exercised')
            proof['gradients']=gradients
        proofs.append(proof)
    if any(name=='mlx' or name.startswith('mlx.') or name=='mlx_lm' or name.startswith('mlx_lm.') for name in sys.modules):raise RuntimeError('Offline audit unexpectedly imported model runtime')
    return {**base,'status':'passed_complete','equivalence_closeout':closeout_binding,'worker_receipt':receipt_binding,
        'real_equivalence_measurements_verified':True,'full_base_files_verified':True,'snapshots':snapshots,'proofs':proofs,
        'scope':'Offline replay of saved actual measurements only. This is not a Qwen rerun, long backward fit, trained model, inference speed or game result. Root review is still required before a separate optimized long pilot.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',required=True,type=Path)
    parser.add_argument('--run',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',default=PLAN_SHA)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    project=args.project.resolve();run=local_path(project,args.run);packet=local_path(project,args.plan);out=local_path(project,args.out)
    if not all(path.resolve().is_relative_to((project/'.local').resolve()) for path in (run,packet,out)):raise ValueError('Use private project-local evidence/output')
    try:report=audit(project,run,packet,args.plan_sha256)
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        report={'schema':SCHEMA,'status':'incomplete_or_failed','approved_plan_sha256':args.plan_sha256,'auditor_sha256':sha(Path(__file__)),
            'error':f'{type(exc).__name__}: {exc}','model_executed':False,'eligible_for_tuning_or_serving':False,
            'optimized_long_pilot_verified':False,'real_equivalence_measurements_verified':False,'full_base_files_verified':False,
            'raw_evidence_sha256':{p.name:sha(p) for p in run.iterdir() if p.is_file()} if run.is_dir() else {}}
    report['created_utc']=datetime.now(timezone.utc).isoformat();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'receipt':str(out),'sha256':sha(out)},indent=2))
    raise SystemExit(0 if report['status']=='passed_complete' else 1)


if __name__=='__main__':main()
