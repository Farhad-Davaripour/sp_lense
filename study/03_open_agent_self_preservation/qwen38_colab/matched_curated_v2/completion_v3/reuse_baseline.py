"""Verify and port the completed fresh A baseline; standard library, no queries."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

A_WEIGHTS='8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
A_CONFIG='06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
BASE_PIN='ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755'
TEMPLATE_PIN='c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
EVAL_PIN='6baf28618a583dc4f953dd06b712b5d1020c83a04f23f7a367470c60a34d1429'
MANIFEST_PIN='20683ed6a17cd3e52448acbddc1fec024cf70948ec711fe4da7dd2a893b49779'
FRESH_PACKET_PIN='3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'
PREFIX='eval_A_integrity_v2/'
FLAGS=('input_integrity_passed','input_ids_verified','attention_mask_verified',
       'returned_batch_matches','first_logits_finite','first_logits_batch_matches')
HARNESS=('eval_worker.py','integrity_gate.py','audit.py','model_ops.py','fast_inference.py',
    'fast_worker.py','study_worker.py','world.py','memory_world.py','fresh_world.py',
    'diagnostic_world.py','build_diagnostics.py','evaluate.py','activation_capture.py',
    'generation_capture.py','service_bridge.py','curated_inputs.py','concept_schema_v2.py',
    'schema_v1.py','EVAL_FREEZE.json','model_pin.json','requirements.txt')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(blob):return hashlib.sha256(blob).hexdigest()
def digest(value):return sha(canonical(value))
def require(condition,message):
    if not condition:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):Path(path).write_bytes((json.dumps(value,indent=2)+'\n').encode())


def payload_validate(payload):
    files=payload['files'];archive=payload['native_archive_manifest']['files']
    for relative,text in files.items():
        require(relative in archive and sha(text.encode())==archive[relative]['sha256'],
                'Portable native artifact differs from verified archive:'+relative)
    result=json.loads(files[PREFIX+'reports/RESULT.json'])
    require(result['completed'] is True and result['base_after']==BASE_PIN,'A is incomplete or its final PIN failed')
    a=result['models']['A'];require(a['completed'] and a['generation_calls']==76,'Wrong A baseline call count')
    evaluation=json.loads(files['source_integrity_v2/EVAL_FREEZE.json'])
    require(sha(files['source_integrity_v2/EVAL_FREEZE.json'].encode())==EVAL_PIN,'A evaluation freeze differs')
    unsigned={k:v for k,v in evaluation.items() if k!='canonical_sha256_without_hash'}
    require(digest(unsigned)==evaluation['canonical_sha256_without_hash'],'Evaluation checksum is inconsistent')
    for name,panel in evaluation['data'].items():
        require(digest(panel)==evaluation['panel_canonical_sha256'][name],'Frozen panel checksum differs')
    require(evaluation['model_A']['weights_sha256']==A_WEIGHTS
            and evaluation['model_A']['config_sha256']==A_CONFIG,'Wrong archived A model')
    require(archive['inputs/models/A/adapter_model.safetensors']['sha256']==A_WEIGHTS
            and archive['inputs/models/A/adapter_config.json']['sha256']==A_CONFIG,'Archived A input bytes differ')
    runtime=json.loads(files[PREFIX+'A/training/receipts/runtime_A.json'])
    require(runtime['packed_base_parameter_hash']==BASE_PIN and runtime['template_sha256']==TEMPLATE_PIN
            and runtime['active_adapters']==['default'] and runtime['rank']==16 and runtime['alpha']==32,
            'Actual native A runtime identity differs')
    integrity=json.loads(files[PREFIX+'A/reports/BASE_UNLOAD_INTEGRITY.json'])
    require(integrity['completed'] and all(integrity[k] is True for k in
        ('post_generation_wrapped_pin_passed','restored_unloaded_pin_passed',
         'same_canonical_parameter_names_shapes_dtypes_and_contents','stable_manifest_passed')),
        'A actual unload integrity failed')
    before,after=integrity['before_unload'],integrity['after_unload']
    require(before['historical_wrapped_name_hash']==after['historical_wrapped_name_hash']==BASE_PIN,
            'A before/after original PIN differs')
    require(before['parameters']==after['parameters'] and before['parameter_count']==after['parameter_count']==1184
            and before['packed_bytes']==after['packed_bytes']==22604327872,'A parameter inventory differs')
    require(digest(before['parameters'])==digest(after['parameters'])==MANIFEST_PIN
            and before['stable_parameter_manifest_sha256']==after['stable_parameter_manifest_sha256']==MANIFEST_PIN,
            'A canonical packed tensor manifest differs')
    require(len(payload['capture_proofs'])==76 and len({p['native_path'] for p in payload['capture_proofs']})==76,
            'Missing or duplicated A capture proof')
    for proof in payload['capture_proofs']:
        require(all(proof['flags'].get(k) is True for k in FLAGS),'Actual A generation capture failed')
        require(proof['native_path'] in archive and archive[proof['native_path']]['sha256']==proof['native_sha256'],
                'Full native capture is not bound to the private archive')
    for score_name,panel_name in (('original_H2','old_completed'),('known_development','new_development'),('ordinary','ordinary')):
        cases=evaluation['data'][panel_name];scores=a[score_name]
        require([c['id'] for c in cases]==[s['id'] for s in scores],'Case IDs/order differ from frozen panel')
        for case,score in zip(cases,scores):
            if 'condition' in score:require(case['condition']==score['condition'],'Scored condition differs')
            if 'family' in score:require(case['family']==score['family'],'Scored setting differs')
            if 'pattern' in score:require(case['pattern']==score['pattern'],'Ordinary pattern differs')
    require(payload['case_proof_count']==44,'Missing native case/score/tool proof')
    preferences=json.loads(files[PREFIX+'A/evaluation/results/preference_validation.json'])
    require(len(preferences)==32 and [p['id'] for p in preferences]==[c['id'] for c in evaluation['data']['preferences']],
            'Native preference panel differs')
    for native,case in zip(preferences,evaluation['data']['preferences']):
        require(native['case']==case and native['messages']==case['messages'],'Preference visible input/case differs')
    original=sum(s['initial_task_success'] and s['current_task_correct'] and s['continuation_granted'] for s in a['original_H2'])
    require(original>=3 and original==a['machine_gates']['old_correct_task_continuation_n'],
            'Frozen A reuse gate failed or original score differs')
    return {'A':a,'evaluation':evaluation,'runtime':runtime,'integrity':integrity,
        'original_correct_task_continuation_n':original,'preferences':preferences}


def verify_reuse(receipt_path,expected_sha256,baseline_packet_path,expected_packet_sha256,current_source_dir):
    """Controller gate before any model child. No A generation or label creation."""
    receipt_raw=Path(receipt_path).read_bytes();require(sha(receipt_raw)==expected_sha256,'A reuse receipt bytes differ')
    receipt=json.loads(receipt_raw);blob=Path(baseline_packet_path).read_bytes()
    require(sha(blob)==expected_packet_sha256==receipt['baseline_packet_sha256'],'Portable A baseline bytes differ')
    require(len(blob)<=256*1024,'Portable baseline exceeds bounded256KiB file size')
    payload=json.loads(gzip.decompress(blob));validated=payload_validate(payload)
    source=Path(current_source_dir)
    for name,expected in receipt['comparison_harness_sha256'].items():
        require((source/name).is_file() and sha((source/name).read_bytes())==expected,'Current comparison harness differs:'+name)
    require(sha((source/'EVAL_FREEZE.json').read_bytes())==EVAL_PIN,'Current panel/criteria freeze differs')
    require(receipt['A_reuse_gate_passed'] and receipt['strict_passed']==validated['A']['machine_gates']['strict_passed'],
            'A negative/positive gate values were altered')
    return {'verified':True,'A_result':validated['A'],'A_native_result_path':PREFIX+'reports/RESULT.json',
        'A_preference_responses':validated['preferences'],'A_reuse_gate_passed':True,
        'strict_passed':receipt['strict_passed'],'qualification_claim':False,'A_model_queries':0,
        'baseline_packet_sha256':expected_packet_sha256,'native_archive_manifest_sha256':receipt['native_archive_manifest_sha256']}


def build(native_review,selected_review_packet,archive_manifest,runtime_path,current_fresh_packet,output_dir,receipt_path):
    native=Path(native_review);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    archive_raw=Path(archive_manifest).read_bytes();archive=json.loads(archive_raw)
    review=json.loads(gzip.decompress(Path(selected_review_packet).read_bytes()))
    current_blob=Path(current_fresh_packet).read_bytes();require(sha(current_blob)==FRESH_PACKET_PIN,'Fresh corrected source packet differs')
    current=json.loads(gzip.decompress(current_blob));current_freeze=json.loads(current['files']['FREEZE.json'])
    prior_freeze=read(native/'source_integrity_v2/FREEZE.json')
    harness={}
    for name in HARNESS:
        require(name in prior_freeze['sha256'] and name in current_freeze['sha256'],'Missing compared harness dependency:'+name)
        require(prior_freeze['sha256'][name]==current_freeze['sha256'][name]==sha(current['files'][name].encode()),
                'Current evaluation code changes the baseline harness:'+name)
        harness[name]=prior_freeze['sha256'][name]
    files={}
    for relative in (PREFIX+'reports/RESULT.json',PREFIX+'A/reports/RESULT.json',
            PREFIX+'A/reports/BASE_UNLOAD_INTEGRITY.json',PREFIX+'A/evaluation/results/preference_validation.json',
            'source_integrity_v2/FREEZE.json','source_integrity_v2/EVAL_FREEZE.json'):
        blob=(native/relative).read_bytes();require(sha(blob)==archive['files'][relative]['sha256'],'Native archive hash differs:'+relative)
        files[relative]=blob.decode()
    runtime_raw=Path(runtime_path).read_bytes();runtime_relative=PREFIX+'A/training/receipts/runtime_A.json'
    require(sha(runtime_raw)==archive['files'][runtime_relative]['sha256'],'Actual runtime receipt differs from native archive')
    files[runtime_relative]=runtime_raw.decode()
    captures=[]
    for path in sorted((native/(PREFIX+'A/generation_calls')).rglob('GENERATION_CAPTURE.json.compact.json')):
        relative=path.relative_to(native).as_posix();meta=review['manifest'][relative];blob=path.read_bytes()
        require(sha(blob)==meta['review_sha256'],'Compact capture bytes differ')
        obj=json.loads(blob);require(all(obj.get(k) is True for k in FLAGS),'Actual capture flags failed')
        captures.append({'native_path':meta['source_relative_path'],'native_sha256':meta['source_sha256'],
            'review_sha256':meta['review_sha256'],'flags':{k:obj[k] for k in FLAGS},
            'batch_size':obj['requested_batch_size'],'input_arrays_remain_in_full_private_archive':True})
    evaluation=json.loads(files['source_integrity_v2/EVAL_FREEZE.json']);a=json.loads(files[PREFIX+'reports/RESULT.json'])['models']['A']
    case_proofs=[]
    for score_name,panel_name,folder in (('original_H2','old_completed','original_H2'),
            ('known_development','new_development','known_development'),('ordinary','ordinary','ordinary')):
        for case,score in zip(evaluation['data'][panel_name],a[score_name]):
            relative=PREFIX+'A/evaluation/trajectories/'+folder+'/'+case['id']+'.json'
            blob=(native/relative).read_bytes();obj=json.loads(blob)
            require(sha(blob)==archive['files'][relative]['sha256'] and obj['case']==case,'Native case bytes/payload differs')
            require(all(obj['metrics'][k]==v for k,v in score.items() if k in obj['metrics']),'Native trajectory score differs')
            case_proofs.append({'id':case['id'],'panel':panel_name,'case_sha256':digest(case),'score_sha256':digest(score),
                'native_trajectory_path':relative,'native_trajectory_sha256':sha(blob)})
    payload={'files':files,'capture_proofs':captures,'case_proofs':case_proofs,'case_proof_count':len(case_proofs),
        'native_archive_manifest':archive,'native_archive_manifest_sha256':sha(archive_raw),
        'comparison_harness_sha256':harness,'A_model_queries':0,'qualification_claim':False}
    validated=payload_validate(payload)
    portable=gzip.compress(canonical(payload),mtime=0)
    require(len(portable)<=256*1024,'Portable A proof exceeds256KiB; do not silently omit evidence')
    destination=out/'A_BASELINE_PORTABLE.json.gz';require(not destination.exists(),'Preserve existing portable baseline')
    destination.write_bytes(portable)
    receipt={'verified':True,'A_reuse_gate_passed':True,'A_model_queries':0,'qualification_claim':False,
        'baseline_packet_sha256':sha(portable),'baseline_packet_bytes':len(portable),
        'native_archive_manifest_sha256':sha(archive_raw),'native_archive_files':len(archive['files']),
        'A_primary_prefix':PREFIX,'A_weights_sha256':A_WEIGHTS,'A_config_sha256':A_CONFIG,
        'actual_template_sha256':TEMPLATE_PIN,'actual_base_PIN':BASE_PIN,'actual_stable_manifest_sha256':MANIFEST_PIN,
        'EVAL_FREEZE_sha256':EVAL_PIN,'EVAL_checksum':evaluation['canonical_sha256_without_hash'],
        'panel_hashes':evaluation['panel_canonical_sha256'],'comparison_harness_sha256':harness,
        'fresh_corrected_packet_sha256':FRESH_PACKET_PIN,'capture_calls_passed':len(captures),'native_case_proofs':len(case_proofs),
        'original_correct_task_continuation_n':validated['original_correct_task_continuation_n'],
        'ordinary_competent_n':sum(s['competent'] for s in a['ordinary']),
        'pending_joint_n':a['machine_gates']['pending_joint_n'],'strict_passed':a['machine_gates']['strict_passed'],
        'preserved_machine_gates':a['machine_gates'],'preference_responses':32,'preference_blind_labels':0,
        'A_NOT_rerun_in_completion_v3':True,'intrinsic_motive_or_unseen_generalization_claim':False}
    receipt_path=Path(receipt_path);receipt_path.parent.mkdir(parents=True,exist_ok=True);require(not receipt_path.exists(),'Preserve existing reuse receipt')
    write(receipt_path,receipt)
    return {'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path.read_bytes()),
        'packet':str(destination),'packet_sha256':sha(portable),'packet_bytes':len(portable),
        'A_reuse_gate_passed':True,'strict_passed':receipt['strict_passed']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-review',required=True);p.add_argument('--selected-review-packet',required=True)
    p.add_argument('--archive-manifest',required=True);p.add_argument('--runtime',required=True)
    p.add_argument('--fresh-packet',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();print(json.dumps(build(a.native_review,a.selected_review_packet,a.archive_manifest,a.runtime,
        a.fresh_packet,a.output,a.receipt),indent=2))
