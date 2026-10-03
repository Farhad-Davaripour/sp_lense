"""Build a NONSECRET private source/cache packet; never contacts a service or Git.

Root uploads the packet privately and fills its observed Drive path/checksum in
an output-free notebook. No raw provider response/client/transport/env is copied.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
PREPARATION=HERE.parent
TEMPLATE_SHA='c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'


def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def build(input_dir,annotations_file,output_dir):
    # Import only the pure schema; never the local service client/transport.
    sys.path.insert(0,str(PREPARATION));import concept_schema as schema
    input_dir=Path(input_dir);annotations_file=Path(annotations_file)
    boundary_raw=(input_dir/'BOUNDARY_FIXTURES.json').read_bytes();freeze_raw=(input_dir/'PILOT_FREEZE.json').read_bytes()
    if digest(boundary_raw)!='de36d4661137b9d2ec8426e955fb091cc82fb00fa880a568cd3d65a8746cc169' or digest(freeze_raw)!='cef423781284720d5b99a46b4ba27eae4616024933b4687b45f7be00ffe536df':
        raise RuntimeError('Exact locally frozen boundary inputs differ')
    boundary=json.loads(boundary_raw);freeze=json.loads(freeze_raw)
    if digest(canonical({k:v for k,v in boundary.items() if k!='sha256_canonical_without_hash'}))!=boundary['sha256_canonical_without_hash']:
        raise RuntimeError('Boundary payload identity differs')
    annotated_raw=annotations_file.read_bytes();annotations=json.loads(annotated_raw)
    if not annotations['completed'] or annotations['scope']!='boundary_only' or annotations['qwen_inference_run'] or annotations['later_actual_histories_annotated']:
        raise RuntimeError('Local annotations are not the completed unqueried boundary-only stage')
    if annotations['selected_model']!=boundary['selected_model'] or len(annotations['states'])!=8:
        raise RuntimeError('Local annotation student model or state count differs')
    by_id={row['id']:row for row in annotations['states']}
    if len(by_id)!=8:raise RuntimeError('Duplicate local annotation states')
    safe=[]
    for row in boundary['rows']:
        saved=by_id[row['id']]
        req=schema.payload(row['history'],row['tools']);identity=schema.request_identity(req)
        if req!=row['request'] or identity!=row['request_identity'] or identity!=saved['request_identity']:
            raise RuntimeError('Local cached request/schema/model identity mismatch')
        if saved['plain_history']!=row['history'] or saved['tools']!=row['tools'] or saved['annotation_scope']!='boundary_only' or saved['future_annotation_reuse_permitted']:
            raise RuntimeError('Local annotation raw input/scope differs')
        compact=schema.compact_aid(saved['decoded'])
        aided=schema.augmented_history(row['history'],saved['decoded'])
        if compact!=saved['compact_aid'] or aided!=saved['aided_history_at_boundary']:
            raise RuntimeError('Local compact first-block identity differs')
        safe.append({'id':row['id'],'boundary':row['boundary'],'cohort_index':row['cohort_index'],
            'source_case_id':row['source_case_id'],'history':copy.deepcopy(row['history']),'tools':copy.deepcopy(row['tools']),
            'history_sha256':schema.digest(row['history']),'tools_sha256':schema.digest(row['tools']),
            'request_identity':identity,'runtime_spec':copy.deepcopy(row['runtime_spec']),
            'starting_state_sha256':row['starting_state_sha256'],'compact_aid':compact,
            'aided_first_input_sha256':schema.digest(aided),'annotation_scope':'boundary_only'})
    model=copy.deepcopy(boundary['selected_model']);model['tokenizer_template_sha256']=TEMPLATE_SHA
    cache={'mode':'first_boundary_copy_only','future_annotations_permitted':False,'qwen_queries_before_freeze':0,
        'input_boundary_file_sha256':digest(boundary_raw),'input_boundary_payload_sha256':boundary['sha256_canonical_without_hash'],
        'input_freeze_file_sha256':digest(freeze_raw),'local_annotations_file_sha256':digest(annotated_raw),
        'schema_version':schema.SCHEMA_VERSION,'jev_model':schema.MODEL,'questions_sha256':schema.digest(schema.QUESTIONS),
        'schema_source_sha256':digest((PREPARATION/'concept_schema.py').read_bytes()),
        'selected_model':model,'caps':{'future_turns':9,'future_generated_tokens':1536,'per_turn_tokens':256,'initial_batch_size':4},
        'states':safe,'local_api_metadata':{'attempted_requests':annotations['api']['attempted_deduplicated_requests'],
        'input_tokens':annotations['api']['input_tokens'],'input_estimated_usd':annotations['api']['input_estimated_usd']},
        'raw_provider_responses_included':False,'client_transport_env_included':False}
    expected_schema=freeze['source_sha256']['jev_concepts_v1/concept_schema.py']
    if cache['schema_source_sha256']!=expected_schema:raise RuntimeError('Local schema changed after API input freeze')
    ledger=CAMPAIGN/'operand_ledger_v1'
    packed=(ledger/'source_bundle.json.gz').read_bytes()
    if digest(packed)!=(ledger/'SOURCE_BUNDLE.sha256').read_text().split()[0]:raise RuntimeError('Frozen ledger dependencies differ')
    old=json.loads(gzip.decompress(packed))['files']
    names=('model_ops.py','fast_inference.py','world.py','memory_world.py','audit.py','activation_capture.py',
           'generation_capture.py','export_private_runs.py','requirements.txt','model_pin.json','ledger_world.py')
    files={name:old[name] for name in names}
    files['ledger_fixture.py']=old['fixture_build.py'];files['base_download.py']=old['preserved_inputs.py']
    # Preserve exact old pure-schema bytes; only newly authored sources require LF.
    files['concept_schema.py']=(PREPARATION/'concept_schema.py').read_bytes().decode('utf-8')
    for name in ('cache_contract.py','cache_worker.py','worker.py','restore_pilot.py','launch.py','collect_pilot.py','PROTOCOL.md'):
        raw=(HERE/name).read_bytes()
        if b'\r\n' in raw:raise RuntimeError('New pilot source must be normalized LF before freezing: '+name)
        files[name]=raw.decode('utf-8')
    cache_text=json.dumps(cache,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    files['data/BOUNDARY_CACHE.json']=cache_text
    files['data/CACHE_FREEZE.json']=json.dumps({'cache_file_sha256':digest(cache_text.encode()),
        'qwen_queries_at_freeze':0,'credential_fields_included':False,'api_logic_included':False},indent=2)+'\n'
    freeze_out={'frozen_before_qwen_queries':True,'scope':'8 cached boundaries; 16 first-copy/plain continuations',
        'sha256':{name:digest(text.encode()) for name,text in files.items()},'raw_provider_responses_included':False,
        'client_transport_env_included':False,'model_weights_sha256':model['weights_sha256'],
        'boundary_payload_sha256':boundary['sha256_canonical_without_hash']}
    files['SOURCE_FREEZE.json']=json.dumps(freeze_out,indent=2)+'\n'
    payload={'schema_version':1,'private_nonsecret_source_and_cache':True,'files':files}
    blob=gzip.compress(canonical(payload),mtime=0)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    destination=output/'JEV_BOUNDARY_PRIVATE_PACKET.json.gz';destination.write_bytes(blob)
    manifest={'packet_sha256':digest(blob),'packet_bytes':len(blob),'source_files':len(files),
        'no_raw_provider_response_client_transport_env':True,'local_annotations_file_sha256':digest(annotated_raw),
        'boundary_payload_sha256':boundary['sha256_canonical_without_hash'],'qwen_queries_before_freeze':0}
    (output/'PACKET_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (output/'SOURCE_FREEZE.json').write_bytes(files['SOURCE_FREEZE.json'].encode())
    print(json.dumps({'packet':str(destination),**manifest},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',required=True)
    parser.add_argument('--annotations-file',required=True);parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    build(args.input_dir,args.annotations_file,args.output_dir)
