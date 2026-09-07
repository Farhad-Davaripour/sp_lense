"""Read-only binding to the original f04 selection and already-tested science."""
import json
from core import HERE,ROOT,read,require,sha,git
PARENT='diagnostics/semantic_editor_f04_assay_v1'
COMMIT='b608415e9e8fd628f5a41e94f9e8a1a9628ab4b8'
INVENTORY='3ceb624695270dc2105b663f04352ab3b3ab74a827abadd6943a9c8135e7494e'
SELECTION='df159c27f49b49aea512189b6bed6cf7cf6d21842836adfcd20699d51c38fb09'

def parent_entries():
    raw=git('show',COMMIT+':'+PARENT+'/FINAL_INVENTORY.json')
    require(sha(raw)==INVENTORY,'original immutable f04 inventory')
    return {r['path']:r for r in json.loads(raw)['files']}

def authenticated(name,entries):
    raw=(ROOT/PARENT/name).read_bytes();row=entries[name]
    require(sha(raw)==row['sha256'] and len(raw)==row['bytes'],'authenticated parent '+name)
    return raw

def check_selection_freeze():
    # Original selection is authenticated where it was frozen, not reconstructed in V2.
    raw=(ROOT/PARENT/'selection_freeze.json').read_bytes();require(sha(raw)==SELECTION,'original selection lock')
    for name,digest in json.loads(raw)['source_sha256'].items():
        require(sha((ROOT/PARENT/name).read_bytes())==digest,'original selection/science '+name)
    return SELECTION

def reuse():
    entries=parent_entries();check_selection_freeze();same={}
    for path in HERE.iterdir():
        if path.is_file() and path.name in entries:
            original=authenticated(path.name,entries)
            if path.read_bytes()==original:same[path.name]=sha(original)
    needed=('editor.py','saved_judge.py','run.py','entry.py','inputs.py','inputs.json','input_lock.json',
            'tokens_01.json','tokens_02.json','production_plan.json','source_bindings.json','fitted_parameters.json',
            'guard_candidate.py','hook_record.py','numeric_audit.py','word_scoring.py','word_reference.py',
            'mixed_boundary.py','mixed_scoring.py','learned_gate.py','gate_reload.py','gate_reference.py',
            'locked_backend.py','final_adjudication.py','judge.py','usage_receipt.py','model_cache_lock.json')
    require(all(n in same for n in needed),'unchanged inputs, scientific and production sources')
    evidence={}
    for name in ('fake_batch_receipt.json','pure_changed_paths.json','POST_BATCH_ADJUDICATION.json',
                 'synthetic/full_pair/final_closeout.json','synthetic/full_pair/judge_results.json',
                 'synthetic/full_pair/capture.json','synthetic/full_pair/audit/capture.json',
                 'synthetic/deadline/capture.json'):
        raw=authenticated(name,entries);evidence[name]={'sha256':sha(raw),'value':json.loads(raw)}
    require(evidence['synthetic/full_pair/final_closeout.json']['value']['classification']=='PASS','prior complete fake matrix')
    require(all(evidence[n]['value']['status']=='complete_valid' for n in ('synthetic/full_pair/capture.json','synthetic/full_pair/audit/capture.json')),'prior normal worker/audit evidence')
    require(evidence['synthetic/deadline/capture.json']['value']['owned_worker']['cleanup_faults'],'preserved prior deadline failure')
    return {'parent_commit':COMMIT,'parent_inventory_sha256':INVENTORY,'byte_identical_files':same,
            'evidence':{n:{'sha256':v['sha256']} for n,v in evidence.items()},
            'prior_full_fake_matrix_and_normal_worker_audit_reused_not_rerun':True,
            'parent_overall_remains':'INCONCLUSIVE_PREPARATION_NOT_RELEASED','inputs_or_tokenizers_rerun':False}
