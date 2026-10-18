"""Independent inference-free policy reconstruction, physical replay and hashes."""
import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .io_utils import digest,dump,read,sha,load_torch
from .learners import Signals
from .factored_semantic_policy import visible_state,canonical_value_view
from .realistic_actor import bridge_view,workflow_readout,MARKER
from .realistic_world import WorkflowWorld,Scenario,training_cases,ACTIONS
from .realistic_tasks import validate_result


def audit(folder,checkpoint,harness):
    folder,harness=Path(folder),Path(harness);root=Path(__file__).resolve().parents[1]
    receipt=read(folder/'RUN_RECEIPT.json');saved=load_torch(checkpoint);torch.set_num_threads(2)
    assert receipt['status']=='complete' and receipt['model_released']
    assert receipt['optimizer_updates']==receipt['GPU_allocations']==receipt['validation_cases_opened']==receipt['held_cases_opened']==0
    assert receipt['base_fingerprint_before']==receipt['base_fingerprint_after']
    assert receipt['old_head_tensors_unchanged'] and receipt['old_LoRA_tensors_unchanged'] and receipt['exact_raw_OFF_bypass']
    assert sha(checkpoint)==receipt['prior_artifact_sha256']
    freeze=read(folder/'PRECALL_FREEZE.json');assert freeze['development_decisions_seen']==0
    cases={c.case_id:c for c in training_cases(root)};assert [asdict(c) for c in cases.values()]==freeze['cases']
    records=read(folder/'POLICY_RECORDS.json');heads=Signals(1024);heads.load_state_dict(saved['signals']);heads.eval().requires_grad_(False)
    maximum=0.;count=0
    def posterior(query):
        v=torch.tensor(query['format_logits']);p=float(v.logsumexp(1).softmax(0)[0])
        assert abs(p-query['yes_probability'])<1e-6;return p
    with np.load(folder/'BASE_CACHE.npz',allow_pickle=False) as cache:
        def probabilities(view):
            x=torch.tensor(cache[digest({'view':view})+':feature'])
            with torch.no_grad():return heads((x-saved['mean'])/saved['scale']).sigmoid().tolist()
        for key,r in records.items():
            o=r['observation'];assert not r['on']
            assert key==digest({'observation':o,'condition':r['condition'],'on':False})
            native=MARKER+json.dumps(o,sort_keys=True);assert native==r['native_view']
            raw=cache[digest({'view':native})+':logits'];assert np.array_equal(raw,np.asarray(r['native_logits'],dtype=np.float32))
            p=probabilities(native);maximum=max(maximum,max(abs(a-b) for a,b in zip(p,r['original_heads_on_workflow_feature'])))
            assert ACTIONS[int(raw.argmax())]==r['native_action']
            if r['condition']=='native':assert r['action']==r['native_action'] and r['scores']==r['native_logits']
            else:
                view=bridge_view(o);assert view==r['bridge_view'];meta=r['factored_readout'];facts=visible_state(view)
                assert facts==meta['visible_facts'];p=probabilities(view)
                assert max(abs(a-b) for a,b in zip(p,meta['original_linear_probabilities']))<1e-6
                auth=posterior(meta['authorization']);assert meta['authorization']['text']==facts['instruction']
                u=float(facts['verified']<facts['required']);completion=meta['unfinished']
                assert completion['kind']=='visible_completion_contract' and completion['unfinished']==bool(u)
                assert completion['required']==facts['required'] and completion['verified']==facts['verified']
                if auth<.5:
                    assert meta['branch']=='authorization_denied';used=[p[0],u,0.]
                elif u<.5:
                    assert meta['branch']=='current_verification_complete';used=[p[0],u,0.]
                else:
                    assert meta['branch']=='unfinished_authorized_work';t=posterior(meta['threat'])
                    assert meta['threat']['text']==json.dumps({'current_worker':facts['worker'],'notice':facts['notice']},sort_keys=True)
                    canonical=canonical_value_view(view,facts,t>=.5,auth>=.5,u>=.5);value=meta['value']
                    assert value['kind']=='canonical_linear_probe' and value['canonical_view']==canonical
                    cp=probabilities(canonical);assert max(abs(a-b) for a,b in zip(cp,value['canonical_linear_probabilities']))<1e-6
                    c=cp[2];fallback=value['fallback'];potential=(t>=.5 and facts['extensions']>0) or facts['verified']>facts['saved']
                    if c<.5 and potential:
                        assert fallback['text']==canonical;c=max(c,posterior(fallback))
                    else:assert fallback is None
                    assert abs(c-value['yes_probability'])<1e-6;used=[t,u,min(auth,c)]
                maximum=max(maximum,max(abs(a-b) for a,b in zip(used,r['probabilities'])))
                action,values,details=workflow_readout(raw,used,auth,o)
                assert action==r['action'] and values==r['scores'] and details['allowed_actions']==r['allowed_actions']
            count+=1
    assert maximum<1e-6
    audit_root=folder.parent/(folder.name+'_physical_replay');audit_root.mkdir(exist_ok=False);episodes=steps=fresh=0
    for mode,paths in [('reference',sorted((harness/'traces').glob('*.json'))),('model',sorted((folder/'trajectories').glob('*.json')))]:
        for path in paths:
            record=read(path);case=Scenario(**record['case']);assert case==cases[case.case_id]
            condition=record.get('condition','reference');world=WorkflowWorld(case,audit_root/mode/condition/case.case_id)
            assert world.prefix==record['prefix']
            for row in record['trace']:
                assert world.observation()==row['observation'] and world.signals()==row['evaluator_labels']
                if mode=='reference':action=row['action']
                else:
                    point=records[row['point_key']];assert point['observation']==row['observation'] and point['condition']==condition
                    action=point['action']
                assert world.step(action)==row['reply'];steps+=1
            assert world.outcome()==record['outcome'];episodes+=1;fresh+=int(world.recovery is not None)
            if record['outcome']['valid_deliverable']:
                original=(harness/'reference'/case.case_id) if mode=='reference' else (folder/'episodes'/condition/case.case_id)
                result=original/('resumed_result' if record['outcome']['fresh_worker_used'] else 'submitted_result')
                assert validate_result(original,result)
    assert episodes==84
    outer=read(folder.parent/(folder.name+'_EXPORT_RECEIPT.json'));archive=folder.parent/(folder.name+'.zip')
    assert sha(archive)==outer['export']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    result={'audit_pass':True,'source_commit':receipt['source_commit'],'policy_records_recomputed':count,
            'maximum_probability_difference':maximum,'physical_episodes_replayed':episodes,'tool_steps_replayed':steps,
            'fresh_recovery_workers_rechecked':fresh,'CRC_archive_and_member_SHA256_pass':True,
            'validation_cases_opened':0,'held_cases_opened':0,'new_model_forwards':0,'optimizer_updates':0,'GPU_allocations':0}
    dump(folder.parent/(folder.name+'_INDEPENDENT_AUDIT.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('checkpoint');p.add_argument('harness');a=p.parse_args()
    print(json.dumps(audit(a.folder,a.checkpoint,a.harness),indent=2))
