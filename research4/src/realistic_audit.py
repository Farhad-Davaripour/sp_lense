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
from .learners import Signals,GateState
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
    maximum=0.;count=0;enabled_effects=[]
    def posterior(query):
        v=torch.tensor(query['format_logits']);p=float(v.logsumexp(1).softmax(0)[0])
        assert abs(p-query['yes_probability'])<1e-6;return p
    with np.load(folder/'BASE_CACHE.npz',allow_pickle=False) as cache:
        def probabilities(view):
            feature=cache[digest({'view':view})+':feature'];assert feature.shape==(1024,) and np.isfinite(feature).all()
            x=torch.tensor(feature)
            with torch.no_grad():return heads((x-saved['mean'])/saved['scale']).sigmoid().tolist()
        for key,r in records.items():
            o=r['observation']
            assert key==digest({'observation':o,'condition':r['condition'],'on':r['on']})
            native=MARKER+json.dumps(o,sort_keys=True);assert native==r['native_view']
            raw=cache[digest({'view':native})+':logits'];assert raw.shape==(6,) and np.isfinite(raw).all()
            actual=np.asarray(r['native_logits'],dtype=np.float32);assert actual.shape==(6,) and np.isfinite(actual).all()
            if not r['on']:assert np.array_equal(raw,actual)
            else:enabled_effects.append(float(np.max(np.abs(raw-actual))))
            p=probabilities(native);maximum=max(maximum,max(abs(a-b) for a,b in zip(p,r['original_heads_on_workflow_feature'])))
            assert ACTIONS[int(actual.argmax())]==r['native_action']
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
                action,values,details=workflow_readout(actual,used,auth,o)
                assert action==r['action'] and values==r['scores'] and details['allowed_actions']==r['allowed_actions']
            count+=1
    assert maximum<1e-6
    audit_root=folder.parent/(folder.name+'_physical_replay');audit_root.mkdir(exist_ok=False);episodes=steps=fresh=0
    fault_root=folder.parents[1]/'fault_controls_01'
    paths_by_mode=[('reference',sorted((harness/'traces').glob('*.json'))),('model',sorted((folder/'trajectories').glob('*.json')))]
    if fault_root.exists():paths_by_mode.append(('fault',sorted((fault_root/'traces').glob('*.json'))))
    model_paths=paths_by_mode[1][1];expected_model_episodes=36 if receipt.get('phase')=='remaining_conditions' else 24
    assert len(model_paths)==expected_model_episodes
    model_records=[read(p) for p in model_paths]
    for condition,expected in receipt['closed_loop'].items():
        subset=[r for r in model_records if r['condition']==condition];assert len(subset)==12
        assert {r['case']['case_id'] for r in subset}==set(freeze['closed_loop_case_ids'])
        recomputed={'cases':12,'succeeded':sum(r['outcome']['success'] for r in subset),'valid_deliverables':sum(r['outcome']['valid_deliverable'] for r in subset),
                    'raw_cancellation_violations':sum(r['outcome']['raw_cancellation_violations'] for r in subset),
                    'irrelevant_preservation':sum(r['outcome']['irrelevant_preservation'] for r in subset if r['case']['condition'] not in ['renewal','checkpoint','denied_renewal'])}
        assert recomputed==expected
    for mode,paths in paths_by_mode:
        for path in paths:
            record=read(path);case=Scenario(**record['case']);assert case==cases[case.case_id]
            condition=record.get('condition',record.get('policy','reference'));world=WorkflowWorld(case,audit_root/mode/condition/case.case_id)
            if 'prefix' in record:assert world.prefix==record['prefix']
            gate=GateState(saved);active=False
            for row in record['trace']:
                assert world.observation()==row['observation']
                if 'evaluator_labels' in row:assert world.signals()==row['evaluator_labels']
                if mode in ['reference','fault']:action=row['action']
                else:
                    point=records[row['point_key']];assert point['observation']==row['observation'] and point['condition']==condition
                    if condition=='decoder_always_on':expected_on=True
                    elif condition=='decoder_oracle':
                        t,u,c=world.signals();active=bool(u and c and not world.cancelled and (t or active));expected_on=active
                    elif condition=='decoder_learned':
                        with np.load(folder/'BASE_CACHE.npz',allow_pickle=False) as arrays:feature=arrays[digest({'view':bridge_view(row['observation'])})+':feature']
                        predicted,prob,expected_on=gate.update(feature);router=row['router']
                        assert max(abs(a-b) for a,b in zip(predicted,router['signal_probabilities']))<1e-6
                        assert abs(prob-router['gate_probability'])<1e-6 and expected_on==router['gate_ON']
                    else:expected_on=False
                    assert point['on']==expected_on
                    if 'raw_gate' in row:assert row['raw_gate']==expected_on
                    action=point['action']
                assert world.step(action)==row['reply'];steps+=1
            assert world.outcome()==record['outcome'];episodes+=1;fresh+=int(world.recovery is not None)
            if record['outcome']['valid_deliverable']:
                original=(harness/'reference'/case.case_id) if mode=='reference' else (fault_root/condition/case.case_id) if mode=='fault' else (folder/'episodes'/condition/case.case_id)
                result=original/('resumed_result' if record['outcome']['fresh_worker_used'] else 'submitted_result')
                assert validate_result(original,result)
    assert episodes==60+expected_model_episodes+(180 if fault_root.exists() else 0)
    if receipt.get('phase')=='remaining_conditions':
        assert enabled_effects and max(enabled_effects)>1e-5
        assert all(r['bitwise_equal'] for r in read(folder/'CACHE_VALIDATION.json'))
    representation=folder.parents[1]/'representation_control_01/REPRESENTATION_CONTROL.json';paired_checked=False
    if representation.exists():
        paired=read(representation);case=cases[paired['base_training_case_id']];world=WorkflowWorld(case,audit_root/'paired_control'/case.case_id)
        assert world.observation()==paired['with_successor'] and world.signals()==paired['labels_with_successor']
        world.successor=False
        assert world.observation()==paired['without_successor'] and world.signals()==paired['labels_without_successor']
        assert paired['labels_with_successor'][2]==1 and paired['labels_without_successor'][2]==0
        assert bridge_view(paired['with_successor'])==bridge_view(paired['without_successor']);paired_checked=True
    outer=read(folder.parent/(folder.name+'_EXPORT_RECEIPT.json'));archive=folder.parent/(folder.name+'.zip')
    assert sha(archive)==outer['export']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
    result={'audit_pass':True,'source_commit':receipt['source_commit'],'audit_implementation_sha256':sha(Path(__file__)),'policy_records_recomputed':count,
            'maximum_probability_difference':maximum,'physical_episodes_replayed':episodes,'tool_steps_replayed':steps,
            'fresh_recovery_workers_rechecked':fresh,'CRC_archive_and_member_SHA256_pass':True,
            'supplementary_counterfactual_pair_rechecked':paired_checked,
            'enabled_adapter_records':len(enabled_effects),'maximum_enabled_logit_effect':max(enabled_effects,default=0.),
            'validation_cases_opened':0,'held_cases_opened':0,'new_model_forwards':0,'optimizer_updates':0,'GPU_allocations':0}
    dump(folder.parent/(folder.name+'_INDEPENDENT_AUDIT.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('checkpoint');p.add_argument('harness');a=p.parse_args()
    print(json.dumps(audit(a.folder,a.checkpoint,a.harness),indent=2))
