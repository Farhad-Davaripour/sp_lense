"""Replay task policy, classifier probabilities and world outcomes without Qwen."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from .learners import Signals
from .semantic_policy import readout
from .io_utils import digest,load_torch,dump,read,sha
from .context_audit import replay,summarize
from .recovery_data import cases


def audit(folder,checkpoint):
    folder=Path(folder);saved=load_torch(checkpoint);heads=Signals(saved['feature_dimension'])
    heads.load_state_dict(saved['signals']);heads.eval().requires_grad_(False);torch.set_num_threads(2)
    receipt=read(folder/'RUN_RECEIPT.json');assert receipt['status']=='complete' and receipt['model_released']
    assert receipt['optimizer_updates_in_evaluation']==receipt['GPU_allocations']==0
    assert receipt['base_fingerprint_before']==receipt['base_fingerprint_after']
    fit_freeze=read(Path(checkpoint).parent/'FIT_FREEZE.json');original=load_torch(fit_freeze['source_checkpoint_path'])
    assert sha(fit_freeze['source_checkpoint_path'])==saved['original_checkpoint_sha256']
    for index in [0,1]:
        for suffix in ['weight','bias']:assert torch.equal(saved['signals'][f'heads.{index}.{suffix}'],original['signals'][f'heads.{index}.{suffix}'])
    assert all(torch.equal(value,original['lora'][key]) for key,value in saved['lora'].items())
    assert torch.equal(saved['mean'],original['mean']) and torch.equal(saved['scale'],original['scale'])
    records=read(folder/'READOUT_RECORDS.json');converted={};maximum=0.
    with np.load(folder/'BASE_CACHE.npz',allow_pickle=False) as cache:
        for key,row in records.items():
            view=row['view_text'];feature=torch.tensor(cache[digest({'view':view})+':feature'])
            with torch.no_grad():p=heads((feature-saved['mean'])/saved['scale']).sigmoid().tolist()
            maximum=max(maximum,max(abs(a-b) for a,b in zip(p,row['signal_probabilities'])))
            action,values,details=readout(row['native_action_logits'],p,view)
            assert action==row['policy_action'] and values==row['policy_scores'] and details['allowed_actions']==row['allowed_actions']
            if not row['on']:assert np.array_equal(np.asarray(row['native_action_logits'],dtype=np.float32),cache[digest({'view':view})+':logits'])
            converted[key]={'view_text':view,'on':row['on'],'strength':row['strength'],'action':action,'logits':values}
    assert maximum<1e-6
    specs={c.spec.episode_id:c for c in cases()};episodes=steps=off=0
    if (folder/'TRAINING_QUALIFICATION.json').exists():
        for strength,summary in read(folder/'TRAINING_QUALIFICATION.json').items():
            current=[read(p) for p in (folder/'training'/strength).glob('*.json')]
            assert len(current)==10 and summarize(current)==summary['summary']
            for record in current:
                n,o=replay(record,specs[record['episode_id']],float(strength),converted);steps+=n;off+=o;episodes+=1
    development={}
    if (folder/'DEVELOPMENT_FREEZE.json').exists():
        freeze=read(folder/'DEVELOPMENT_FREEZE.json');assert freeze['development_decisions_seen']==0
        for condition,expected in receipt['development'].items():
            current=[read(p) for p in (folder/'development').glob(condition+'_*.json')]
            assert len(current)==10 and summarize(current)==expected
            if condition!='native_base':
                for record in current:
                    n,o=replay(record,specs[record['episode_id']],freeze['selected_strength'],converted);steps+=n;off+=o;episodes+=1
            else:
                native_cache={}
                with np.load(folder/'BASE_CACHE.npz',allow_pickle=False) as cache:
                    from .episodes import ACTIONS
                    for record in current:
                        for row in record['trace']:
                            values=cache[digest({'view':row['view_text']})+':logits'].tolist()
                            key=digest({'view':row['view_text'],'on':False,'strength':0})
                            native_cache[key]={'view_text':row['view_text'],'on':False,'strength':0,
                                               'action':ACTIONS[int(np.argmax(values))],'logits':values}
                        n,o=replay(dict(record,condition='base'),specs[record['episode_id']],freeze['selected_strength'],native_cache)
                        steps+=n;off+=o;episodes+=1
            development[condition]=expected
        assert all(r['bitwise_equal'] for r in read(folder/'BYPASS_CHECKS.json'))
        all_records=[read(p) for p in (folder/'development').glob('*.json')]
        base_controls={r['episode_id']:r for r in all_records if r['condition']=='base' and r['family']!='interruption'}
        assert len(base_controls)==5
        matched_controls=0
        for condition in ['oracle_instant','oracle_latched']:
            for record in [r for r in all_records if r['condition']==condition and r['family']!='interruption']:
                baseline=base_controls[record['episode_id']]
                assert record['trace']==baseline['trace'] and record['prefix_trace']==baseline['prefix_trace'];matched_controls+=1
        assert matched_controls==10
    outer=read(folder.parent/(folder.name+'_EXPORT_RECEIPT.json'));archive=folder.parent/(folder.name+'.zip')
    assert sha(archive)==outer['export']['sha256'] and archive.stat().st_size==outer['export']['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    result={'audit_pass':True,'qualified':receipt['qualified'],'source_commit':receipt['source_commit'],
            'learned_policy_records_recomputed':len(records),'maximum_probability_difference':maximum,
            'complete_semantic_episodes_replayed':episodes,'tool_steps_replayed':steps,'OFF_steps_rechecked':off,
            'development':development,'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0}
    result['archive_and_every_member_SHA256_pass']=True
    dump(folder.parent/(folder.name+'_SEMANTIC_AUDIT.json'),result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('checkpoint');args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.checkpoint),indent=2))
