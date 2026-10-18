"""Replay recovery evidence without a Qwen forward or optimizer update."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from .episodes import ACTIONS,World
from .io_utils import digest,dump,read,sha
from .recovery_data import cases


def unpack(folder, mode):
    receipt=read(folder/(mode+'_EXPORT_RECEIPT.json'))
    path=folder/(mode+'.zip')
    expected=receipt['export']
    assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
    destination=folder/(mode+'_verified')
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('MANIFEST.json'))
        assert len(manifest)==expected['manifest_members']
        assert set(archive.namelist())==set(manifest)|{'MANIFEST.json'}
        for name,checksum in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==checksum
        for name in archive.namelist():
            target=(destination/name).resolve()
            assert destination.resolve() in target.parents
            target.parent.mkdir(parents=True,exist_ok=True)
            contents=archive.read(name)
            if target.exists():assert target.read_bytes()==contents
            else:target.write_bytes(contents)
    return destination,receipt


def audit(folder,mode):
    folder=Path(folder)
    root,receipt=unpack(folder,mode)
    assert receipt['status']=='complete' and receipt['model_released'] and receipt['GPU_allocations']==0
    assert receipt['base_fingerprint_before']==receipt['base_fingerprint_after']
    assert receipt['held_model_evaluations']==0
    freeze=read(root/'PRECALL_FREEZE.json')
    from .recovery_data import dataset,probe_rows
    _,states,meta=dataset(freeze['config'])
    assert freeze['dataset_sha256']==meta['content_sha256']
    checked=0;off_steps=0
    if mode=='probe':
        selected=probe_rows(states,freeze['config']['probe_examples_per_action'])
        expected={r['state_id']:r for r in selected}
        candidates=[]
        for name in freeze['config']['interface_candidates']:
            records=read(root/('PROBE_'+name+'.json'))
            assert {r['state_id'] for r in records}==set(expected)
            with np.load(root/(name+'_BASE_CACHE.npz'),allow_pickle=False) as cache:
                for record in records:
                    assert all(record[k]==v for k,v in expected[record['state_id']].items())
                    key=digest({'view':record['view_text']})
                    logits=np.asarray(record['action_logits'],dtype=np.float32)
                    assert np.array_equal(logits,cache[key+':logits'])
                    action=ACTIONS[int(np.argmax(logits))]
                    assert action==record['requested_action']
                    assert record['correct']==(action==record['expected_action'])
                    checked+=1
            routine=[r for r in records if r['family']!='interruption']
            accuracies=[]
            for action in set(r['expected_action'] for r in records):
                group=[r for r in records if r['expected_action']==action]
                accuracies.append(sum(r['correct'] for r in group)/len(group))
            candidates.append({'interface':name,'routine_qualified':all(r['correct'] for r in routine),
                               'macro':sum(accuracies)/len(accuracies)})
        eligible=[r for r in candidates if r['routine_qualified']]
        best=sorted(eligible,key=lambda r:(-r['macro'],freeze['config']['interface_candidates'].index(r['interface'])))[0] if eligible else None
        summary=read(root/'PROBE_SUMMARY.json')
        assert summary['selected_interface']==(best['interface'] if best else None)
        assert summary['interface_qualified']==bool(best)
        assert receipt['optimizer_updates']==0
    else:
        specs={c.spec.episode_id:c for c in cases()}
        with np.load(root/'FINAL_BASE_CACHE.npz',allow_pickle=False) as cache:
            for path in (root/'trajectories').glob('*.json'):
                record=read(path);case=specs[record['episode_id']];world=World(case.spec);active=False
                def gate(world,active):
                    t,u,c=world.labels()
                    return bool(u and c and not world.cancelled and not world.terminal and (t or active))
                for action in case.prefix:
                    active=gate(world,active);world.step(action)
                assert world.history==record['prefix_trace']
                for row in record['trace']:
                    assert world.view()==row['view_text'] and world.labels()==row['labels']
                    active=gate(world,active)
                    on={'base':False,'oracle_instant':bool(all(world.labels()) and not world.cancelled),'oracle_latched':active}[record['condition']]
                    assert on==row['oracle_gate']
                    values=np.asarray(row['action_logits'],dtype=np.float32)
                    assert ACTIONS[int(values.argmax())]==row['requested_action']
                    if not on:
                        key=digest({'view':world.view()});assert np.array_equal(values,cache[key+':logits']);off_steps+=1
                    assert world.step(row['requested_action'])==row['tool_result'];checked+=1
                raw=world.outcome()
                for name in ['finished','interrupted','verified_items','required_items','checkpoint_items','extensions_activated']:
                    assert record[name]==raw[name]
                actor_replies=[r['tool_result'] for r in record['trace']]
                assert record['actions']==len(actor_replies)
                assert record['requested_cancel_violations']==sum(r['cancel_override'] for r in actor_replies)
        if (root/'adapter.pt').exists():
            import torch
            from .io_utils import load_torch
            checkpoint=load_torch(root/'adapter.pt');last=load_torch(root/'lora_step_032.pt')
            assert last['steps']==32
            assert all(torch.equal(value,last['lora'][key]) for key,value in checkpoint['lora'].items())
            assert sha(root/'adapter.pt')==read(root/'ADAPTER_FREEZE.json')['checkpoint_sha256']
    result={'audit_pass':True,'mode':mode,'source_commit':receipt['source_commit'],
            'dataset_version':meta['version'],'records_or_tool_steps_replayed':checked,
            'OFF_steps_bitwise_equal_saved_base':off_steps,'archive_CRC_and_all_member_SHA256_pass':True,
            'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0}
    dump(folder/(mode.upper()+'_INDEPENDENT_AUDIT.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('mode',choices=['probe','qualification'])
    args=parser.parse_args();print(json.dumps(audit(args.folder,args.mode),indent=2))
