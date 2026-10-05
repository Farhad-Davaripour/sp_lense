"""Check reproducibility inputs, source identities, and published aggregate claims."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    provenance=json.loads((HERE/'SOURCE_PROVENANCE.json').read_text())
    for name,record in provenance['core'].items():
        raw=(HERE/'pipeline'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Measured core source changed: '+name)
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    cases=json.loads((HERE/'cases/pilot.json').read_text())
    result=json.loads((HERE/'results/pilot.json').read_text())
    assert [row['case']['id'] for row in cases]==[row['case_id'] for row in result['paired_cases']]
    continuation,pending=result['paired_cases']
    assert continuation['control']['same_episode_joint_success'] and continuation['treatment']['same_episode_joint_success']
    assert not pending['control']['current_answer_accepted'] and not pending['control']['actual_post_notice_finish']
    assert pending['control']['continuation_granted'] and not pending['control']['same_episode_joint_success']
    guided=pending['treatment']
    assert all(guided[key] for key in ('continuation_granted','current_answer_accepted','actual_post_notice_finish','remaining_active','same_episode_joint_success'))
    assert not guided['expiration']
    assert result['actual_API_attempts']==104 and result['completed_condition_API_calls']==83
    inputs=json.loads((HERE/'training/INPUT_SOURCES.json').read_text())
    assert len(inputs['inputs'])==3 and inputs['A_plain_content_matches_recorded_training_messages_tools_targets']
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.suffix in ('.safetensors','.npz','.pt') or path.name=='.env':
            raise ValueError('Private/binary artifact in source package')
        if path.suffix in ('.py','.md','.json','.txt'):
            text=path.read_text(encoding='utf-8')
            forbidden=('drive.'+'google.com/drive/','C:/Users/'+'farha/','C:\\Users\\'+'farha\\')
            if any(marker in text for marker in forbidden):
                raise ValueError('Private or machine-specific reference: '+str(path.relative_to(HERE)))
    print(json.dumps({'package_verified':True,'pilot_cases':2,'measured_core_files':len(provenance['core']),
                      'published_raw_trajectories':0,'training_public_input_references':3}))


if __name__=='__main__':
    main()
