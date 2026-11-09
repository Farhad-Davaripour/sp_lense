"""Separate version-two stage packets; no launch or current-run modification."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from game_all_layers_data import DATA_SHA, ROOT, PROJECT
from game_all_layers_selection import GRID, choose

MODEL_ID='mlx-community/Qwen3.8-27B-4bit'
REVISION='10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
MODEL_SHA='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
ADDITIONAL_INSTALLED_SOURCES={'mlx_lm/models/gated_delta.py':'f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428'}
CONFIG=PROJECT/'experiments/sp_lens_game/all_layers_v1/experiment.yaml'
PROTOCOL=CONFIG.with_name('PROTOCOL.md')
FILES=('game_all_layers_data.py','game_all_layers_loss.py','game_all_layers_selection.py','game_all_layers_select.py','game_all_layers_layout.py',
       'game_all_layers_plan.py','game_all_layers_worker.py','game_all_layers_supervisor.py',
       'game_training_renderer.py','owned_validation_process.py','qwen38_local.py','qwen38_config.json')
STAGES={'pilot_rank4':('pilot',4,0.00001,3,900),'pilot_rank8':('pilot',8,0.00001,3,900),
        **{name:('trial',rank,lr,48,7200)for name,rank,lr in GRID}}


def sha(path):
    if path.is_symlink()or not path.is_file():raise ValueError('Expected regular project file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def path_for(name):
    value=PurePosixPath(name)
    if value.is_absolute()or'..'in value.parts or not value.parts:raise ValueError('Unsafe project-relative path')
    current=PROJECT
    for part in value.parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink paths require review')
    if not current.resolve().is_relative_to(PROJECT.resolve())or any('onedrive'in p.casefold()for p in current.resolve().parts):
        raise ValueError('Outside local Research4 storage')
    return current


def read_closed_stage(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Supervised stage closeout bytes differ')
    closeout=json.loads(path.read_text())
    if (closeout.get('schema'),closeout.get('status'),closeout.get('cleanup_confirmed'))!= (
        'research4.sp-lens-game-all-layers-stage-closeout.v2','complete',True):raise ValueError('Verified owned-process closeout required')
    worker=path_for(closeout['worker_receipt']['path'])
    if sha(worker)!=closeout['worker_receipt']['sha256']:raise ValueError('Closed worker receipt differs')
    receipt=json.loads(worker.read_text())
    if (receipt.get('status'),receipt.get('stage'),receipt.get('plan_sha256'))!=('complete',closeout['stage'],closeout['plan_sha256']):
        raise ValueError('Worker and supervised closeout differ')
    return receipt


def read_plan(path,expected_sha=None):
    if not path.resolve().is_relative_to(ROOT.resolve())or path.is_symlink():raise ValueError('Stage packet must be experiment-local')
    if expected_sha is not None and sha(path)!=expected_sha:raise ValueError('Stage packet bytes differ')
    plan=json.loads(path.read_text())
    if (plan.get('schema'),plan.get('model'),plan.get('revision'),plan.get('base_manifest_sha256'),plan.get('dataset_manifest_sha256'))!= (
        'research4.sp-lens-game-all-layers-stage.v2',MODEL_ID,REVISION,MODEL_SHA,DATA_SHA):raise ValueError('All-layer identity differs')
    if plan.get('additional_installed_source_sha256')!=ADDITIONAL_INSTALLED_SOURCES:raise ValueError('GDN primitive source binding differs')
    baseline_path=path_for(plan['baseline_completion']['path'])
    if sha(baseline_path)!=plan['baseline_completion']['sha256']:raise ValueError('Last-layer completion differs')
    baseline=json.loads(baseline_path.read_text())
    if (baseline.get('schema'),baseline.get('status'),baseline.get('completed_steps'),baseline.get('expected_steps'),
        baseline.get('all_finite'),baseline.get('nonzero_lora_b'),baseline.get('base_files_unchanged'))!= (
        'research4.sp-lens-game-completion.v1','passed',768,768,True,True,True):raise ValueError('Finish and verify the last-layer run first')
    if (baseline.get('model',{}).get('manifest_sha256'),baseline.get('artifacts',{}).get('dataset_manifest',{}).get('sha256'))!= (
        MODEL_SHA,DATA_SHA):raise ValueError('Last-layer model/data identity differs')
    if plan.get('stage')=='final':
        selection=json.loads(path_for(plan['selection']['path']).read_text())
        if sha(path_for(plan['selection']['path']))!=plan['selection']['sha256']or selection.get('schema')!='research4.sp-lens-game-tuning-selection.v2':
            raise ValueError('Final selection differs')
        winner=selection['winner']; expected=('final',winner['rank'],winner['learning_rate'],768,43200)
        if (winner['rank'],winner['learning_rate'])not in {(r,lr)for _,r,lr in GRID}:raise ValueError('Winner is outside declared grid')
    else:
        if plan.get('stage')not in STAGES:raise ValueError('Unapproved stage')
        expected=STAGES[plan['stage']]
    if tuple(plan.get(k)for k in ('mode','rank','learning_rate','iterations','max_wall_seconds'))!=expected:raise ValueError('Stage recipe differs')
    constants={'num_layers':64,'scale':2.0,'dropout':0.0,'batch_size':1,'max_seq_length':1024,'gradient_checkpointing':True,
        'gradient_accumulation_steps':1,'seed':20261108,'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,
        'global_wall_seconds':22*3600,'retries':0,'initialization':'fresh_pinned_base','loss_version':'shifted_suffix_P_le_position_lt_L.v2'}
    if any(plan.get(k)!=v for k,v in constants.items()):raise ValueError('All-layer constants differ')
    if set(plan.get('source_sha256',{}))!=set(FILES):raise ValueError('All-layer source inventory differs')
    for name,expected_sha in plan['source_sha256'].items():
        if sha(PROJECT/'mac_studio'/name)!=expected_sha:raise ValueError('Source differs: '+name)
    for key in ('protocol','config','cohort'):
        if sha(path_for(plan[key]['path']))!=plan[key]['sha256']:raise ValueError(key+' bytes differ')
    if not plan.get('catalog_sha256'):raise ValueError('Scenario YAML bindings required')
    for name,value in plan['catalog_sha256'].items():
        if sha(path_for(name))!=value:raise ValueError('Scenario YAML bytes differ')
    cohort=json.loads(path_for(plan['cohort']['path']).read_text());root=path_for(plan['cohort']['path']).parent
    if cohort.get('dataset_manifest_sha256')!=DATA_SHA:raise ValueError('Cohort source differs')
    if sha(path_for(cohort['dataset_manifest_path']))!=DATA_SHA:raise ValueError('Original r4 manifest differs')
    for name,value in cohort['files'].items():
        if Path(name).name!=name or sha(root/name)!=value:raise ValueError('Prepared cohort file differs')
    previous=[]
    for entry in plan.get('required_receipts',[]):
        source=path_for(entry['path'])
        if sha(source)!=entry['sha256']:raise ValueError('Required preceding stage receipt differs')
        value=json.loads(source.read_text())
        if value.get('schema')=='research4.sp-lens-game-all-layers-resource-review.v2':
            if value.get('status')!='complete':raise ValueError('Resource review incomplete')
            previous.append(value)
        else:previous.append(read_closed_stage(source,entry['sha256']))
    stages={item.get('stage')for item in previous}
    if plan['stage']=='pilot_rank8'and'pilot_rank4'not in stages:raise ValueError('Rank-4 compatibility pilot must pass first')
    if plan['mode']in('trial','final'):
        if not {'pilot_rank4','pilot_rank8'}<=stages:raise ValueError('Both rank pilots must complete before a sweep/final run')
        reviews=[item for item in previous if item.get('schema')=='research4.sp-lens-game-all-layers-resource-review.v2']
        if len(reviews)!=1 or reviews[0].get('sweep_approved')is not True or reviews[0].get('global_wall_seconds')!=22*3600:
            raise ValueError('Measured resource/time review required before the sweep')
    if plan['mode']=='final':
        trials=[]
        for trial in selection['trial_receipts']:
            receipt_path=path_for(trial['path'])
            if sha(receipt_path)!=trial['sha256']:raise ValueError('Selection trial receipt differs')
            closed=read_closed_stage(path_for(trial['closeout']['path']),trial['closeout']['sha256'])
            if closed!=json.loads(receipt_path.read_text()):raise ValueError('Selection trial was not the supervised completed run')
            trials.append(closed)
        recomputed=choose(trials,cohort['development_ids'],DATA_SHA,cohort['development_target_tokens'])
        if recomputed['winner']!=selection['winner']:raise ValueError('Final winner differs from raw development likelihoods')
    return plan,cohort,root


def draft(stage,cohort_path,out,baseline_completion,selection=None,required_receipts=()):
    if stage=='final':
        if selection is None:raise ValueError('Final stage requires the separately reviewed selection')
        chosen=json.loads(selection.read_text())['winner'];recipe=('final',chosen['rank'],chosen['learning_rate'],768,43200)
    elif stage in STAGES:recipe=STAGES[stage]
    else:raise ValueError('Unapproved stage')
    plan={'schema':'research4.sp-lens-game-all-layers-stage.v2','external_model_name':'sp-lense-adaptors-all-layers',
        'stage':stage,**dict(zip(('mode','rank','learning_rate','iterations','max_wall_seconds'),recipe)),
        'model':MODEL_ID,'revision':REVISION,'base_manifest_sha256':MODEL_SHA,'dataset_manifest_sha256':DATA_SHA,
        'additional_installed_source_sha256':ADDITIONAL_INSTALLED_SOURCES,
        'num_layers':64,'scale':2.0,'dropout':0.0,'batch_size':1,'max_seq_length':1024,'gradient_checkpointing':True,
        'gradient_accumulation_steps':1,'seed':20261108,'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,
        'global_wall_seconds':22*3600,'retries':0,'initialization':'fresh_pinned_base','loss_version':'shifted_suffix_P_le_position_lt_L.v2',
        'source_sha256':{name:sha(PROJECT/'mac_studio'/name)for name in FILES},
        'protocol':{'path':rel(PROTOCOL),'sha256':sha(PROTOCOL)},'config':{'path':rel(CONFIG),'sha256':sha(CONFIG)},
        'cohort':{'path':rel(cohort_path),'sha256':sha(cohort_path)},
        'baseline_completion':{'path':rel(baseline_completion),'sha256':sha(baseline_completion)},
        'required_receipts':[{'path':rel(p),'sha256':sha(p)}for p in required_receipts],
        'scope':'Draft packet only; separate exact-hash reviewed --execute required. No retries, cloud or previous adapter.'}
    source_manifest=json.loads(path_for(json.loads(cohort_path.read_text())['dataset_manifest_path']).read_text())
    plan['catalog_sha256']={'scenarios/sp_lens_game_v1/authorized_management_v1_r4/'+name:value
        for name,value in source_manifest['yaml_file_sha256'].items()}
    if selection is not None:plan['selection']={'path':rel(selection),'sha256':sha(selection)}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x')as stream:json.dump(plan,stream,indent=2)
    read_plan(out)
    return {'path':str(out),'sha256':sha(out)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True)
    parser.add_argument('--cohort',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-completion',type=Path,required=True)
    parser.add_argument('--selection',type=Path);parser.add_argument('--require-receipt',type=Path,action='append',default=[])
    args=parser.parse_args();print(json.dumps(draft(args.stage,args.cohort,args.out,args.baseline_completion,args.selection,args.require_receipt),indent=2))
