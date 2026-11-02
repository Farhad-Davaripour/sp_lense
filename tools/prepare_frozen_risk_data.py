"""Prepare exact development-only risk-reference source; never launch compute."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,sha,dump,digest
from src.composite_source_archive import archive_source
from src.composite_prepared_transport import source_manifest,write_transport
from src.composite_frozen_risk_data_profile import MODE,PROFILE,NOTEBOOK,HUMAN,KAGGLE_HUMAN,SELECTION_SHA,HUMAN_SHA,expected_profile,validate_source
from src.composite_kaggle_v2_profile import identity
from src.composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST


def prepare(repo,commit,model_inputs,human_selection,kaggle_scope,output):
    if sha(human_selection)!=SELECTION_SHA or sha(kaggle_scope)!=HUMAN_SHA:
        raise PermissionError('Exact existing human OptionA/Kaggle scope required')
    original=Path(model_inputs)
    if sha(original/'MANIFEST.json')!=ORIGINAL_MODEL_MANIFEST:raise PermissionError('Original full model-file freeze required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);source=output/'bundle/source'
    archive_source(repo,commit,source);inputs=source/'inputs';inputs.mkdir(exist_ok=False)
    shutil.copyfile(original/'MANIFEST.json',inputs/'PARENT_MODEL_INPUT_MANIFEST.json')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(original/name)!=read(original/'MANIFEST.json')[name]:raise PermissionError('Model pin bytes changed')
        shutil.copyfile(original/name,inputs/name)
    shutil.copyfile(human_selection,inputs/HUMAN);shutil.copyfile(kaggle_scope,inputs/KAGGLE_HUMAN)
    cfg=read(source/'config/composite_generalization_amendment_v2.json');profile=expected_profile(commit,cfg)
    dump(inputs/PROFILE,profile);dump(inputs/'WORKSPACE.json',identity(NOTEBOOK))
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in sorted(inputs.iterdir()) if p.name!='MANIFEST.json'})
    manifest=source_manifest(source,profile,'RISK_DATA_PACKAGE_FILES.json')
    validate_source(source,commit)
    archive,notebook=write_transport(output,source,profile,manifest,'FROZEN_RISK_DATA_PACKAGE.zip',
                                   'research4_frozen_risk_data.ipynb','notebooks/composite_task_skill_data.py')
    receipt={'status':'prepared_CPU_risk_development_reference_not_submitted','mode':MODE,'phase':'risk_data_export',
      'source_commit':commit,'notebook':NOTEBOOK,'platform':'kaggle','account':'farhad89','destination':profile['destination'],
      'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(manifest),
      'profile_sha256':sha(inputs/PROFILE),'input_manifest_sha256':sha(inputs/'MANIFEST.json'),
      'notebook_file_sha256':sha(notebook),'development_roster_sha256':digest(profile['development_roster']),
      'expected_cases':36,'original_risk_trajectories_required':180,'scientific_GPU_count':0,
      'new_reference_cases_generated':False,'new_model_forwards':0,'new_optimizer_updates':0,
      'new_model_weights_downloaded':False,'reference_policy_unchanged':True,
      'behavior_fitting_or_selection_admitted':False,'scientific_data_admitted':False,
      'scientific_allocation_admitted_by_preparation':False,'heldout_content_read':False}
    dump(output/'PREPARATION.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['repo','commit','model-inputs','human-selection','kaggle-scope','output']:parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
