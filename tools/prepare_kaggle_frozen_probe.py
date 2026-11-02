"""Package the separate frozen qualification, default denied; never submit compute."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.prepare_kaggle_recognition import prepare as shared_prepare,digest
from src.composite_frozen_probe import (MODE,BOUNDS,CONDITIONS,ORIGINAL_INPUTS,TRANSPORT,
    AUTHORIZATION,METADATA,PROFILE_FILE,verify_original_model_pin,authorized_load_plan)
from src.composite_placement import candidate_plan,metadata_bounds
from src.composite_frozen_worker import DISK_POLICY


def prepare(repo,commit,frozen_inputs,transport,output,notebook,tensor_metadata,authorization=None):
    def build(source,pin):
        if pin['original_scientific_input_manifest_sha256']!=ORIGINAL_INPUTS or pin['transport_manifest_sha256']!=TRANSPORT:
            raise ValueError('The complete original input/transport freeze is mandatory')
        config=json.loads((source/'config/composite_controller_rank32_v2.json').read_text())['qwen']
        profile=json.loads((source/PROFILE_FILE).read_text())
        metadata=json.loads(Path(tensor_metadata).read_text())
        metadata_bounds(metadata,candidate_plan(config,profile))
        model_pin=json.loads((source/'inputs/QWEN_PIN.json').read_text())
        verify_original_model_pin(model_pin)
        shutil.copyfile(tensor_metadata,source/'inputs'/METADATA)
        pin.update(mode=MODE,**BOUNDS,scientific_visible_devices='0,1',disk_policy=DISK_POLICY,
            destination='/kaggle/working/research4_campaign/runs/'+commit+'/frozen_qualification_01',
            cache_root='/kaggle/temp/research4_frozen_probe_cache/'+commit+'/qualification_01',
            controller_root='/kaggle/working/research4_campaign/controllers/'+commit+'/qualification_01',
            hardware={'physical_allocation':'Kaggle_T4_pair','visible_devices':'0,1'})
        authority={'status':'proposal_not_authorized','authorized_by':None,
            'human_message_evidence_locator':None,'mode':MODE,'scope':'frozen_base_activation_analysis_only',
            'source_commit':commit,'notebook':notebook,'repository':config['repository'],'revision':config['revision'],
            'profile_sha256':digest(source/PROFILE_FILE),
            'original_scientific_input_manifest_sha256':ORIGINAL_INPUTS,'transport_manifest_sha256':TRANSPORT,
            'scientific_visible_GPU_count':2,'maximum_new_stage_submissions':1,**BOUNDS,
            'optimizer_updates':0,'behavior_adapter_fitting':False,'heldout_content_read':False,
            'training_probe_conditions':CONDITIONS,'complete_prompts_no_truncation':True}
        if authorization is not None:
            authority=json.loads(Path(authorization).read_text())
            if authority.get('pre_model_parent') is not None:
                pin['pre_model_parent']=authority['pre_model_parent']
        (source/'inputs'/AUTHORIZATION).write_text(json.dumps(authority,indent=2)+'\n')
        return pin,('prepared_authorized_frozen_probe_not_submitted' if authorization else
                    'proposal_requires_separate_human_two_device_exception')
    receipt=shared_prepare(repo,commit,frozen_inputs,transport,output,notebook,_stage_builder=build)
    if authorization is not None:
        source=Path(output)/'bundle/source'
        authorized_load_plan(source,commit,json.loads((source/'config/composite_controller_rank32_v2.json').read_text())['qwen'])
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['repo','commit','frozen-inputs','transport','output','notebook','tensor-metadata']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--authorization')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
