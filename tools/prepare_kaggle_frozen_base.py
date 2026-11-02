"""Package one original-base partition using shared source/transport primitives."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_kaggle_recognition import prepare as shared_prepare, digest
from src.composite_runtime import FrozenInputs
from src.composite_frozen_base import (MODE, AUTHORIZATION, PLAN, BOUNDS, ORIGINAL_INPUTS, PARENTS,
    roster, partition_plan, validate_transport, authorized_load_plan)
from src.composite_frozen_probe import METADATA
from src.composite_frozen_base_worker import DISK_POLICY

PARENT_FILES = {'recognition_closeout': 'RUN_361470090_CLOSEOUT.json',
                'recognition_audit': 'RUN_361470090_INDEPENDENT_AUDIT.json',
                'qualification_closeout': 'RUN_361504592_CLOSEOUT.json',
                'qualification_audit': 'RUN_361504592_INDEPENDENT_AUDIT.json'}


def prepare(repo, commit, frozen_inputs, transport, output, notebook, tensor_metadata,
            partition, parent_evidence, human_authorization, operational_v2=False):
    from src.io_utils import read
    from src.composite_frozen_base import EPISODES, PROFILE_FILE
    human = read(human_authorization)
    if (human.get('source') != 'direct latest human user message in this chat'
        or human.get('thread_id') != '01a1227a-ba0a-77f1-8020-9cf781e727e3'
        or human.get('human_message_verbatim') != "Remove the limits and just simply do the job, and don't ask, don't wait for my approval. Go ahead."):
        raise PermissionError('Bind the existing actual human continuation, not a proposal or automatic message')
    def build(source, pin):
        required=['src/composite_frozen_base_worker.py','src/composite_frozen_base_launch.py',
                  'src/composite_frozen_base_admission.py','src/composite_frozen_base_audit.py']
        if any(not (source/name).is_file() for name in required) or MODE not in (source/'notebooks/composite_kaggle.py').read_text():
            raise ValueError('The selected source predates the registered frozen-base execution endpoint')
        if pin['original_scientific_input_manifest_sha256'] != ORIGINAL_INPUTS:
            raise ValueError('The original scientific input freeze changed')
        data = FrozenInputs(transport, pin['transport_manifest_sha256']); validate_transport(data)
        cfg = read(source/'config/composite_controller_rank32_v2.json')
        plan = partition_plan(roster(data.get(EPISODES), cfg), partition)
        (source/'inputs'/PLAN).write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
        shutil.copyfile(tensor_metadata, source/'inputs'/METADATA)
        shutil.copyfile(human_authorization, source/'inputs/HUMAN_CONTINUATION.json')
        for key, expected in PARENTS.items():
            original = Path(parent_evidence)/PARENT_FILES[key]
            if digest(original) != expected: raise ValueError('The audited parent receipt changed')
            shutil.copyfile(original, source/'inputs'/(key+'.json'))
        suffix = 'frozen_base_validation_'+str(partition).zfill(2)
        pin.update(mode=MODE, **BOUNDS, scientific_visible_devices='0,1', disk_policy=DISK_POLICY,
            destination='/kaggle/working/research4_campaign/runs/'+commit+'/'+suffix,
            cache_root='/kaggle/temp/research4_frozen_probe_cache/'+commit+'/'+suffix,
            controller_root='/kaggle/working/research4_campaign/controllers/'+commit+'/'+suffix,
            hardware={'physical_allocation':'Kaggle_T4_pair','visible_devices':'0,1'}, partition=partition)
        authority = {'status':'explicit_human_continuation','authorized_by':'human_user',
            'human_message_evidence_locator':str(Path(human_authorization).resolve())+'#human_message_verbatim',
            'human_message_receipt_sha256':digest(human_authorization),
            'mode':MODE,'source_commit':commit,'notebook':notebook,
            'plan_sha256':digest(source/'inputs'/PLAN),'profile_sha256':digest(source/PROFILE_FILE),
            'transport_manifest_sha256':pin['transport_manifest_sha256'],
            'original_scientific_input_manifest_sha256':ORIGINAL_INPUTS,
            'scientific_visible_GPU_count':2,'maximum_new_stage_submissions':1, **BOUNDS,
            'optimizer_updates':0,'behavior_adapter_fitting':False,'heldout_content_read':False}
        if operational_v2:
            from src.composite_frozen_operational import (PROFILE,DOCUMENT,PARENT_PREFIX_AUDIT,
                PARENT_CLOSEOUT,PARENT_RELEASE,document,validate_parent_inputs)
            required=['src/composite_cuda_maintenance.py','src/composite_frozen_operational_admission.py',
                      'src/composite_frozen_continuation.py']
            if any(not (source/name).is_file() for name in required):
                raise ValueError('The selected source predates the explicit operational recovery implementation')
            value=document(commit,partition)
            (source/'inputs'/DOCUMENT).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
            for origin,name in [
                (Path(parent_evidence)/'RUN_361586756_PARTIAL_PREFIX_AUDIT.json',PARENT_PREFIX_AUDIT),
                (Path(parent_evidence)/'RUN_361586756_CLOSEOUT.json',PARENT_CLOSEOUT),
                (Path(parent_evidence)/'frozen_base_validation_8b868880_p03/bundle/source/RELEASE.json',PARENT_RELEASE)]:
                shutil.copyfile(origin,source/'inputs'/name)
            validate_parent_inputs(source/'inputs')
            pin['operational_profile']=PROFILE
            authority.update(operational_profile=PROFILE,
                operational_document_sha256=digest(source/'inputs'/DOCUMENT))
        (source/'inputs'/AUTHORIZATION).write_text(json.dumps(authority,indent=2)+'\n',encoding='utf-8')
        return pin, 'prepared_authorized_frozen_base_not_submitted'
    receipt = shared_prepare(repo, commit, frozen_inputs, transport, output, notebook, _stage_builder=build)
    source = Path(output)/'bundle/source'
    authorized_load_plan(source, commit, read(source/'config/composite_controller_rank32_v2.json')['qwen'], transport)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['repo','commit','frozen-inputs','transport','output','notebook','tensor-metadata','parent-evidence','human-authorization']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--partition',type=int,required=True)
    parser.add_argument('--operational-v2',action='store_true')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
