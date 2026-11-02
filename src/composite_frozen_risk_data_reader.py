"""Read-only development-risk reference audit; no model or policy fitting."""
import base64
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
from .io_utils import read,sha,digest
from .composite_frozen_risk_data_profile import validate_source,PROFILE,MODE,BOUNDS
from .composite_v2_model_audit import EvidenceFiles

_ARCHIVE_KEY=object()
_KEY=object()


class RiskDataArchive:
    def __init__(self,context,files,closure,key=None):
        if key is not _ARCHIVE_KEY:raise PermissionError('Bind the actual complete raw/source/provider archive')
        self._key=key;self.context=context;self.files=files;self.closure=closure
        self.reference_actions={e['scene']['case_id']:[a['action'] for a in e['actions']]
                                for e in files.json('development/EPISODES.json')}
        if (set(self.reference_actions)!={s['case_id'] for s in context.profile()['development_roster']}
                or any(len(a)>24 for a in self.reference_actions.values())):
            raise PermissionError('Exact finite existing reference actions are required for replay')


class RiskDevelopmentData:
    def __init__(self,value,scenes,rows,key=None):
        if key is not _KEY:raise PermissionError('Read the complete actual development-only reference archive')
        self._key=key;self._value=deepcopy(value);self._scenes=deepcopy(scenes);self._rows=deepcopy(rows)
    def receipt(self):return deepcopy(self._value)
    def scenes(self):return deepcopy(self._scenes)
    def reference_rows(self):return deepcopy(self._rows)


def require_data(value):
    if not isinstance(value,RiskDevelopmentData) or value._key is not _KEY:
        raise PermissionError('Genuine raw-audited/released risk development data required')
    report=value.receipt()
    if (report.get('risk_development_data_admitted') is not True or report.get('owned_compute_released') is not True
            or report.get('complete_cases')!=36 or report.get('role')!='development'
            or report.get('behavior_fitting_or_selection_admitted') is not False or report.get('heldout_content_read') is not False):
        raise PermissionError('Complete development-only raw/native/provider data admission required')
    return report


@contextmanager
def verified_metadata_tokenizers(files,root):
    from .composite_frozen_probe import verify_metadata_bytes
    with tempfile.TemporaryDirectory(prefix='r4-risk-reference-metadata-') as directory:
        result={}
        for name,pinfile in [('qwen','QWEN_PIN.json'),('bert','MODERNBERT_PIN.json')]:
            pin=read(Path(root)/'inputs'/pinfile);proof=files.json('metadata/'+name+'_FILES.json')
            encoded=files.json('metadata/'+name+'_BYTES.json');verify_metadata_bytes(pin,proof,encoded)
            target=Path(directory)/name;target.mkdir()
            for filename,value in encoded.items():
                path=(target/filename).resolve()
                if not path.is_relative_to(target.resolve()):raise PermissionError('Only original tokenizer/config metadata may be reconstructed')
                path.write_bytes(base64.b64decode(value,validate=True))
            from transformers import AutoTokenizer
            result[name]=AutoTokenizer.from_pretrained(target,local_files_only=True,trust_remote_code=False)
        yield result['qwen'],result['bert']


def audit_data(raw,raw_sha256,root,commit,closure):
    context=validate_source(root,commit);profile=context.profile()
    files=EvidenceFiles(raw,raw_sha256,BOUNDS['maximum_raw_export_bytes'])
    try:
        from .composite_task_skill_model_audit import KaggleClosure,_KEY as CLOSURE_KEY
        if (not isinstance(closure,KaggleClosure) or closure._key is not CLOSURE_KEY
                or closure._value['source_commit']!=commit or closure._value['notebook']!=profile['notebook']
                or closure._value['archive_sha256']!=raw_sha256 or closure._value['owned_compute_released'] is not True):
            raise PermissionError('Exact independent CPU reference-data provider release required')
        expected_code=(Path(root)/'notebooks/composite_task_skill_data.py').read_text(encoding='utf-8-sig').replace(
          'SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(Path(root).parent/'RISK_DATA_PACKAGE_FILES.json'))
        if closure._value['actual_code_cells_sha256']!=digest([expected_code]):raise PermissionError('Actual risk data notebook readback differs from source/input freeze')
        receipt=files.json('RUN_RECEIPT.json')
        expected={'status':'complete','source_commit':commit,'stage':MODE,'phase':'risk_data_export',
          'completed_roles':['development'],'model_forwards':0,'optimizer_updates':0,'GPU_required':False,
          'model_weights_downloaded':False,'heldout_content_read':False,'public_fixture_test_mode':False,
          'reference_policy_unchanged':True,'behavior_fitting_or_selection_admitted':False,
          'risk_reference_candidate_complete':True}
        if any(digest(receipt.get(k))!=digest(v) for k,v in expected.items()):
            raise PermissionError('Only a complete genuine development-only reference export is admissible')
        freeze=files.json('PRECALL_FREEZE.json')
        if digest(freeze)!=digest({'source_commit':commit,'profile':profile,'study_configuration':context.cfg(),
          'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),
          'source_release_sha256':sha(Path(root)/'RELEASE.json')}):raise PermissionError('Data pre-call source/profile/input freeze differs')
        scenes=files.json('development/SCENES.json');episodes=files.json('development/EPISODES.json')
        rows=files.json('development/DETECTION_ROWS.json');manifest=files.json('development/COHORT_MANIFEST.json')
        if digest(scenes)!=digest(profile['development_roster']) or [e['scene'] for e in episodes]!=scenes:
            raise PermissionError('All36 exact risk scenarios required; no role or identity resampling')
        if not rows or len(rows)>900 or any(r.get('split')!='development' for r in rows):
            raise PermissionError('Every complete reference state must remain development-only')
        required={'RUN_RECEIPT.json','PRECALL_FREEZE.json','RISK_DATA_RELEASE.json',
          'development/SCENES.json','development/EPISODES.json','development/DETECTION_ROWS.json',
          'development/COHORT_MANIFEST.json','development/REFERENCE_AUDIT.json','development/NATIVE_BUDGET.json'}
        required|={'metadata/'+name+'_'+kind+'.json' for name in ['qwen','bert'] for kind in ['FILES','BYTES']}
        required|={'development/case_exports/'+s['case_id']+'.json' for s in scenes}
        if set(files.manifest)!=required:raise PermissionError('Exact development-only raw coverage required; no training or sealed exports')
        for episode in episodes:
            if digest(files.json('development/case_exports/'+episode['scene']['case_id']+'.json'))!=digest(episode):
                raise PermissionError('Complete original per-case prefix differs from final data export')
        archive=RiskDataArchive(context,files,closure,_ARCHIVE_KEY)
        from .composite_frozen_risk_runtime import reference_replay_scope
        from .composite_generalized_audit import audit_cohort,native_budget_audit
        with reference_replay_scope(archive):
            audit=audit_cohort(episodes,rows,manifest,context.cfg(),reference_replay=True)
        if digest(audit)!=digest(files.json('development/REFERENCE_AUDIT.json')):raise PermissionError('Reference replay audit differs')
        with verified_metadata_tokenizers(files,root) as (qwen,bert):
            budget=native_budget_audit(rows,qwen,bert)
        if budget['status']!='complete_native_budget_audit_pass' or digest(budget)!=digest(files.json('development/NATIVE_BUDGET.json')):
            raise PermissionError('All complete original context/native budgets must pass without truncation')
        release=files.json('RISK_DATA_RELEASE.json')
        expected_release={'version':MODE,'source_commit':commit,'role':'development','world_version':profile['world_version'],
          'cases':36,'original_risk_trajectories_required':180,'rows':len(rows),
          'scenes_sha256':digest(scenes),'episodes_sha256':digest(episodes),'rows_sha256':digest(rows),
          'reference_audit_sha256':digest(audit),'native_budget_sha256':digest(budget),
          'scientific_semantic_core_sha256':profile['scientific_semantic_core_sha256'],
          'reference_actions_present_for_replay_only':True,'behavior_fitting_or_selection_admitted':False,
          'frozen_neural_evaluation_performed':False,'heldout_content_read':False}
        if digest(release)!=digest(expected_release):raise PermissionError('Risk reference release bindings differ')
        return RiskDevelopmentData({'reader':'option-a-risk-development-reference-raw-v1','source_commit':commit,
          'raw_archive_sha256':raw_sha256,'provider_run_id':closure._value['scriptVersionId'],
          'risk_development_data_admitted':True,'owned_compute_released':True,'role':'development',
          'complete_cases':36,'complete_reference_states':len(rows),'source_release':release,'release_sha256':digest(release),
          'native_budget':budget,'reference_audit':audit,'behavior_fitting_or_selection_admitted':False,
          'new_model_forwards':0,'new_optimizer_updates':0,'heldout_content_read':False},scenes,rows,_KEY)
    finally:files.close()
