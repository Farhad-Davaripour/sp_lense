"""Read the complete admitted development archive; never create new cases."""
import hashlib,json
from .io_utils import digest
from .composite_v2_model_audit import EvidenceFiles
from .composite_v2_data_contract import source_fingerprint
from .composite_v2_development_contract import validate_roster


def load_development_data(raw,context,*,reference_rows=False):
    from .composite_generalized_data import require_runtime
    require_runtime()
    stage=context['stage'];files=EvidenceFiles(raw,stage['data_export']['sha256'],256*1024**2)
    try:
        release=files.json('DATA_RELEASE.json');receipt=files.json('RUN_RECEIPT.json')
        if (receipt.get('status')!='complete' or receipt.get('scientific_dataset_admitted') is not True
            or receipt.get('public_fixture_test_mode') is not False
            or release.get('heldout_content_read') is not False
            or release['data_admission'].get('training_and_development_admitted') is not True):
            raise PermissionError('Only a complete genuine admitted nonsealed data archive')
        if digest(release)!=stage['data_release_sha256'] or release['workspace']!=context['workspace']:
            raise PermissionError('Data release belongs to another exact workspace/freeze')
        if release['world_source_files']!=source_fingerprint(context['root']):raise PermissionError('World/interface source changed')
        cohort=release['cohorts']['development']
        def member(name):
            value=files.get('development/'+name)
            if hashlib.sha256(value).hexdigest()!=cohort['files'][name]:raise PermissionError('Development file changed: '+name)
            return json.loads(value)
        scenes=member('SCENES.json')
        keys=list(stage['development_roster'][0])
        roster=[{k:value[k] for k in keys} for value in scenes]
        validate_roster(roster,context['design'])
        if digest(roster)!=digest(stage['development_roster']):raise PermissionError('Exact complete case roster differs')
        result={'release':release,'scenes':scenes,'rows':None,'reference_actions_read':False}
        if reference_rows:
            rows=member('DETECTION_ROWS.json');episodes=member('EPISODES.json')
            from .composite_generalized_audit import audit_cohort
            audit_cohort(episodes,rows,files.json('development/COHORT_MANIFEST.json'),context['design'],reference_replay=True)
            if len(rows)>context['partition_plan']['maximum_reference_detector_states']:
                raise PermissionError('Complete cohort exceeds predeclared state bound; never truncate')
            result.update(rows=rows,reference_actions_read=True)
        return result
    finally:files.close()
