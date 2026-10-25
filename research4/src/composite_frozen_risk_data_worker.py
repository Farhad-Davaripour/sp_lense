"""CPU development-risk references only; no neural model, fitting or submit API."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback
from .io_utils import read,sha,digest
from .composite_frozen_risk_data_profile import validate_source,MODE,BOUNDS
from .composite_frozen_risk_runtime import data_scope
from .composite_generalized_world import GeneralizedScene
from .composite_generalized_teacher import teacher_episode
from .composite_generalized_audit import audit_cohort,native_budget_audit
from .composite_v2_data_worker import Writer,metadata_tokenizers
from .composite_exports import make_evidence_archive


def run(root,commit):
    context=validate_source(root,commit);context.require_runtime();profile=context.profile();cfg=context.cfg()
    output=Path(profile['destination']);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def guard():
        if time.monotonic()-start>=BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']:
            raise TimeoutError('Protected risk-reference export deadline')
    writer=Writer(output,guard,BOUNDS['maximum_raw_export_bytes'],BOUNDS['terminal_metadata_reserve_bytes'])
    receipt={'stage':MODE,'phase':'risk_data_export','source_commit':commit,'status':'running','model_forwards':0,
      'optimizer_updates':0,'GPU_required':False,'model_weights_downloaded':False,'heldout_content_read':False,
      'public_fixture_test_mode':False,'automatic_scientific_retries':0,'completed_roles':[],
      'reference_policy_unchanged':True,'reference_actions_may_include_existing_bounded_continuity_transitions':True,
      'behavior_fitting_or_selection_admitted':False,'risk_reference_candidate_complete':False,
      'complete_case_exports':0}
    try:
        writer.json('PRECALL_FREEZE.json',{'source_commit':commit,'profile':profile,'study_configuration':cfg,
          'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),
          'source_release_sha256':sha(Path(root)/'RELEASE.json')})
        from .composite_frozen_worker import protected_alarm
        with protected_alarm(BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']),data_scope(context):
            with tempfile.TemporaryDirectory(prefix='research4-risk-reference-metadata-',dir='/kaggle/temp') as cache:
                qwen,bert=metadata_tokenizers(root,cache,writer)
                scenes=profile['development_roster'];episodes=[]
                for row in scenes:
                    guard();episode=teacher_episode(GeneralizedScene(**row))
                    writer.json('development/case_exports/'+row['case_id']+'.json',episode);episodes.append(episode)
                    receipt['complete_case_exports']=len(episodes)
                if [e['scene'] for e in episodes]!=scenes:raise PermissionError('Exact original full development roster required')
                rows=[{**state,'split':'development'} for episode in episodes for state in episode['states']]
                if not rows or len(rows)>900:raise ValueError('Every complete reference state must fit its original bound; never truncate')
                for row in rows:
                    if any(value in row['text'] for value in [row['case_id'],row['group_id'],row['mechanism'],row['task_structure'],'teacher_action','gate_target']):
                        raise PermissionError('Evaluator/reference metadata leaked into a visible observation')
                files={name:writer.json('development/'+name,value)['sha256'] for name,value in [
                  ('SCENES.json',scenes),('EPISODES.json',episodes),('DETECTION_ROWS.json',rows)]}
                manifest={'version':profile['world_version'],'role':'development','source_commit':commit,
                  'groups':4,'episodes':36,'states':len(rows),'case_ids':[s['case_id'] for s in scenes],
                  'group_ids':sorted({s['group_id'] for s in scenes}),'files':files,
                  'reference_actions_for_replay_only':True,'behavior_fitting_or_selection_admitted':False,
                  'scientific_dataset_admitted':False,'heldout_content_read':False,'model_forwards':0,'optimizer_updates':0}
                writer.json('development/COHORT_MANIFEST.json',manifest)
                audit=audit_cohort(episodes,rows,manifest,cfg,reference_replay=True)
                budget=native_budget_audit(rows,qwen,bert)
                writer.json('development/REFERENCE_AUDIT.json',audit);writer.json('development/NATIVE_BUDGET.json',budget)
                if budget['status']!='complete_native_budget_audit_pass' or budget['states']!=len(rows):
                    raise ValueError('Full original native/context budget admission failed; no reference or question reduction')
                release={'version':MODE,'source_commit':commit,'role':'development','world_version':profile['world_version'],
                  'cases':36,'original_risk_trajectories_required':180,'rows':len(rows),
                  'scenes_sha256':digest(scenes),'episodes_sha256':digest(episodes),'rows_sha256':digest(rows),
                  'reference_audit_sha256':digest(audit),'native_budget_sha256':digest(budget),
                  'scientific_semantic_core_sha256':profile['scientific_semantic_core_sha256'],
                  'reference_actions_present_for_replay_only':True,'behavior_fitting_or_selection_admitted':False,
                  'frozen_neural_evaluation_performed':False,'heldout_content_read':False}
                writer.json('RISK_DATA_RELEASE.json',release)
                receipt.update(status='complete',completed_roles=['development'],risk_reference_candidate_complete=True)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        writer.json('FAILURE.json',{'traceback':traceback.format_exc()},terminal=True)
    finally:
        receipt.update(seconds=time.monotonic()-start,worker_exiting=True,provider_release='external_controlled_exit_required')
        writer.json('RUN_RECEIPT.json',receipt,terminal=True)
    archive=output.with_name(output.name+'_EVIDENCE.zip')
    proof=make_evidence_archive(output,archive,BOUNDS['maximum_raw_export_bytes'])
    return {**receipt,**proof,'archive':str(archive)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    value=run(**vars(parser.parse_args()));print(json.dumps(value))
    if value['status']!='complete':raise SystemExit(1)
