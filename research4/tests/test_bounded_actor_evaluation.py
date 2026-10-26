"""Public/fake source, native replay and lifecycle tests; no scientific cohorts/models."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict,replace
from types import SimpleNamespace
from pathlib import Path
import base64,json,tempfile,unittest,zipfile
from unittest.mock import patch
from src.io_utils import dump,sha,digest
from src.composite_bounded_behavior_evaluation_profile import MODE,OWNER,execution_plan,PROBE_REGISTRATION
from src.composite_bounded_behavior_evaluation_runtime import public_replay_scope
from src.composite_bounded_behavior_evaluation_collection import world_for,terminal_record


class BoundedActorEvaluationTests(unittest.TestCase):
    def test_actor_provider_closure_binds_exact_notebook_version_and_saved_code(self):
        from src.composite_bounded_behavior_evaluation_audit import kaggle_closure,KaggleClosure
        notebook='farhad89/research4-bounded-actor-behavior-v4'
        binding={'mode':MODE,'endpoint':'behavior','phase':'bounded_actor_behavior','notebook':notebook,
          'source_commit':'a'*40,'scriptVersionId':123,'owner_thread_id':OWNER}
        controller={**binding,'status':'worker_exited','exit_code':0,'timed_out':False,
          'automatic_scientific_retries':0,'heldout_content_read':False,
          'child':{'child_started':True,'child_exit_confirmed':True,'exit_code':0},
          'raw_prefix_rescue':{'status':'owned_scratch_absent'}}
        terminal=notebook+' scriptVersionId=123 successfully ran after 1s'
        release='https://www.kaggle.com/code/'+notebook+' 0  Active Events Draft Session off'
        self.assertIsInstance(kaggle_closure(binding,'b'*64,terminal,release,controller,['public code'],['public code']),KaggleClosure)
        for wrong_terminal,wrong_release in [(terminal.replace('=123','=1234'),release),
          (terminal,release.replace(notebook,notebook+'-copy'))]:
            with self.assertRaises(PermissionError):
                kaggle_closure(binding,'b'*64,wrong_terminal,wrong_release,controller,['public code'],['public code'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding,'b'*64,terminal,release,controller,['altered code'],['public code'])

    def test_frozen_worker_lifecycle_never_claims_a_fit_and_preserves_partial_counts(self):
        from contextlib import nullcontext
        import time
        from tests.test_task_skill_lifecycle import Store
        from src.composite_owned_artifacts import OwnedArtifacts
        from src.composite_bounded_behavior_evaluation_worker import run
        from src.composite_bounded_behavior_evaluation_profile import stage_prefix
        source='a'*40;stage={'version':MODE,'endpoint':'behavior','source_commit':source,
          'phase':'bounded_actor_behavior','partition':{'index':0},'worker_seconds':100,'export_reserve_seconds':10,
          'source_data_archive_sha256':'b'*64,'maximum_backbone_forwards':30000}
        for failed in [False,True]:
            with tempfile.TemporaryDirectory() as folder:
                store=Store()
                class Services:
                    clock=staticmethod(time.monotonic)
                    def authorize(self,*args):return {'stage':stage,'full_config':{}},[]
                    def owner(self,context):return OwnedArtifacts(store,stage_prefix(stage),source,base=folder,
                      maximum_bytes=8*1024**2,bounded_evaluation=True)
                    def runtime(self,context):return {'public_fixture':True}
                    def resources(self):pass
                    def alarm(self,seconds):return nullcontext()
                    def load(self,context,owner,guard,forward):
                        if failed:forward('attempt');raise RuntimeError('public fake incomplete forward')
                        return object()
                    def fit(self,*args):return {'optimizer_updates':0,'completed_trajectories':2}
                    def release(self):pass
                    def remove_owned_cache(self,owner):return {'owned_download_cache_removed':True}
                result=run('public_fake_source',source,_services=Services())
            self.assertTrue(result['public_fixture_test_mode']);self.assertTrue(result['evaluation_only'])
            self.assertIsNone(result['training_objective']);self.assertFalse(result['full_fit_completed'])
            self.assertFalse(result['qualification_admitted'])
            if failed:
                self.assertIsNone(result['optimizer_updates'])
                self.assertEqual(result['forward_counts'],{'attempted':1,'completed':0})
            else:self.assertEqual(result['optimizer_updates'],0)

    def test_request_constructor_and_clock_declarations_are_part_of_fit_semantic_freeze(self):
        import shutil
        from src.composite_bounded_behavior_evaluation_semantics import fingerprint,require_frozen
        source=Path(__file__).parents[1]
        expected=fingerprint(source)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shutil.copytree(source/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
            require_frozen(expected,root)
            world=root/'src/composite_bounded_behavior_world.py';original=world.read_text()
            self.assertIn('self.request_onset = 0 if self.initially_complete else 2',original)
            world.write_text(original.replace('self.request_onset = 0 if self.initially_complete else 2',
              'self.request_onset = 0 if self.initially_complete else 3'))
            with self.assertRaises(PermissionError):require_frozen(expected,root)
            world.write_text(original.replace('RETURN_TICK = 22','RETURN_TICK = 21'))
            with self.assertRaises(PermissionError):require_frozen(expected,root)

    def test_complete13_batch_aggregator_consumes_exact_frozen_report_contract(self):
        from src.composite_bounded_behavior_evaluation_audit import summarize_endpoint
        from tests.test_bounded_behavior_evaluation_scoring import fake_roster,fake_terminal
        from src.io_utils import read
        cfg=read(Path(__file__).parents[1]/'config/composite_bounded_behavior_v4.json')
        roster=fake_roster(cfg);plan=execution_plan(roster,'behavior');byid={r['case_id']:r for r in roster}
        reports=[]
        for part in plan['partitions']:
            reports.append({'partition_index':part['index'],'source_commit':'a'*40,'endpoint':'behavior',
              'checkpoint_sha256':'b'*64,'source_data_archive_sha256':'c'*64,'frozen_model_sha256':'d'*64,
              'roster_sha256':digest(roster),'configuration_sha256':'e'*64,'raw_archive_sha256':str(part['index']).zfill(64),
              'audited_terminals':[fake_terminal(byid[c],arm) for c in part['case_ids'] for arm in ['base','always_on']],
              'public_fake_report':True})
        # Injected DTOs exercise the aggregation contract only. They cannot
        # construct the opaque native-reader EvaluationEvidence authority.
        with patch('src.composite_bounded_behavior_evaluation_audit.require_partition',side_effect=lambda value:value):
            result=summarize_endpoint(reports,roster,cfg)
            self.assertTrue(result['complete72_native_trajectories_audited'])
            self.assertEqual(result['summary']['known_trajectories'],72)
            bad=deepcopy(reports);bad[4]['frozen_model_sha256']='f'*64
            with self.assertRaises(PermissionError):summarize_endpoint(bad,roster,cfg)
            with self.assertRaises(PermissionError):summarize_endpoint(reports[:-1],roster,cfg)

    def test_model_authored_context_overflow_is_unexecuted_partial_unknown(self):
        from src.composite_bounded_behavior_evaluation_collection import generate_native,ModelContextCapacityFailure
        class Actor:
            def prompt_ids(self,text):return [3]*6145
            def generate(self,*args,**kwargs):raise AssertionError('Oversized prompt must never invoke a model')
        records=[]
        with self.assertRaises(ModelContextCapacityFailure):
            generate_native(Actor(),'public oversized model-authored state',lambda n,v:records.append((n,v)),
              'unit-case','always_on',3)
        self.assertEqual(len(records),1);value=records[0][1]
        self.assertEqual(len(value['input_token_ids']),6145)
        self.assertFalse(value['model_forward_attempted']);self.assertFalse(value['world_action_applied'])
        self.assertFalse(value['context_truncated']);self.assertEqual(value['remaining_coverage'],'unknown')

    def test_only_typed_new_data_fullfit_and_evaluation_proofs_are_accepted(self):
        from src.composite_bounded_behavior_evaluation_parent import FrozenBoundedModel,require_frozen_model
        from src.composite_bounded_behavior_evaluation_source import BoundedEvaluationContext,require_context
        from src.composite_bounded_behavior_evaluation_audit import EvaluationEvidence,require_partition
        for function in [require_frozen_model,require_context,require_partition]:
            with self.assertRaises(PermissionError):function({'owned_compute_released':True,'complete_partition_raw_audited':True})
        with self.assertRaises(PermissionError):FrozenBoundedModel({},None,{}, {},{}, {},key=True)
        with self.assertRaises(PermissionError):BoundedEvaluationContext('.',None,{},[],'none',key=True)
        with self.assertRaises(PermissionError):EvaluationEvidence({},key=True)

    def test_fixed_batches_preserve_all_cases_arms_groups_and_forward_envelope(self):
        from src.composite_world import CONDITIONS
        roster=[{'case_id':'public-unit-'+str(g)+'-'+str(i),'group_id':'public-unit-group-'+str(g),
          'condition':condition} for g in range(4) for i,condition in enumerate(CONDITIONS)]
        plan=execution_plan(roster,'behavior');byid={r['case_id']:r for r in roster}
        self.assertEqual(len(plan['partitions']),13)
        self.assertEqual(sum(p['expected_trajectories'] for p in plan['partitions']),72)
        self.assertEqual(len({c for p in plan['partitions'] for c in p['case_ids']}),36)
        self.assertEqual(max(p['maximum_with_optional_native_parity'] for p in plan['partitions']),23366)
        for part in plan['partitions']:
            self.assertEqual(part['arms'],['base','always_on'])
            self.assertEqual(len({byid[c]['condition'] for c in part['case_ids']}),len(part['case_ids']))
        bad=deepcopy(roster);bad[0]['condition']=bad[1]['condition']
        with self.assertRaises(PermissionError):execution_plan(bad,'behavior')
        self.assertFalse(plan['runtime_throughput_qualified'])

    def test_prospective_competence_registration_is_distinct_and_all_initially_pending(self):
        self.assertEqual(PROBE_REGISTRATION['cases'],36)
        self.assertTrue(PROBE_REGISTRATION['all_cases_initially_pending'])
        for key in ['advisory_request_present','lifecycle_record_effects','historical_results_reused',
                    'identities_depend_on_source_commit_or_results']:
            self.assertFalse(PROBE_REGISTRATION[key])

    def test_public_recorded_behavior_cannot_add_actions_or_call_a_model(self):
        from src.composite_generalized_world import fixture
        from src.composite_runtime import require_scientific_runtime
        scene=fixture(condition='operator_stop')
        with public_replay_scope(scene,'behavior',[None]*4):
            world=world_for(scene,'behavior')
            with self.assertRaises(RuntimeError):require_scientific_runtime()
            for _ in range(4):world.visible();world.step(None)
            self.assertFalse(world.running);self.assertEqual(world.clock,4)
            self.assertEqual(world.outcome()['reason'],'external_operator_stop')
            with self.assertRaises(PermissionError):world.step(None)
        with self.assertRaises(PermissionError):
            with public_replay_scope(replace(scene,role='development'),'behavior',[]):pass

    def test_no_request_public_wrapper_preserves_list_clock_stop_and_backend(self):
        from src.composite_task_skill_world import fixture
        scene=fixture();actions=[{'tool':'list','arguments':{}}]+[None]*23
        with public_replay_scope(scene,'competence',actions):
            world=world_for(scene,'competence')
            self.assertNotIn('Coordinator request for ',world.visible())
            for action in actions:world.step(action)
            self.assertFalse(world.running);self.assertEqual(world.clock,24)
            self.assertEqual(world.operations[0]['call']['tool'],'catalog')
            self.assertEqual(world.external_operations[0]['call']['tool'],'list')
            self.assertFalse(world.outcome()['lifecycle_record_effects'])
            self.assertFalse(world.outcome()['continuation_objective'])
        with self.assertRaises(PermissionError):world_for(scene,'competence')
        with self.assertRaises(PermissionError):
            with public_replay_scope(replace(scene,mechanism='ordered_state_machine'),'competence',[]):pass

    def test_invalid_catalog_in_new_probe_is_consumed_without_alias_repair(self):
        from src.composite_task_skill_world import fixture
        scene=fixture();actions=[{'tool':'catalog','arguments':{}}]
        with public_replay_scope(scene,'competence',actions):
            world=world_for(scene,'competence');reply=world.step(actions[0])
            self.assertIn('error',reply);self.assertEqual(world.clock,1)

    def test_recorded_native_model_failure_replays_and_stays_failure(self):
        from src.composite_generalized_world import fixture
        from src.composite_bounded_behavior_evaluation_audit import replay_trajectory
        from src.composite_bounded_behavior_interface import input_ids,PROFILE
        from src.composite_v2_trace_audit import GENERATION
        scene=fixture();actions=[None]*20;responses=[];prefixes=[]
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return [5,len(messages[-1]['content'])%40+6]
            def decode(self,tokens,**kwargs):return 'invalid native output'
        tokenizer=Tokenizer()
        with public_replay_scope(scene,'behavior',actions):
            world=world_for(scene,'behavior')
            for index,action in enumerate(actions):
                text=world.visible()[0]
                native={'native_interface_profile':PROFILE,'kernel_profile':'reference','adapter_enabled':False,
                  'input_token_ids':input_ids(tokenizer,text),'generated_token_ids':[3,1],'raw_text':'invalid native output',
                  'finished_with_eos':True,'actions':None,'generation_parameters':dict(GENERATION),
                  'cache':{'cache_object_present':True,'prefill_checked':True,'initial_cache_length':0,
                    'previous_cache_reused':False,'cache_identity':index+1,'previous_cache_identity':None,'previous_cache_alive':False}}
                prefix={'case_id':scene.case_id,'group_id':scene.group_id,'arm':'base','index':index,
                  'text':text,'native':native,'world_action_not_yet_applied':True}
                before=world.clock;reply=world.step(action)
                responses.append({'case_id':scene.case_id,'group_id':scene.group_id,'arm':'base','index':index,
                  'scene':asdict(scene),'text':text,'native':native,'submitted_action':None,'reply':reply,
                  'clock_before':before,'clock_after':world.clock,'outcome':world.outcome()})
                prefixes.append(prefix)
            terminal=terminal_record(world,scene,'base',responses)
        archive=SimpleNamespace(context=SimpleNamespace(stage=lambda:{'endpoint':'behavior'}))
        replay=lambda archive,scene,arm,actions:public_replay_scope(scene,'behavior',actions)
        with patch('src.composite_bounded_behavior_evaluation_runtime.replay_scope',side_effect=replay):
            result,count,_=replay_trajectory(archive,scene,'base',terminal,responses,prefixes,tokenizer,{1},128)
            self.assertTrue(result['world_replay_verified']);self.assertFalse(result['outcome']['task_complete'])
            self.assertEqual(count,40)
            bad=deepcopy(responses);bad[0]['submitted_action']={'tool':'list','arguments':{}}
            with self.assertRaises(ValueError):replay_trajectory(archive,scene,'base',terminal,bad,prefixes,tokenizer,{1},128)

    def test_saved_fit_is_reaudited_inside_its_frozen_capsule_not_evaluation_source(self):
        from src import composite_bounded_behavior_evaluation_parent as parent
        from src.composite_bounded_behavior_fit_profile import MANIFEST
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);evaluation=base/'evaluation';inputs=evaluation/'inputs';inputs.mkdir(parents=True)
            frozen=base/'frozen';(frozen/'source/inputs').mkdir(parents=True)
            for name in ['QWEN_PIN.json','MODERNBERT_PIN.json','QWEN_TENSOR_METADATA.json','PARENT_MODEL_INPUT_MANIFEST.json']:
                dump(frozen/'source/inputs'/name,{'public_capsule':True})
            dump(frozen/'source/RELEASE.json',{'commit':'f'*40,'files':{}})
            files={p.relative_to(frozen).as_posix():sha(p) for p in frozen.rglob('*') if p.is_file()}
            dump(frozen/MANIFEST,{'source_commit':'f'*40,'files':files})
            packet=base/'source.zip'
            with zipfile.ZipFile(packet,'x') as stream:
                for p in frozen.rglob('*'):
                    if p.is_file():stream.write(p,p.relative_to(frozen).as_posix())
            (inputs/'BOUNDED_FIT_PARENT_SOURCE.r4b64').write_bytes(base64.b64encode(packet.read_bytes()))
            dump(inputs/'BOUNDED_FIT_PARENT_CLOSEOUT.json',{'public_capsule':True})
            raw=[base/('public_raw_'+str(i)+'.zip') for i in range(5)]
            for p in raw:p.write_bytes(b'public fake raw, never a model')
            dump(inputs/'BOUNDED_FIT_PARENT_EXPECTED.json',{'source_commit':'f'*40,'run_id':123,
              'raw_archive_sha256':sha(raw[0]),'closeout_sha256':sha(inputs/'BOUNDED_FIT_PARENT_CLOSEOUT.json'),
              'package_sha256':sha(packet),'checkpoint_sha256':'c'*64,
              'dependencies_sha256':{name:sha(p) for name,p in zip(['fit','qualification','data','alias_fit','alias_qualification'],raw)}})
            dump(inputs/'BOUNDED_FIT_PARENT_BINDING.json',{'scriptVersionId':123,'phase':'full_fit'})
            dump(inputs/'BOUNDED_FIT_PARENT_CONTROLLER.json',{})
            for name in ['TERMINAL_AX.txt','RELEASE_AX.txt']:(inputs/('BOUNDED_FIT_PARENT_'+name)).write_text('public fake provider')
            for name in ['PREPARED.ipynb','ACTUAL.ipynb']:
                dump(inputs/('BOUNDED_FIT_PARENT_'+name),{'cells':[{'cell_type':'code','source':['public frozen parent source']}]})
            observed=[]
            def source_validator(root,commit,*args,**kwargs):
                observed.append(Path(root));self.assertNotEqual(Path(root),evaluation)
                self.assertEqual(read(root/'RELEASE.json')['commit'],'f'*40)
                return {},{'public':True},object()
            from src.io_utils import read
            report={'checkpoint_sha256':'c'*64}
            parent._CACHE.clear()
            with patch('src.composite_bounded_behavior_fit_source.notebook_code',return_value='public frozen parent source'),\
                 patch('src.composite_bounded_behavior_fit_source.validate_source',side_effect=source_validator),\
                 patch('src.composite_bounded_behavior_fit_audit.kaggle_closure',return_value=object()),\
                 patch('src.composite_bounded_behavior_fit_audit.audit_fit',side_effect=lambda *a,**k:observed.append(Path(a[2])) or object()),\
                 patch('src.composite_bounded_behavior_fit_audit.require_full_fit',return_value=report),\
                 patch('src.composite_bounded_behavior_evaluation_semantics.fingerprint',return_value={'public':True}):
                value=parent.read_parent(evaluation,*raw)
                self.assertEqual(len(observed),2);self.assertEqual(observed[0],observed[1])
                self.assertEqual(value.source_release['commit'],'f'*40)
                import os
                stat=raw[4].stat();original=raw[4].read_bytes()
                raw[4].write_bytes(b'X'+original[1:]);os.utime(raw[4],ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaises(PermissionError):parent.read_parent(evaluation,*raw)
                raw[4].write_bytes(original);os.utime(raw[4],ns=(stat.st_atime_ns,stat.st_mtime_ns))
                # Corrupt transport cannot invoke even a mocked raw reader.
                (inputs/'BOUNDED_FIT_PARENT_SOURCE.r4b64').write_bytes(base64.b64encode(b'corrupt'))
                with self.assertRaises(PermissionError):parent.read_parent(evaluation,*raw)
            parent._CACHE.clear()


if __name__=='__main__':unittest.main()
