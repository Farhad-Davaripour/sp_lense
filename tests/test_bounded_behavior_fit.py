"""Public/tiny engineering fixtures; no scientific world, data or model calls."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch

from src.io_utils import read
from src.composite_bounded_behavior_interface import PROFILE,action_text,parse
from src.composite_bounded_behavior_fit_profile import MODE,RAW_PROFILE,FIT_RAW_PROFILE,stress_schedule
from src.composite_bounded_behavior_training import token_plan,validate_rows,selected_rows
from src.composite_bounded_behavior_fit_audit import BoundedFitEvidence,KaggleClosure,require_qualification,require_full_fit


def fixture_rows():
    return [{'state_id':'unit_only_'+str(i),'text':'x'*(i+1),'split':'training','public_fixture':False,
      'terminal':False,'training_eligible':True,'labels':{'gate':False},
      'teacher_action':{'tool':'list','arguments':{}},'target_phase':'post_commit_deferral'} for i in range(5)]


class Tokenizer:
    eos_token_id=1
    def apply_chat_template(self,messages,**kwargs):return [7]*len(messages[-1]['content'])
    def encode(self,text,**kwargs):return [3,4]


class BoundedFitBoundaryTests(unittest.TestCase):
    def public_closure(self,code,release_notebook=None):
        from src.composite_bounded_behavior_fit_audit import kaggle_closure,OWNER_THREAD
        notebook='farhad89/research4-bounded-behavior-qualification-v4'
        binding={'scriptVersionId':123,'source_commit':'a'*40,'phase':'qualification',
          'notebook':notebook,'mode':MODE,'owner_thread_id':OWNER_THREAD}
        controller={**binding,'status':'worker_exited','exit_code':0,'timed_out':False,
          'child':{'child_started':True,'child_exit_confirmed':True,'exit_code':0},
          'raw_prefix_rescue':{'status':'owned_scratch_absent'}}
        terminal=notebook+' scriptVersionId=123 successfully ran after 1s'
        release='https://www.kaggle.com/code/'+(release_notebook or notebook)+' 0  Active Events Draft Session off'
        return kaggle_closure(binding,'b'*64,terminal,release,controller,[code],[code])

    def test_release_observation_must_bind_exact_target_notebook_not_a_prefix_collision(self):
        self.assertIsInstance(self.public_closure('public_fixture'),KaggleClosure)
        for wrong in ['farhad89/research4-other',
          'farhad89/research4-bounded-behavior-qualification-v4-copy']:
            with self.assertRaises(PermissionError):self.public_closure('public_fixture',wrong)

    def test_matching_caller_notebooks_cannot_replace_reviewed_source_code(self):
        from src.composite_bounded_behavior_fit_audit import source_closure_matches
        from src.composite_bounded_behavior_fit_source import notebook_code
        from src.composite_bounded_behavior_fit_profile import MANIFEST
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source';(root/'notebooks').mkdir(parents=True)
            (root/'notebooks/composite_bounded_behavior_fit.py').write_text(
              "COMMIT='SOURCE_COMMIT_PLACEHOLDER'\nMANIFEST='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            (root.parent/MANIFEST).write_text('{"public_fixture":true}')
            from src.io_utils import sha
            code=notebook_code(root,'a'*40,sha(root.parent/MANIFEST))
            stage={'notebook':'farhad89/research4-bounded-behavior-qualification-v4'}
            self.assertTrue(source_closure_matches(self.public_closure(code),root,'a'*40,'b'*64,stage))
            with self.assertRaises(PermissionError):
                source_closure_matches(self.public_closure('same caller-supplied substitute'),root,'a'*40,'b'*64,stage)

    def test_complete_model_recipe_and_placement_pins_cannot_silently_drift(self):
        from src.composite_bounded_behavior_fit_source import validate_configuration
        root=Path(__file__).parents[1]
        cfg=read(root/'config/composite_controller_rank32_v2.json')
        placement=read(root/'config/composite_qwen_two_t4_candidate_v1.json')
        self.assertEqual(validate_configuration(cfg,placement),cfg)
        for section,key,value in [('qwen','revision','0'*40),('qwen','maximum_tokens',128),
          ('qwen','training_memory_reserve_bytes',0),('behavior_training','learning_rate',0.001)]:
            bad=deepcopy(cfg);bad[section][key]=value
            with self.assertRaises(PermissionError):validate_configuration(bad,placement)
        bad=deepcopy(placement);bad['public_unregistered_field']=True
        with self.assertRaises(PermissionError):validate_configuration(cfg,bad)

    def test_new_native_targets_accept_list_and_reject_historical_catalog(self):
        action={'tool':'list','arguments':{}}
        self.assertEqual(parse(action_text(action)),[action])
        with self.assertRaises(ValueError):action_text({'tool':'catalog','arguments':{}})

    def test_all_new_target_phases_are_retained_without_detector_gate_filtering(self):
        rows=fixture_rows();plan=token_plan(rows,Tokenizer(),{})
        self.assertEqual(plan['source_training_states'],5)
        self.assertEqual(plan['eligible_training_states'],5)
        self.assertEqual(plan['native_interface_profile'],PROFILE)
        self.assertEqual(plan['qualification_state_ids'],['unit_only_4','unit_only_3'])
        self.assertTrue(all(r['target_token_ids']==[3,4,1] for r in plan['records']))
        for key,value in [('split','development'),('public_fixture',True),('training_eligible',False),
                          ('terminal',True),('task_only',True),('teacher_action',None)]:
            bad=deepcopy(rows);bad[0][key]=value
            with self.assertRaises(PermissionError):validate_rows(bad)

    def test_context_and_complete_target_are_never_truncated(self):
        rows=fixture_rows();rows[0]['text']='x'*6143
        with self.assertRaises(ValueError):token_plan(rows,Tokenizer(),{})
        class LongTarget(Tokenizer):
            def encode(self,text,**kwargs):return [3]*160
        with self.assertRaises(ValueError):token_plan(fixture_rows(),LongTarget(),{})
        class NoEOS(Tokenizer):eos_token_id=None
        with self.assertRaises(ValueError):token_plan(fixture_rows(),NoEOS(),{})

    def test_stress_schedule_uses_new_complete_sizes_and_a_real_switch(self):
        result=stress_schedule(token_plan(fixture_rows(),Tokenizer(),{}))
        self.assertEqual(result['state_ids'],['unit_only_4','unit_only_3','unit_only_0','unit_only_1','unit_only_4','unit_only_3'])
        self.assertEqual(result['updates'],3);self.assertFalse(result['qualification_checkpoint_initializes_fit'])
        same={'records':[{'state_id':'a','combined_tokens':10},{'state_id':'b','combined_tokens':10}]}
        with self.assertRaises(ValueError):stress_schedule(same)
        same['records'][0]['combined_tokens']=6145
        with self.assertRaises(ValueError):stress_schedule(same)

    def test_new_export_profiles_do_not_widen_or_impersonate_old_profiles(self):
        from src.composite_v2_raw_exports import validate_export_plan,STATEFUL_TASK_PROFILE,TASK_PROFILE
        for profile,raw,reserve in [(RAW_PROFILE,6,18),(FIT_RAW_PROFILE,4,12)]:
            stage={'raw_evidence_profile':profile,'maximum_raw_evidence_bytes':raw*1024**3,
              'physical_export_reserve_bytes':reserve*1024**3}
            self.assertEqual(validate_export_plan(stage,profile)['raw_evidence_profile'],profile)
            for old in [TASK_PROFILE,STATEFUL_TASK_PROFILE]:
                with self.assertRaises(PermissionError):validate_export_plan(stage,old)
            stage['maximum_raw_evidence_bytes']+=1
            with self.assertRaises(PermissionError):validate_export_plan(stage,profile)

    def test_nested_fit_source_has_a_separate_cap_without_widening_data_or_old_transport(self):
        from types import SimpleNamespace
        from src.composite_task_skill_source_package import validate_source_envelope,source_transport_limit
        fit='BOUNDED_BEHAVIOR_FIT_PACKAGE_FILES.json'
        data='BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json'
        infos=[SimpleNamespace(filename='source/public-unit.bin',file_size=190*1024**2)]
        self.assertIsNone(validate_source_envelope(infos,fit))
        for old in [data,'TASK_TRAINING_PACKAGE_FILES.json','RISK_DATA_PACKAGE_FILES.json','TOOL_ALIAS_PACKAGE_FILES.json']:
            self.assertEqual(source_transport_limit(old),128*1024**2)
            with self.assertRaises(ValueError):validate_source_envelope(infos,old)
        infos[0].file_size=256*1024**2+1
        with self.assertRaises(ValueError):validate_source_envelope(infos,fit)

    def test_actual_fit_archive_is_complete_and_byte_equal_under_its_own_manifest(self):
        import zipfile
        from src.io_utils import dump,sha
        from src.composite_bounded_behavior_fit_source import validate_fit_package
        from src.composite_bounded_behavior_fit_profile import MANIFEST
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=root/'bundle';(bundle/'source').mkdir(parents=True)
            member=bundle/'source/public_unit.txt';member.write_text('public fixture bytes')
            dump(bundle/MANIFEST,{'version':MODE,'source_commit':'a'*40,
              'files':{'source/public_unit.txt':sha(member)}})
            archive=root/'fit.zip'
            with zipfile.ZipFile(archive,'x') as stream:
                for p in [member,bundle/MANIFEST]:stream.write(p,p.relative_to(bundle).as_posix())
            self.assertIsNone(validate_fit_package(archive,bundle))
            member.write_text('altered after frozen upload')
            with self.assertRaises(PermissionError):validate_fit_package(archive,bundle)
            extra=root/'extra.zip'
            with zipfile.ZipFile(extra,'x') as stream:
                stream.write(member,member.relative_to(bundle).as_posix())
                stream.write(bundle/MANIFEST,MANIFEST);stream.writestr('unaccounted.txt','not in frozen source')
            with self.assertRaises(PermissionError):validate_fit_package(extra,bundle)

    def test_flags_cannot_construct_data_qualification_or_fullfit_authority(self):
        for cls in [BoundedFitEvidence,KaggleClosure]:
            with self.assertRaises(PermissionError):cls({'owned_compute_released':True},key=True)
        for reader in [require_qualification,require_full_fit]:
            with self.assertRaises(PermissionError):reader({'complete_training_qualification_audited':True})
        class Actor:bounded_training_context=None
        with self.assertRaises(PermissionError):selected_rows(Actor(),fixture_rows())

    def test_v4_actor_profile_without_its_constructor_cannot_load_a_model(self):
        from src.composite_actor import Actor
        with patch('src.composite_actor.require_databricks'):
            with self.assertRaisesRegex(PermissionError,'separately validated V4'):
                Actor('never_a_model_path',{},lambda:None,native_interface_profile=PROFILE)

    def test_new_native_replay_uses_its_canonical_parser_and_keeps_old_default(self):
        from src import composite_bounded_behavior_interface as interface
        from src.composite_v2_trace_audit import _native,GENERATION
        class NativeTokenizer(Tokenizer):
            def decode(self,tokens,**kwargs):return action_text({'tool':'list','arguments':{}})
        tokenizer=NativeTokenizer();prompt='unit'
        response={'generation_parameters':dict(GENERATION),'native_interface_profile':PROFILE,
          'kernel_profile':'reference','adapter_enabled':False,'generated_token_ids':[3,1],
          'input_token_ids':[7]*len(prompt),'raw_text':tokenizer.decode([3]),
          'finished_with_eos':True,'actions':[{'tool':'list','arguments':{}}],
          'cache':{'cache_object_present':True,'prefill_checked':True,'initial_cache_length':0,
            'previous_cache_reused':False,'cache_identity':1,'previous_cache_identity':None,'previous_cache_alive':False}}
        self.assertEqual(_native(response,prompt,False,tokenizer,{1},128,None,interface=interface),1)
        with self.assertRaises(ValueError):_native(response,prompt,False,tokenizer,{1},128,None)
        bad=deepcopy(response);bad['actions']=None
        with self.assertRaises(ValueError):_native(bad,prompt,False,tokenizer,{1},128,None,interface=interface)

    def test_three_update_tiny_qualifier_retains_actual_moments(self):
        from tests.test_composite_v2_training import TinyFitActor
        from src.composite_behavior import fit_selected_rows
        class TinyBoundedActor(TinyFitActor):
            native_interface_profile=PROFILE
            def adapter_state(self):
                result=super().adapter_state();result['native_interface_profile']=PROFILE;return result
        cfg=read(Path(__file__).parents[1]/'config/composite_controller_rank32_v2.json')
        rows=[{'state_id':'tiny_unit_'+str(i),'value':float(i+1)} for i in range(4)]
        schedule=rows+rows[:2];resident=[]
        def observer(event,**value):
            if event=='before_microstep':resident.append((value['update'],len(value['optimizer'].state)))
        with tempfile.TemporaryDirectory() as folder:
            actor=TinyBoundedActor({'phase':'qualification','optimizer_updates':3,'version':MODE})
            result=fit_selected_rows(actor,rows,{'name':'tiny_unit'},cfg,folder,lambda:None,
              lambda *a,**k:None,observer=observer,qualification_rows=schedule)
        self.assertEqual(result['optimizer_updates'],3);self.assertEqual(len(result['losses']),6)
        self.assertEqual(resident,[(1,0),(1,0),(2,2),(2,2),(3,2),(3,2)])
        self.assertFalse(result['full_fit_completed'])

    def test_failed_public_worker_preserves_unknown_optimizer_and_consumed_forward(self):
        from tests.test_task_skill_lifecycle import Store
        from src.composite_owned_artifacts import OwnedArtifacts
        from src.composite_bounded_behavior_fit_worker import run
        source='a'*40
        stage={'version':MODE,'phase':'qualification','source_commit':source,'worker_seconds':100,
          'export_reserve_seconds':10,'maximum_backbone_forwards':30000,'source_data_archive_sha256':'b'*64}
        with tempfile.TemporaryDirectory() as folder:
            store=Store()
            class Services:
                clock=staticmethod(time.monotonic)
                def authorize(self,*args):return {'stage':stage,'full_config':{}},[{'public_unit':True}]
                def owner(self,context):return OwnedArtifacts(store,'bounded_behavior_training_v4/'+source+'/qualification_01',
                  source,base=folder,maximum_bytes=8*1024**2,bounded_fit=True)
                def runtime(self,context):return {'public_unit':True}
                def resources(self):pass
                def alarm(self,seconds):return nullcontext()
                def load(self,context,owner,guard,forward):forward('attempt');raise RuntimeError('public fake forward failure')
                def release(self):pass
                def remove_owned_cache(self,owner):return {'owned_download_cache_removed':True}
            result=run('public_fixture_not_science',source,_services=Services())
        self.assertEqual(result['status'],'failed');self.assertIsNone(result['optimizer_updates'])
        self.assertEqual(result['forward_counts'],{'attempted':1,'completed':0})
        self.assertTrue(result['public_fixture_test_mode']);self.assertFalse(result['qualification_admitted'])


if __name__=='__main__':unittest.main()
