"""Public toy schedule/source tests only; no real rows, Qwen or provider work."""
from contextlib import nullcontext,contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch

from src.io_utils import digest,dump,sha
from src.composite_bounded_sampling_fit_profile import MODE,MANIFEST
from src.composite_bounded_sampling_training import approved_full_fit_rows


def public_fixture():
    rows=[{'state_id':'FAKE_ONLY::'+str(i),'split':'training','public_fixture':False,
      'training_eligible':True,'terminal':False,'teacher_action':{'tool':'list','arguments':{}},
      'text':'public toy','value':i+1} for i in range(3)]
    ids=[rows[i]['state_id'] for i in [1,1,0,2]*32]
    plan={'state_ids':ids,'coverage_passed':True,'draws':128,'updates':64,'accumulation':2}
    cfg={'behavior_training':{'steps':64,'accumulation':2,'seed':202610212,
                             'learning_rate':0.0005,'clip_norm':1.0}}
    data=SimpleNamespace(fitting_rows=lambda:deepcopy(rows),fitting_draw_plan=lambda:deepcopy(plan))
    stage={'version':MODE,'phase':'full_fit','sampling':'approved_48anchors80uniform_exact128_ids_v1',
           'approved_draw_plan_sha256':digest(plan),'optimizer_updates':64}
    return rows,plan,cfg,data,stage


class ApprovedSamplingFitTests(unittest.TestCase):
    def test_new_fit_evaluation_parent_dispatches_only_its_own_capsule_and_mode(self):
        import base64,zipfile
        from src import composite_bounded_behavior_evaluation_parent as parent
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'evaluation';inputs=root/'inputs';inputs.mkdir(parents=True)
            packet_root=base/'packet';source=packet_root/'source';(source/'inputs').mkdir(parents=True)
            for name in ['QWEN_PIN.json','MODERNBERT_PIN.json','QWEN_TENSOR_METADATA.json','PARENT_MODEL_INPUT_MANIFEST.json']:
                dump(source/'inputs'/name,{'public_fixture':True})
            dump(source/'RELEASE.json',{'commit':'f'*40,'files':{}})
            dump(packet_root/MANIFEST,{'version':MODE,'source_commit':'f'*40,
                 'files':{p.relative_to(packet_root).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}})
            archive=base/'source.zip'
            with zipfile.ZipFile(archive,'x') as stream:
                for p in packet_root.rglob('*'):
                    if p.is_file():stream.write(p,p.relative_to(packet_root).as_posix())
            dump(inputs/'BOUNDED_FIT_PARENT_CLOSEOUT.json',{'public_fixture':True})
            raw=[base/('raw_'+str(i)+'.zip') for i in range(5)]
            for p in raw:p.write_bytes(b'public source-dispatch fixture')
            expected={'fit_mode':MODE,'source_commit':'f'*40,'run_id':123,'package_sha256':sha(archive),
                'raw_archive_sha256':sha(raw[0]),'closeout_sha256':sha(inputs/'BOUNDED_FIT_PARENT_CLOSEOUT.json'),
                'checkpoint_sha256':'c'*64,'dependencies_sha256':{n:sha(p) for n,p in
                  zip(['fit','qualification','data','alias_fit','alias_qualification'],raw)}}
            dump(inputs/'BOUNDED_FIT_PARENT_EXPECTED.json',expected)
            (inputs/'BOUNDED_FIT_PARENT_SOURCE.r4b64').write_bytes(base64.b64encode(archive.read_bytes()))
            dump(inputs/'BOUNDED_FIT_PARENT_BINDING.json',{'scriptVersionId':123,'phase':'full_fit'})
            dump(inputs/'BOUNDED_FIT_PARENT_CONTROLLER.json',{})
            for name in ['TERMINAL_AX.txt','RELEASE_AX.txt']:(inputs/('BOUNDED_FIT_PARENT_'+name)).write_text('public')
            for name in ['PREPARED.ipynb','ACTUAL.ipynb']:
                dump(inputs/('BOUNDED_FIT_PARENT_'+name),{'cells':[{'cell_type':'code','source':['public capsule']} ]})
            observed=[]
            def validate(capsule,commit,*args,**kwargs):
                self.assertNotEqual(Path(capsule),root);observed.append(Path(capsule))
                return {},{},object()
            parent._CACHE.clear()
            with patch('src.composite_bounded_sampling_fit_source.notebook_code',return_value='public capsule'),\
                 patch('src.composite_bounded_sampling_fit_source.validate_source',side_effect=validate),\
                 patch('src.composite_bounded_sampling_fit_audit.kaggle_closure',return_value=object()),\
                 patch('src.composite_bounded_sampling_fit_audit.audit_fit',return_value=object()),\
                 patch('src.composite_bounded_sampling_fit_audit.require_full_fit',return_value={'runtime_profile':MODE,'checkpoint_sha256':'c'*64}),\
                 patch('src.composite_bounded_behavior_fit_source.validate_source',side_effect=AssertionError('Old uniform-data route is forbidden')),\
                 patch('src.composite_bounded_behavior_evaluation_semantics.fingerprint',return_value={'public':True}):
                parent.read_parent(root,*raw);self.assertEqual(len(observed),1)
                expected['fit_mode']='wrong-mode';dump(inputs/'BOUNDED_FIT_PARENT_EXPECTED.json',expected)
                with self.assertRaises(PermissionError):parent.read_parent(root,*raw)
            parent._CACHE.clear()

    def test_fit_closure_requires_one_exact_notebook_version_url(self):
        from src.composite_bounded_sampling_fit_audit import kaggle_closure,OWNER_THREAD
        notebook='farhad89/research4-bounded-sampling-fit-v1'
        binding={'mode':MODE,'phase':'full_fit','notebook':notebook,'owner_thread_id':OWNER_THREAD,
                 'source_commit':'a'*40,'scriptVersionId':123}
        controller={**binding,'status':'worker_exited','exit_code':0,'timed_out':False,
          'child':{'child_started':True,'child_exit_confirmed':True,'exit_code':0},
          'raw_prefix_rescue':{'status':'owned_scratch_absent'}}
        terminal='https://www.kaggle.com/code/'+notebook+'?scriptVersionId=123\nsuccessfully ran after 1s'
        release='https://www.kaggle.com/code/'+notebook+'\n0  Active Events\nDraft Session off'
        value=kaggle_closure(binding,'b'*64,terminal,release,controller,['public'],['public'])
        self.assertEqual(value._value['scriptVersionId'],123)
        for bad in [terminal.replace('=123','=1234'),terminal.replace(notebook,notebook+'-other'),
                    terminal.replace(notebook,'farhad89/other')+'\n'+notebook,
                    notebook+' scriptVersionId=123 successfully ran after 1s',
                    terminal.replace('=123','=124')+'\nscriptVersionId=123']:
            with self.subTest(terminal=bad),self.assertRaises(PermissionError):
                kaggle_closure(binding,'b'*64,bad,release,controller,['public'],['public'])

    def test_distinct_fit_route_and_old_data_authority_cannot_be_relabelled(self):
        from src.composite_bounded_sampling_fit_bridge import fit_modules,require_actor_data
        profile,source,audit=fit_modules(MODE)
        self.assertEqual(profile.MANIFEST,MANIFEST)
        self.assertTrue(source.__name__.endswith('composite_bounded_sampling_fit_source'))
        self.assertTrue(audit.__name__.endswith('composite_bounded_sampling_fit_audit'))
        with self.assertRaises(PermissionError):fit_modules('unregistered')
        with self.assertRaises(PermissionError):require_actor_data({'approved_sampling_data_admitted':True})

    def test_exact128_occurrences_drive_all_raw_microsteps_without_uniform_rng(self):
        import torch
        from src.composite_behavior import fit_selected_rows
        rows,plan,cfg,data,stage=public_fixture()
        class ToyActor:
            def __init__(self):
                self.adapters={};self.base_references=[];self.v2_training_plan=stage
                self.bounded_training_context={'bounded_data_evidence':data,'stage':stage,'full_config':cfg}
            def configure(self,*args,**kwargs):
                module=torch.nn.Module();module.A=torch.nn.Parameter(torch.tensor([[0.2]]))
                module.B=torch.nn.Parameter(torch.tensor([[0.0]]));self.adapters={'toy':module}
            def adapter_state(self):
                return {'weights':{n+'.'+part:getattr(m,part).detach().clone()
                                  for n,m in self.adapters.items() for part in ['A','B']}}
            def eval(self):return self
            def training_context(self):return nullcontext()
            def supervised_loss(self,row):
                m=self.adapters['toy'];return (m.A*row['value']+m.B-0.5).square().mean()
            def verify_frozen(self):pass
        actor=ToyActor();journal=[]
        def observe(event,**value):
            if event=='before_microstep':journal.append((value['update'],value['microstep'],value['row']['state_id']))
        with tempfile.TemporaryDirectory() as folder,\
             patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value={'public_fake':True}),\
             patch('src.composite_behavior.np.random.default_rng',side_effect=AssertionError('No IID stream in approved full fit')):
            result=fit_selected_rows(actor,rows,{'name':'all_blocks','scope':'all_blocks'},cfg,folder,
                                     lambda:None,lambda *a,**k:None,observer=observe)
        self.assertEqual(result['optimizer_updates'],64)
        self.assertEqual(result['sampled_state_ids'],plan['state_ids'])
        self.assertEqual(journal,[(i//2+1,i%2,state) for i,state in enumerate(plan['state_ids'])])
        self.assertEqual(len(journal),128)

    def test_changed_or_foreign_schedule_fails_before_any_fit(self):
        rows,plan,cfg,data,stage=public_fixture()
        actor=SimpleNamespace(bounded_training_context={'bounded_data_evidence':data,'stage':stage,'full_config':cfg})
        with patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value={'public_fake':True}):
            self.assertEqual([r['state_id'] for r in approved_full_fit_rows(actor,rows,cfg)],plan['state_ids'])
            for bad in [{**plan,'state_ids':plan['state_ids'][:-1]},
                        {**plan,'state_ids':['FOREIGN']*128}, {**plan,'coverage_passed':False}]:
                data.fitting_draw_plan=lambda bad=bad:deepcopy(bad)
                with self.assertRaises(PermissionError):approved_full_fit_rows(actor,rows,cfg)

    def test_failed_original_core_is_checked_in_its_own_capsule(self):
        from src.composite_bounded_sampling_fit_bridge import verify_failed_core,FAILED_SOURCE,FAILED_PACKAGE_SHA
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);capsule=base/'capsule';current=base/'current'
            (capsule/'src').mkdir(parents=True);(current/'src').mkdir(parents=True)
            core=capsule/'src/public_core.py';core.write_text('original public semantic core')
            (current/'src/public_core.py').write_text('separate source tree, never accepted as old capsule')
            release={'commit':FAILED_SOURCE,'files':{'src/public_core.py':sha(core)}}
            dump(capsule/'RELEASE.json',release)
            parent=SimpleNamespace(receipt=lambda:{'source_commit':FAILED_SOURCE,'package_sha256':FAILED_PACKAGE_SHA},
                original_source=lambda:b'public fake source packet',source_release=lambda:release)
            data=SimpleNamespace(failed_reference=lambda:parent);observed=[]
            @contextmanager
            def unpack(raw,expected,commit,**kwargs):
                self.assertEqual(raw,b'public fake source packet');self.assertEqual(commit,FAILED_SOURCE)
                yield capsule,{}
            with patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value={'public_fake':True}),\
                 patch('src.composite_bounded_behavior_fit_source.DATA_CORE',{'src/public_core.py'}),\
                 patch('src.composite_bounded_sampling_fit_bridge.frozen_source_package',side_effect=unpack),\
                 patch('src.composite_bounded_behavior_evaluation_semantics.fingerprint',side_effect=lambda root:observed.append(root) or {'fake':True}),\
                 patch('src.composite_bounded_behavior_evaluation_semantics.require_frozen',side_effect=lambda expected,root:observed.append(root)):
                result=verify_failed_core(data,current)
                self.assertTrue(result['original_failed_status_preserved'])
                self.assertEqual(observed,[capsule,current])
                core.write_text('tampered original capsule')
                with self.assertRaises(PermissionError):verify_failed_core(data,current)


if __name__=='__main__':unittest.main()
