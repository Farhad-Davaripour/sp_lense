from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from src import composite_frozen_worker as worker
from src import composite_frozen_launch as launch
from src import composite_probe_ownership as ownership
from src.composite_exports import make_evidence_archive, verify_evidence_archive
from src.composite_runtime import run_logged


class FrozenWorkerTests(unittest.TestCase):
    commit='a'*40
    token='b'*32

    def fixture(self, root):
        source=root/'source';(source/'config').mkdir(parents=True);(source/'inputs').mkdir()
        (source/'config/composite_controller_rank32_v2.json').write_text('{"qwen":{}}')
        (source/'inputs/MANIFEST.json').write_text('{}')
        output_base=root/'output';target=output_base/self.commit/'qualification'
        cache_base=root/'cache';cache=cache_base/self.commit/'qualification'
        ownership.create_owned_cache(cache,self.commit,self.token,cache_base)
        context={'source_commit':self.commit,'authorization_sha256':'c'*64,
                 'model_pin':{'files':[{'size':worker.DOWNLOAD_BYTES}]},
                 'pin':{'destination':str(target),'cache_root':str(cache),'disk_policy':copy.deepcopy(worker.DISK_POLICY)}}
        clock=[0.0];events=[];state={}
        services=worker.Services()
        services.clock=lambda:clock[0]
        services.authorize=lambda *args:context
        services.ownership=lambda *args:ownership.verify_owned_cache(*args,namespace=cache_base)
        services.verify_transport=lambda *args:{}
        services.output_base=str(output_base)
        services.runtime=lambda *args:{'synthetic_device_metadata_only':True}
        services.preflight_data=lambda *args:{'source_training_observations':720,'heldout_content_read':False}
        services.disk=lambda cache,output,pin:worker.disk_admission(cache,output,pin,usage=lambda path:SimpleNamespace(free=10**12))
        def reserve(target,*args):
            (Path(target)/'EXPORT_RESERVE.bin').write_bytes(b'owned fixture reserve')
            return {'fixture':True}
        services.reserve=reserve
        services.alarm=lambda seconds:nullcontext()
        services.resources=lambda:None
        services.release_model=lambda:events.append('model_released')
        def load(root,commit,config,cache,guard,callback,persist):
            events.append('load');guard()
            actor=SimpleNamespace(frozen_probe_counts={'attempted':0,'completed':0})
            state['actor']=actor;state['callback']=callback;state['persist']=persist
            persist('FROZEN_LOADED_LAYOUT.json',{'synthetic_fixture_only':True})
            return actor,context,{'fixture':True}
        def collect(actor,ctx,persist):
            for index in range(2):
                actor.frozen_probe_counts['attempted']+=1;state['callback']('attempt')
                actor.frozen_probe_counts['completed']+=1;state['callback']('complete')
                persist('SYNTHETIC_PROBE_'+str(index)+'.json',{'fixture_only':True})
            return {'probes':[{},{}],'forward_counts':dict(actor.frozen_probe_counts),
                    'optimizer_updates':0,'behavior_adapter_fitting':False,'heldout_content_read':False}
        services.load=load;services.collect=collect
        return source,target,cache,cache_base,context,clock,events,state,services

    def invoke(self, fixture):
        source,target,cache,*_,services=fixture
        with redirect_stdout(io.StringIO()):
            return worker.run(source,self.commit,str(target),str(cache),self.token,services)

    def test_unauthorized_stops_before_output_or_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=self.fixture(Path(temp));source,target,cache,*_=fixture
            with self.assertRaises(PermissionError):
                worker.run(source,self.commit,str(target),str(cache),self.token)
            self.assertFalse(target.exists())

    def test_controlled_success_seals_counts_and_shared_export(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=self.fixture(Path(temp));value=self.invoke(fixture);target=fixture[1]
            self.assertEqual(value['status'],'complete')
            self.assertEqual(value['forward_counts'],{'attempted':2,'completed':2})
            self.assertTrue(value['counts_exact_at_controlled_exit'])
            self.assertFalse(value['full_validation_admitted'])
            self.assertIn('model_released',fixture[6])
            self.assertFalse((target/'EXPORT_RESERVE.bin').exists())
            verify_evidence_archive((target/'FROZEN_EVIDENCE.zip').read_bytes())
            counts=launch.observed_exit_counts(target,self.commit,{'child_started':True,'child_exit_confirmed':True},self.token)
            self.assertTrue(counts['counts_exact_at_exit'])

    def test_disk_download_load_and_mid_forward_failures_preserve_partial_evidence(self):
        for failure in ['disk','download','load','mid_forward']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                f=self.fixture(Path(temp));services=f[-1]
                if failure=='disk':services.disk=lambda *args:(_ for _ in ()).throw(OSError('fixture disk rejection'))
                elif failure in ['download','load']:
                    def failing(*args):raise ConnectionError('fixture '+failure+' failure')
                    services.load=failing
                else:
                    def mid(actor,ctx,persist):
                        actor.frozen_probe_counts['attempted']+=1;f[7]['callback']('attempt')
                        persist('PARTIAL_PREFILL.json',{'fixture':True})
                        raise RuntimeError('fixture forward did not return')
                    services.collect=mid
                value=self.invoke(f)
                self.assertEqual(value['status'],'failed')
                self.assertTrue(value['counts_exact_at_controlled_exit'])
                self.assertEqual(value['forward_counts'],{'attempted':1 if failure=='mid_forward' else 0,'completed':0})
                self.assertTrue((f[1]/'FROZEN_EVIDENCE.zip').exists())
                self.assertIn('model_released',f[6])
                self.assertFalse((f[1]/'EXPORT_RESERVE.bin').exists())

    def test_protected_time_starts_before_download_and_forward_401_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(Path(temp));services=f[-1]
            def delayed(*args):
                f[5][0]=2221
                args[4]()  # The owner guard already includes the download phase.
            services.load=delayed
            value=self.invoke(f)
            self.assertEqual(value['error_type'],'TimeoutError')
            self.assertEqual(value['forward_counts'],{'attempted':0,'completed':0})
            self.assertGreaterEqual(value['seconds'],2221)
        snapshots=[];budget=worker.ForwardBudget(0,lambda:0,snapshots.append)
        for _ in range(400):budget.event('attempt');budget.event('complete')
        with self.assertRaises(TimeoutError):budget.event('attempt')
        self.assertEqual((budget.attempted,budget.completed),(400,400))

    def test_near_cap_and_orphan_pending_leave_terminal_and_rescue_headroom(self):
        policy={'maximum_raw_evidence_bytes':32768,'terminal_metadata_headroom_bytes':8192}
        with patch.dict(worker.DISK_POLICY,policy),tempfile.TemporaryDirectory() as temp:
            f=self.fixture(Path(temp));target=f[1]
            def near(actor,ctx,persist):
                raw=target/'raw';(raw/'INTERRUPTED_COUNTER.pending').write_bytes(b'{"incomplete":')
                used=sum(file.stat().st_size for file in raw.rglob('*') if file.is_file())
                size=worker.DISK_POLICY['maximum_raw_evidence_bytes']-worker.DISK_POLICY['terminal_metadata_headroom_bytes']-used-200
                persist('NEAR_LIMIT.json',{'payload':'x'*size})
                raise RuntimeError('synthetic near-cap failure')
            f[-1].collect=near
            value=self.invoke(f);self.assertEqual(value['status'],'failed')
            raw_hash=(target/'raw/NEAR_LIMIT.json').read_bytes()
            (target/'EXPORT_RECEIPT.json').unlink()  # Fixture: interrupted final export publication.
            state={'child_started':True,'child_exit_confirmed':True,'exit_code':-9}
            observation={'status':'post_exit_fixture','counts':launch.observed_exit_counts(target,self.commit,state,self.token)}
            proof=launch.preserve_after_exit(target,self.commit,state,observation)
            self.assertEqual(proof['evidence_kind'],'controller_post_exit_rescue')
            with zipfile.ZipFile(proof['archive']) as archive:
                self.assertEqual(archive.read('NEAR_LIMIT.json'),raw_hash)
                self.assertEqual(archive.read('INTERRUPTED_COUNTER.pending'),b'{"incomplete":')
                self.assertIn('RUN_RECEIPT.json',archive.namelist())
            verify_evidence_archive(Path(proof['archive']).read_bytes())

    def test_interrupted_atomic_journal_retains_old_value_and_pending_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'FORWARD_COUNTS.json';worker.atomic_json(path,{'attempted':0,'completed':0})
            with patch('src.composite_frozen_worker.os.replace',side_effect=RuntimeError('interrupted rename')):
                with self.assertRaises(RuntimeError):worker.atomic_json(path,{'attempted':1,'completed':0})
            self.assertEqual(json.loads(path.read_text()),{'attempted':0,'completed':0})
            self.assertEqual(len(list(path.parent.glob('*.pending'))),1)

    def test_abrupt_exit_reports_unknown_final_counts_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp);raw=target/'raw';raw.mkdir()
            worker.atomic_json(raw/'FORWARD_COUNTS.json',{'attempted':1,'completed':0})
            value=launch.observed_exit_counts(target,self.commit,{'child_started':True,'child_exit_confirmed':True})
            self.assertFalse(value['counts_exact_at_exit'])
            self.assertEqual(value['forward_counts'],{'attempted':None,'completed':None})
            self.assertEqual(value['last_durable_counts']['attempted'],1)

    def test_owned_cache_cleanup_requires_confirmed_exit_and_matching_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'cache';cache=base/self.commit/'owned'
            ownership.create_owned_cache(cache,self.commit,self.token,base)
            (cache/'fixture.data').write_bytes(b'preserve until exit')
            for exited,token in [(False,self.token),(None,self.token),(True,'c'*32)]:
                with self.assertRaises(PermissionError):ownership.cleanup_owned_cache(cache,self.commit,token,exited,base)
                self.assertTrue((cache/'fixture.data').exists())
            with self.assertRaises(ValueError):ownership.cache_target(root/'outside',self.commit,base)
            self.assertTrue(ownership.cleanup_owned_cache(cache,self.commit,self.token,True,base)['owned_cache_removed'])

    def test_external_timeout_confirms_actual_child_exit_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);observed={}
            result=run_logged([sys.executable,'-c','import time; time.sleep(10)'],root,dict(os.environ),root/'log',0.05,
                              exit_observer=observed.update)
            self.assertTrue(result['timed_out'])
            self.assertTrue(observed['child_exit_confirmed'])
            self.assertIsNotNone(observed['exit_code'])

    def test_reader_start_failure_terminates_waits_and_observes_owned_child(self):
        with tempfile.TemporaryDirectory() as temp:
            observed={};events=[]
            class Process:
                pid=123;stdout=io.StringIO('fixture');returncode=None
                def poll(self):return self.returncode
                def kill(self):events.append('kill');self.returncode=-9
                def wait(self,timeout):events.append(('wait',timeout));return self.returncode
            process=Process()
            class Reader:
                ident=None
                def start(self):raise RuntimeError('fixture reader startup failure')
                def join(self,timeout):raise AssertionError('Cannot join an unstarted reader')
                def is_alive(self):return False
            with patch('src.composite_runtime.subprocess.Popen',return_value=process),patch('src.composite_runtime.threading.Thread',return_value=Reader()):
                with self.assertRaisesRegex(RuntimeError,'reader startup'):
                    run_logged(['never-launched-model'],temp,{},Path(temp)/'log',1,exit_observer=observed.update)
            self.assertEqual(events,['kill',('wait',15)])
            self.assertTrue(observed['child_exit_confirmed']);self.assertEqual(observed['exit_code'],-9)
            self.assertTrue(process.stdout.closed)

    def test_parent_never_cleans_cache_or_touches_old_output_until_exit_and_owner_confirmed(self):
        for exited in [True,False]:
            with self.subTest(exited=exited),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source';(source/'config').mkdir(parents=True)
                (source/'config/composite_controller_rank32_v2.json').write_text('{"qwen":{}}')
                base=root/'output';target=base/self.commit/'qualification';target.mkdir(parents=True)
                (target/'preserved.txt').write_bytes(b'older output must survive')
                cache_base=root/'cache';cache=cache_base/self.commit/'owned';controllers=root/'controllers'
                pin={'mode':worker.MODE,'notebook':'farhad89/synthetic-only','destination':str(target),
                     'cache_root':str(cache),'controller_root':str(controllers/self.commit/'one'),
                     'disk_policy':worker.DISK_POLICY,'input_root':'synthetic-only','transport_manifest_sha256':'c'*64}
                services=launch.LaunchServices();services.authorize=lambda *args:{'pin':pin}
                services.runtime=lambda:None;services.clock=lambda:0;services.output_base=base;services.controller_base=controllers
                services.create_cache=lambda path,commit,token:ownership.create_owned_cache(path,commit,token,cache_base)
                services.cleanup=lambda path,commit,token,exited:ownership.cleanup_owned_cache(path,commit,token,exited,cache_base)
                def runner(*args,exit_observer):
                    exit_observer({'child_started':True,'child_exit_confirmed':exited,'exit_code':1 if exited else None})
                    if not exited:raise RuntimeError('fixture exit confirmation failed')
                    return {'exit_code':1,'timed_out':False}
                services.runner=runner
                receipt=launch.launch(source,self.commit,'never-executed',0,services)
                self.assertFalse(receipt['complete_collection_exported'])
                self.assertEqual(cache.exists(),not exited)
                self.assertEqual((target/'preserved.txt').read_bytes(),b'older output must survive')
                self.assertFalse((target/'controller_rescue').exists())
                self.assertFalse(receipt.get('output_owner_confirmed',False))
