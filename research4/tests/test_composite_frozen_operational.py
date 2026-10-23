import base64,copy,hashlib,json,tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src import composite_frozen_operational as operation
from src.composite_frozen_operational_audit import audit_maintenance
from src.composite_cuda_maintenance import UnusedAllocatorReclaimer
from src.io_utils import sha
import test_composite_cuda_maintenance as maintenance_fixtures
from src.composite_frozen_operational_audit import audit_operational_evidence,aggregate_versioned
from src import composite_frozen_operational_audit as operational_audit


class OperationalTests(unittest.TestCase):
    def test_explicit_new_source_relationship_and_unfinished_parent_are_never_identical_freeze(self):
        value=operation.document('a'*40,3)
        self.assertEqual(value['source_relationship'],operation.RELATIONSHIP)
        self.assertTrue(value['tool_boundary_continuation']);self.assertEqual(value['preserved_parent_native_responses'],14)
        self.assertFalse(value['exact_interrupted_generation_resume']);self.assertFalse(value['original_uniform_source_aggregate_allowed'])
        self.assertFalse(operation.document('a'*40,4)['tool_boundary_continuation'])
        for source,number in [(operation.ORIGINAL_SOURCE,3),('bad',3),('a'*40,2),('a'*40,6)]:
            with self.assertRaises(PermissionError):operation.document(source,number)

    def test_operational_transport_only_adds_the_sealed_parent_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=json.dumps({'version':'frozen-base-kaggle-inputs-v1','files':{'original.txt':'e'*64}}).encode()
            manifest={'version':'frozen-base-operational-inputs-v2','original_base_transport_manifest_bytes':base64.b64encode(raw).decode(),
                'files':{'original.txt':'e'*64,operation.PARENT_ARCHIVE:operation.PARENT_ARCHIVE_SHA256},'heldout_transferred':False}
            with patch.object(operation,'PARENT_TRANSPORT_SHA256',hashlib.sha256(raw).hexdigest()):
                (root/'TRANSPORT_MANIFEST.json').write_text(json.dumps(manifest))
                self.assertEqual(operation.validate_operational_transport(SimpleNamespace(root=root)),manifest['files'])
                for kind in ['drop','extra','heldout']:
                    value=copy.deepcopy(manifest)
                    if kind=='drop':del value['files']['original.txt']
                    elif kind=='extra':value['files']['unregistered']='f'*64
                    else:value['heldout_transferred']=True
                    (root/'TRANSPORT_MANIFEST.json').write_text(json.dumps(value))
                    with self.assertRaises(ValueError):operation.validate_operational_transport(SimpleNamespace(root=root))

    def test_runtime_descriptor_binds_parent_bytes_and_unchanged_scientific_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/'inputs';inputs.mkdir();core=root/'stable.py';core.write_bytes(b'original scientific fixture')
            prefix={'partial_prefix_audited':True,'complete_base_partition_audited':False,
                'sealed_forward_counts':{'attempted':1260,'completed':1260},'native_complete_responses':14,
                'additional_interrupted_response_forwards':1,'exact_interrupted_generation_resume_supported':False}
            closeout={'run_id':operation.PARENT_RUN,'source_commit':operation.ORIGINAL_SOURCE,
                      'owned_compute_released':True,'owned_cache_removed':True}
            release={'commit':operation.ORIGINAL_SOURCE,'files':{'stable.py':sha(core)}}
            for name,value in [(operation.PARENT_PREFIX_AUDIT,prefix),(operation.PARENT_CLOSEOUT,closeout),(operation.PARENT_RELEASE,release)]:
                (inputs/name).write_text(json.dumps(value))
            with patch.multiple(operation,PARENT_PREFIX_AUDIT_SHA256=sha(inputs/operation.PARENT_PREFIX_AUDIT),
                PARENT_CLOSEOUT_SHA256=sha(inputs/operation.PARENT_CLOSEOUT),PARENT_RELEASE_SHA256=sha(inputs/operation.PARENT_RELEASE),CORE_FILES={'stable.py':sha(core)}):
                document=operation.document('a'*40,3);(inputs/operation.DOCUMENT).write_text(json.dumps(document))
                authority={'operational_profile':operation.PROFILE,'operational_document_sha256':sha(inputs/operation.DOCUMENT)}
                pin={'operational_profile':operation.PROFILE,'partition':3}
                self.assertEqual(operation.validate_runtime_profile(root,'a'*40,pin,authority),document)
                core.write_bytes(b'changed scientific fixture')
                with self.assertRaisesRegex(ValueError,'scientific'):operation.validate_runtime_profile(root,'a'*40,pin,authority)

    def test_maintenance_replay_requires_idle_unchanged_live_tensors_and_original_guard(self):
        cuda=maintenance_fixtures.FakeCUDA();values={}
        actor=SimpleNamespace(frozen_analysis_only=True,adapters={},generating=False,kernel_profile='reference',clear_cache=lambda:None)
        UnusedAllocatorReclaimer(cuda,lambda name,value:values.__setitem__(name,value)).before_generation(actor)
        class Files:
            def json(self,name):return {key:'sha' for key in values} if name=='MANIFEST.json' else values[name]
        report=audit_maintenance(Files(),1)
        self.assertTrue(report['successful_reclaimability_observed'])
        self.assertTrue(report['original512MiBguard_retained'])
        for kind in ['guard','live','model_forward','missing']:
            original=copy.deepcopy(values)
            if kind=='guard':values['maintenance/reclaim_00000.json']['minimum_driver_free_bytes']=1
            elif kind=='live':values['maintenance/reclaim_00000.json']['after'][0]['allocated_bytes']+=1
            elif kind=='model_forward':values['maintenance/reclaim_00000.json']['model_forwards']=1
            else:values.clear()
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit_maintenance(Files(),1)
            values.clear();values.update(original)

    def test_full_synthetic_unstarted_partition_replays_current_forwards_and_maintenance(self):
        from test_composite_frozen_base_audit import BaseAuditTests
        from src.composite_frozen_base_audit import replay
        files,context,tokenizers,values,raw=BaseAuditTests().fixture()
        records,replayed=replay(files,context,{},tokenizers,{3})
        operation_profile=operation.document('a'*40,4)
        context.update(operational_profile=operation_profile,source_commit='a'*40)
        cuda=maintenance_fixtures.FakeCUDA()
        actor=SimpleNamespace(frozen_analysis_only=True,adapters={},generating=False,kernel_profile='reference',clear_cache=lambda:None)
        maintenance=UnusedAllocatorReclaimer(cuda,lambda name,value:values.__setitem__(name,value))
        for _ in range(replayed['native_responses']+1):maintenance.before_generation(actor)
        class Files:
            def json(self,name):return {key:'sha' for key in values} if name=='MANIFEST.json' else files.json(name)
            def get(self,name):return files.get(name)
        report=audit_operational_evidence(Files(),context,{'operational_profile':operation_profile},
            records,replayed,None,tokenizers[1],{3})
        self.assertEqual(report['expected_current_run_forwards'],replayed['expected_joint_forwards'])
        self.assertEqual(report['preserved_parent_native_responses'],0)
        self.assertFalse(report['original_uniform_source_aggregate_allowed'])
        del values['maintenance/reclaim_00000.json']
        with self.assertRaises(ValueError):audit_operational_evidence(Files(),context,{'operational_profile':operation_profile},
            records,replayed,None,tokenizers[1],{3})

    def test_descriptive_aggregate_requires_all_original_cases_and_discloses_both_sources(self):
        from src.composite_exports import make_evidence_archive
        from src.composite_world import fixture
        from src.composite_frozen_base import partition_plan
        from src.io_utils import read
        scenes=[asdict(fixture('quiet_pending',3,seed='synthetic-versioned-aggregate-'+str(i))) for i in range(36)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sources=[root/'old',root/'new']
            cfg=read(Path(__file__).resolve().parents[1]/'config/composite_controller_rank32_v2.json')
            for source in sources:
                (source/'config').mkdir(parents=True)
                (source/'config/composite_controller_rank32_v2.json').write_text(json.dumps(cfg))
            contexts=[{'scenes':scenes,'pin':{'transport_manifest_sha256':'b'*64}},
                {'scenes':scenes,'pin':{'transport_manifest_sha256':'c'*64},'operational_profile':{'profile':operation.PROFILE}}]
            closeouts=[]
            for number in range(6):
                folder=root/str(number);folder.mkdir();raw=folder/'raw';raw.mkdir()
                plan=partition_plan(scenes,number);(raw/'trajectories').mkdir()
                for scene in scenes[number*6:number*6+6]:
                    (raw/'trajectories'/(scene['case_id']+'.json')).write_text(json.dumps({'scene':scene,'condition':'base','outcome':{'running':False}}))
                archive=folder/'evidence.zip';make_evidence_archive(raw,archive)
                proof={'partition':number,'source_commit':operation.ORIGINAL_SOURCE if number<3 else 'a'*40,
                    'archive_sha256':sha(archive),'transport_manifest_sha256':('b' if number<3 else 'c')*64,
                    'complete_base_partition_audited':True,'case_ids':plan['case_ids']}
                if number>=3:proof['operational_version_replay']={'profile':operation.PROFILE,
                    'new_operational_source_commit':'a'*40,'original_scientific_source_commit':operation.ORIGINAL_SOURCE,
                    'original_uniform_source_aggregate_allowed':False,'completed_parent_responses_regenerated':0,
                    'preserved_parent_native_responses':14 if number==3 else 0}
                audit_file=folder/'audit.json';audit_file.write_text(json.dumps(proof))
                closeout=folder/'closeout.json';closeout.write_text(json.dumps({'owned_compute_released':True,
                    'complete_base_partition_audited':True,'audit_receipt':str(audit_file),'audit_sha256':sha(audit_file),
                    'archive':str(archive),'archive_sha256':sha(archive)}));closeouts.append(closeout)
            def authorize(source,*args):return contexts[0 if Path(source)==sources[0] else 1]
            with patch('src.composite_frozen_base.authorized_load_plan',side_effect=authorize),patch('src.composite_metrics.summarize',return_value={}):
                def call(items):return aggregate_versioned(items,*sources,'a'*40,'old-transport','new-transport')
                report=call(closeouts)
                self.assertTrue(report['complete_original_base_roster_observed'])
                self.assertTrue(report['operational_versions_disclosed'])
                self.assertFalse(report['original_uniform_source_aggregate_allowed'])
                self.assertFalse(report['full_validation_admitted'])
                with self.assertRaises(ValueError):call(closeouts[:5])
                with self.assertRaises(ValueError):call(closeouts[:5]+[closeouts[0]])
                partial=json.loads(closeouts[3].read_text());partial['complete_base_partition_audited']=False
                closeouts[3].write_text(json.dumps(partial))
                with self.assertRaises(ValueError):call(closeouts)

    def test_detector_terminal_overflow_never_invents_an_actor_maintenance_call(self):
        from test_composite_frozen_base_audit import BaseAuditTests
        from src.composite_frozen_base_audit import audit_trajectory
        for detector_overflow in [True,False]:
            with self.subTest(detector_overflow=detector_overflow):
                actor_attempts=[]
                record,bt,qt=BaseAuditTests().context_fixture(detector_overflow=detector_overflow,
                    actor_overflow=True,on_actor_attempt=actor_attempts.append)
                audit_trajectory(record,.9,bt,qt,{3},'reference',{'detector':2048,'actor':6144})
                self.assertEqual(len(record['trace']),1)
                self.assertEqual(len(actor_attempts),1 if detector_overflow else 2)
                self.assertGreater(len(qt.apply_chat_template([{'content':record['trace'][-1]['observation_after']['text']}])),6144)
                operation_profile=operation.document('a'*40,4)
                context={'operational_profile':operation_profile,'source_commit':'a'*40}
                values={};cuda=maintenance_fixtures.FakeCUDA()
                actor=SimpleNamespace(frozen_analysis_only=True,adapters={},generating=False,kernel_profile='reference',clear_cache=lambda:None)
                maintenance=UnusedAllocatorReclaimer(cuda,lambda name,value:values.__setitem__(name,value))
                # The original first-response parity contributes one call;
                # every actually entered actor.generate contributes another.
                for _ in range(len(actor_attempts)+1):maintenance.before_generation(actor)
                class Files:
                    def json(self,name):return {key:'sha' for key in values} if name=='MANIFEST.json' else values[name]
                replayed={'native_responses':1,'expected_joint_forwards':5 if detector_overflow else 6}
                report=audit_operational_evidence(Files(),context,{'operational_profile':operation_profile},
                    [record],replayed,None,qt,{3})
                self.assertEqual(report['maintenance']['maintenance_calls'],2 if detector_overflow else 3)
                saved=copy.deepcopy(values);maintenance.before_generation(actor)
                with self.assertRaises(ValueError):audit_operational_evidence(Files(),context,{'operational_profile':operation_profile},
                    [record],replayed,None,qt,{3})
                values.clear();values.update(saved);values.pop(sorted(values)[-1])
                with self.assertRaises(ValueError):audit_operational_evidence(Files(),context,{'operational_profile':operation_profile},
                    [record],replayed,None,qt,{3})

    def test_reader_release_identity_is_distinct_and_cannot_hide_changed_reader_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);module=root/'src/composite_frozen_operational_audit.py'
            module.parent.mkdir();module.write_bytes(b'synthetic reader identity fixture')
            with patch.object(operational_audit,'__file__',str(module)):
                identity=operational_audit.audit_reader_identity()
                self.assertEqual(identity['profile'],operational_audit.AUDIT_READER_PROFILE)
                self.assertIsNone(identity['source_commit']);self.assertFalse(identity['source_release_verified'])
                (root/'RELEASE.json').write_text(json.dumps({'commit':'a'*40,
                    'files':{'src/composite_frozen_operational_audit.py':sha(module)}}))
                identity=operational_audit.audit_reader_identity()
                self.assertEqual(identity['source_commit'],'a'*40);self.assertTrue(identity['source_release_verified'])
                self.assertEqual(identity['module_sha256'],sha(module))
                module.write_bytes(b'mutated reader bytes')
                with self.assertRaisesRegex(ValueError,'reader'):operational_audit.audit_reader_identity()


if __name__=='__main__':unittest.main()
