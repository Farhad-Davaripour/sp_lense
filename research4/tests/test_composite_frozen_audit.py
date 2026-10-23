import copy
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src import composite_frozen_audit as audit
from src.composite_exports import make_evidence_archive
from src.composite_frozen_probe import MODE,BOUNDS
from src.composite_frozen_worker import DISK_POLICY
from src.io_utils import sha


class FrozenAuditTests(unittest.TestCase):
    def test_token_replay_reconstructs_only_verified_metadata_locally_then_removes_it(self):
        seen={}
        def tokenizer(path,**kwargs):
            seen.update(path=Path(path),kwargs=kwargs)
            self.assertEqual(json.loads((Path(path)/'generation_config.json').read_text())['eos_token_id'],[3,4])
            self.assertEqual(list(Path(path).glob('*.safetensors')),[])
            return SimpleNamespace(eos_token_id=5)
        record={'model_file_verification':[],'model_metadata_bytes':{
            'generation_config.json':base64.b64encode(b'{"eos_token_id":[3,4]}').decode()}}
        with patch.object(audit,'validate_download_proof'),patch.object(audit,'verify_metadata_bytes'),\
             patch('transformers.AutoTokenizer.from_pretrained',side_effect=tokenizer),\
             patch.object(audit,'audit_probe_record',return_value={'synthetic_metadata_only':True}) as replay:
            result=audit.audit_token_replay(record,[],{'model_pin':{}})
            self.assertTrue(result['synthetic_metadata_only']);self.assertEqual(replay.call_args.args[-1],{3,4,5})
            self.assertEqual(seen['kwargs'],{'local_files_only':True,'trust_remote_code':False})
        self.assertFalse(seen['path'].exists())

    def fixture(self,root,complete=False):
        source=root/'source';(source/'inputs').mkdir(parents=True);(source/'config').mkdir()
        (source/'inputs/MANIFEST.json').write_text('{}')
        (source/'config/composite_controller_rank32_v2.json').write_text('{"qwen":{}}')
        context={'pin':{'notebook':'farhad89/synthetic-only','transport_manifest_sha256':'d'*64}}
        raw=root/'raw';raw.mkdir()
        def write(name,value):(raw/name).write_text(json.dumps(value))
        write('PRECALL_FREEZE.json',{'source_commit':'a'*40,'mode':MODE,'pin':context['pin'],
            'input_manifest_sha256':sha(source/'inputs/MANIFEST.json'),'bounds':BOUNDS,'disk_policy':DISK_POLICY})
        receipt={'source_commit':'a'*40,'stage':MODE,'status':'complete' if complete else 'failed',
                 'heldout_content_read':False,'optimizer_updates':0,'behavior_adapter_fitting':False,
                 'forward_counts':{'attempted':4,'completed':4}}
        write('RUN_RECEIPT.json',receipt)
        counts={'counts_exact_at_exit':True,'forward_counts':receipt['forward_counts']}
        controller={'source_commit':'a'*40,'mode':MODE,'notebook':context['pin']['notebook'],
            'child_state':{'child_exit_confirmed':True,'exit_code':0 if complete else 1},
            'counts':counts,'cleanup':{'owned_cache_removed':True},'complete_collection_exported':complete}
        if complete:
            write('FORWARD_COUNTS.json',{**receipt['forward_counts'],'sealed_for_exit':True})
            record={'forward_counts':receipt['forward_counts'],'probes':[{'input_token_ids':[1]},{'input_token_ids':[2]}],
                    'parameters_before':{},'parameters_after':{},'model_file_verification':[], 'model_metadata_bytes':{}}
            write('FROZEN_QUALIFICATION_RECORD.json',record)
            for index,probe in enumerate(record['probes']):
                write('FROZEN_PROBE_'+str(index)+'.json',probe);write('FROZEN_PREFILL_'+str(index)+'.json',probe)
            for filename,key in [('FROZEN_PARAMETERS_BEFORE.json','parameters_before'),('FROZEN_PARAMETERS_AFTER.json','parameters_after'),
                                 ('MODEL_FILE_VERIFICATION.json','model_file_verification'),('MODEL_METADATA_BYTES.json','model_metadata_bytes')]:write(filename,record[key])
            record.update(loaded_layout={},loading_info={},data_provenance={})
            write('FROZEN_QUALIFICATION_RECORD.json',record)
            write('HARDWARE_PREFLIGHT.json',{})
            write('FROZEN_LOAD_FREEZE.json',{'hardware_preflight':{}})
            write('FROZEN_LOADED_LAYOUT.json',{'hardware_preflight':{},'loaded_layout':{},'loading_info':{},'model_files':[]})
            write('FROZEN_PROBE_SELECTION.json',{'conditions':['quiet_pending','own_pending'],'state_ids':[],
                'prompt_sha256':[],'provenance':{},'selected_before_forwards':True,'heldout_content_read':False})
        return source,raw,context,receipt,controller

    def seal(self,root,raw,receipt,controller):
        archive=root/'export.zip';proof=make_evidence_archive(raw,archive)
        export={**receipt,**proof};controller['evidence']={'evidence_kind':'original_worker_export',**proof}
        return archive,export

    def test_partial_failure_audits_integrity_without_tokenizer_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,raw,context,receipt,controller=self.fixture(root)
            archive,export=self.seal(root,raw,receipt,controller)
            with patch.object(audit,'authorized_load_plan',return_value=context),patch.object(audit,'audit_token_replay',side_effect=AssertionError('partial must not load tokenizer')):
                result=audit.audit_archive(archive,source,'a'*40,root,controller,export)
            self.assertFalse(result['complete_qualification_audited']);self.assertEqual(result['audit_model_forwards'],0)

    def test_complete_record_requires_matching_raw_bytes_and_exact_exit_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,raw,context,receipt,controller=self.fixture(root,True)
            archive,export=self.seal(root,raw,receipt,controller)
            with patch.object(audit,'authorized_load_plan',return_value=context),patch.object(audit,'FrozenInputs'),\
                 patch.object(audit,'load_training_rows',return_value=([],{})),patch.object(audit,'audit_runtime_metadata'),\
                 patch.object(audit,'select_training_probes',return_value=[]),\
                 patch.object(audit,'audit_token_replay',return_value={'synthetic_only':True}) as replay:
                result=audit.audit_archive(archive,source,'a'*40,root,controller,export)
                self.assertTrue(result['complete_qualification_audited']);self.assertFalse(result['full_validation_admitted']);replay.assert_called_once()
                changed=copy.deepcopy(controller);changed['counts']['forward_counts']['completed']=3
                with self.assertRaisesRegex(ValueError,'exact terminal'):audit.audit_archive(archive,source,'a'*40,root,changed,export)
                changed=copy.deepcopy(export);changed['optimizer_updates']=1
                with self.assertRaisesRegex(ValueError,'receipts differ'):audit.audit_archive(archive,source,'a'*40,root,controller,changed)

    def test_rescue_retains_unknown_final_counts_and_rejects_relabelled_unknown_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,raw,context,receipt,controller=self.fixture(root)
            (raw/'RUN_RECEIPT.json').unlink()
            controller['counts']={'counts_exact_at_exit':False,'forward_counts':{'attempted':None,'completed':None},'last_durable_counts':{'attempted':1,'completed':0}}
            (raw/'CONTROLLER_TERMINAL_OBSERVATION.json').write_text(json.dumps(controller))
            archive,_=self.seal(root,raw,receipt,controller);controller['evidence']['evidence_kind']='controller_post_exit_rescue'
            with patch.object(audit,'authorized_load_plan',return_value=context):
                result=audit.audit_archive(archive,source,'a'*40,root,controller)
                self.assertEqual(result['forward_counts'],{'attempted':None,'completed':None})
                controller['counts']['forward_counts']={'attempted':0,'completed':0}
                with self.assertRaisesRegex(ValueError,'remain unknown'):audit.audit_archive(archive,source,'a'*40,root,controller)

    def test_corrupt_archive_or_live_worker_cannot_receive_terminal_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,raw,context,receipt,controller=self.fixture(root)
            archive,export=self.seal(root,raw,receipt,controller)
            with patch.object(audit,'authorized_load_plan',return_value=context):
                controller['child_state']['child_exit_confirmed']=False
                with self.assertRaisesRegex(ValueError,'unconfirmed'):audit.audit_archive(archive,source,'a'*40,root,controller,export)
                controller['child_state']['child_exit_confirmed']=True;controller['evidence']['archive_sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'exact archive'):audit.audit_archive(archive,source,'a'*40,root,controller,export)
