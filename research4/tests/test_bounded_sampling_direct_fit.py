"""Public-only direct waiver/source/hash tests; no study rows/models/provider."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile,unittest,zipfile
from unittest.mock import patch
from src.io_utils import digest
from src.composite_bounded_sampling_fit_profile import DIRECT_MODE,WAIVER,WAIVER_SHA,direct_full_fit_plan
from src.composite_bounded_sampling_fit_source import validate_waiver_value,waiver
from tests.test_bounded_sampling_fit import public_fixture


def fake_waiver(plan):
    return {'version':'research4_direct_fresh64fit_human_qualification_waiver_v1',
        'source_user_message_id':'01a1794c-88cc-7e41-b333-1d4f3fbb99a6',
        'requires_existing_qualifier_actual_terminal_release':True,
        'separate_qualification_waived_not_passed':True,'scientific_full_fit_admitted_by_receipt_alone':False,
        'qualification_run_id':362573972,'approved_data_source_commit':'a'*40,
        'approved_data_raw_sha256':'b'*64,'approved_draw_plan_sha256':digest(plan),
        'approved_data_run_id':123}


class DirectFitTests(unittest.TestCase):
    def test_historical_reference_reads_hashes_only_and_rejects_mutated_archive(self):
        from src.composite_bounded_sampling_fit_parent import historical_initialization_reference
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'public_reference.zip'
            hashes={'public_projection_'+str(i)+'.'+part:hashlib.sha256((str(i)+part).encode()).hexdigest()
                    for i in range(496) for part in ['A','B']}
            raw=json.dumps(hashes).encode()
            with zipfile.ZipFile(path,'x') as stream:
                stream.writestr('ADAPTER_INITIAL_HASHES.json',raw)
                stream.writestr('MANIFEST.json',json.dumps({'ADAPTER_INITIAL_HASHES.json':hashlib.sha256(raw).hexdigest()}))
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            failed=SimpleNamespace(receipt=lambda:{'dependency_content_hashes':{'qualification':expected}})
            data=SimpleNamespace(failed_reference=lambda:failed)
            with patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value={'public_fixture':True}):
                descriptor,observed=historical_initialization_reference(data,path)
                self.assertEqual(observed,hashes)
                self.assertFalse(descriptor['current_training_capacity_qualified'])
                self.assertFalse(descriptor['historical_checkpoint_or_optimizer_loaded'])
                self.assertEqual(descriptor['initial_adapter_hashes_sha256'],digest(hashes))
                with path.open('ab') as stream:stream.write(b'changed')
                with self.assertRaises(PermissionError):historical_initialization_reference(data,path)

    def test_direct_plan_preserves64_and128_without_synthetic_qualification_parent(self):
        rows,plan,cfg,data,_=public_fixture()
        data.data_release=lambda:{'public_fixture':True}
        report={'raw_archive_sha256':'b'*64,'source_commit':'a'*40,'provider_run_id':123,
                'approval_sha256':'c'*64,'rule_sha256':'d'*64}
        approved=fake_waiver(plan)
        with patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value=report),\
             patch('src.composite_bounded_sampling_fit_parent.historical_initialization_reference',return_value=({'public_reference':True},{})):
            value=direct_full_fit_plan('e'*40,data,approved,'public_fake_immutable_reference.zip')
            self.assertEqual(value['version'],DIRECT_MODE)
            self.assertEqual((value['optimizer_updates'],value['accumulation']),(64,2))
            self.assertEqual(value['approved_draw_plan_sha256'],digest(plan))
            self.assertTrue(value['separate_qualification_waived']);self.assertFalse(value['qualification_performed'])
            self.assertFalse(value['stateful_AdamW_overlap_and_size_switch_required'])
            self.assertEqual(value['direct_fit_waiver_sha256'],WAIVER_SHA)
            for name in ['qualification_source_commit','qualification_run_id','qualification_checkpoint_sha256']:
                self.assertNotIn(name,value)
            changed=deepcopy(approved);changed['approved_draw_plan_sha256']='0'*64
            with self.assertRaises(PermissionError):direct_full_fit_plan('e'*40,data,changed)
            with self.assertRaises(PermissionError):validate_waiver_value({**approved,'separate_qualification_waived_not_passed':False},data)

    def test_direct_schedule_cannot_fall_back_to_uniform_or_qualifier_override(self):
        from src.composite_bounded_sampling_training import approved_full_fit_rows
        from src.composite_behavior import fit_selected_rows
        rows,plan,cfg,data,stage=public_fixture();stage['version']=DIRECT_MODE
        actor=SimpleNamespace(v2_training_plan=stage,bounded_training_context={
            'bounded_data_evidence':data,'stage':stage,'full_config':cfg})
        with patch('src.composite_bounded_sampling_reader.require_approved_sampling_data',return_value={'public_fixture':True}):
            self.assertEqual([r['state_id'] for r in approved_full_fit_rows(actor,rows,cfg)],plan['state_ids'])
            with tempfile.TemporaryDirectory() as folder,self.assertRaises(PermissionError):
                fit_selected_rows(actor,rows,{},cfg,folder,lambda:None,lambda *a,**k:None,
                                  qualification_rows=[rows[0]]*128)

    def test_direct_mode_cannot_satisfy_a_qualification_proof(self):
        from src.composite_bounded_sampling_fit_audit import require_qualification
        with self.assertRaises(PermissionError):require_qualification({'runtime_profile':DIRECT_MODE,'phase':'full_fit'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'inputs').mkdir();(root/'inputs'/WAIVER).write_text('{}')
            with self.assertRaises(PermissionError):waiver(root)


if __name__=='__main__':unittest.main()
