from types import SimpleNamespace
from pathlib import Path
import io,json,hashlib,tempfile,unittest,zipfile
from unittest.mock import patch
import torch
from src.composite_v2_raw_exports import (PROFILE,RAW_BYTES,RESERVE_BYTES,validate_export_plan,
    disk_admission,provider_binding,write_optimizer_state,BypassWriter)
from src.composite_v2_parent_audit import extract_source,qualification_from_descriptor
from src.composite_v2_model_audit import audit_optimizer_bytes


def plan():
    return {'source_commit':'a'*40,'phase':'qualification','destination':'/Volumes/unit/qualification',
            'raw_evidence_profile':PROFILE,'maximum_raw_evidence_bytes':RAW_BYTES,
            'physical_export_reserve_bytes':RESERVE_BYTES}


class RawExportTests(unittest.TestCase):
    def test_full_raw_payload_and_disk_envelope_are_admitted_without_allocating(self):
        self.assertGreater(RAW_BYTES,233455616*12)
        self.assertEqual(validate_export_plan(plan())['physical_export_reserve_bytes'],9*1024**3)
        pin={'files':[{'size':55575985859}]}
        result=disk_admission('.', '.',pin,plan(),usage=lambda path:SimpleNamespace(free=100*1024**3))
        self.assertFalse(result['runtime_capacity_admitted'])
        with self.assertRaises(OSError):disk_admission('.', '.',pin,plan(),usage=lambda path:SimpleNamespace(free=1))
        bad=plan();bad['maximum_raw_evidence_bytes']=1024**3
        with self.assertRaises(PermissionError):validate_export_plan(bad)

    def test_provider_binding_is_exact_and_is_not_release_or_allocation_authority(self):
        stage=plan();workspace={'host':'https://adb-7405616135490449.9.azuredatabricks.net'}
        value={k:stage[k] for k in ['source_commit','phase','destination']};value.update(host=workspace['host'],run_id=123)
        self.assertEqual(provider_binding(stage,workspace,value),123)
        for change in [{'run_id':True},{'host':'other'},{'phase':'full_fit'},{'destination':'other'}]:
            with self.assertRaises(PermissionError):provider_binding(stage,workspace,{**value,**change})

    def test_optimizer_export_reads_existing_tiny_state_without_stepping(self):
        a=torch.nn.Parameter(torch.ones(2,2));b=torch.nn.Parameter(torch.ones(2,2))
        optimizer=SimpleNamespace(state={p:{'step':torch.tensor(1.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.ones_like(p)} for p in [a,b]})
        with tempfile.TemporaryDirectory() as folder:
            proof=write_optimizer_state(optimizer,{'p':SimpleNamespace(A=a,B=b)},folder)
            raw=(Path(folder)/proof['path']).read_bytes()
            self.assertEqual(audit_optimizer_bytes(raw,{'p.A':[2,2],'p.B':[2,2]},1)['raw_AdamW_tensors_verified'],4)
            with self.assertRaises(FileExistsError):write_optimizer_state(optimizer,{'p':SimpleNamespace(A=a,B=b)},folder)

    def test_bypass_writer_exports_only_existing_finite_logits_and_requires_all_six(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=BypassWriter(folder)
            writer(0,'wrapped_OFF',[1,2],torch.ones(1,1,3))
            with self.assertRaises(ValueError):writer.manifest()
            for index in [0,1]:
                for kind in ['wrapped_OFF','bare','wrapped_ON']:
                    if index==0 and kind=='wrapped_OFF':continue
                    writer(index,kind,[1,2],torch.ones(1,1,3))
            self.assertEqual(writer.manifest()['logit_shape'],[1,1,3])
            with self.assertRaises(FileExistsError):writer(0,'bare',[1,2],torch.ones(1,1,3))
            with self.assertRaises(ValueError):writer(2,'bare',[1,2],torch.ones(1,1,3))

    def test_source_extraction_rejects_traversal_and_never_executes_code(self):
        def archive(name,release):
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                z.writestr(name,b'raise RuntimeError("source code must never run")')
                z.writestr('RELEASE.json',json.dumps(release))
            return out.getvalue()
        name='src/public.py';raw=archive(name,{'files':{name:'a'*64}})
        with tempfile.TemporaryDirectory() as folder:
            root=extract_source(raw,hashlib.sha256(raw).hexdigest(),Path(folder)/'source')
            self.assertTrue((root/name).is_file())
        for name in ['../escape','.env','inputs/HELDOUT_SEAL.json']:
            raw=archive(name,{'files':{}})
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(PermissionError):extract_source(raw,hashlib.sha256(raw).hexdigest(),Path(folder)/'source')
        raw=archive('src/public.py',{'files':{'../../private':'a'*64}})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PermissionError):extract_source(raw,hashlib.sha256(raw).hexdigest(),Path(folder)/'source')

    def test_serialized_qualification_boolean_fails_before_any_store_access(self):
        stage={'source_commit':'a'*40,'data_release_sha256':'b'*64,'qualification':{'complete_training_qualification_audited':True}}
        store=SimpleNamespace(get=lambda *a:(_ for _ in ()).throw(AssertionError('must not read store')))
        with self.assertRaises(PermissionError):qualification_from_descriptor(stage,{},store,'.',b'public fixture')


if __name__=='__main__':unittest.main()
