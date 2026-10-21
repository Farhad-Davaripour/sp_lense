import unittest,json,tempfile,hashlib
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from src.composite_admission import validate_admission,open_heldout,frozen
from src.io_utils import dump,digest,sha
from src.composite_training import auxiliary_digest
import torch


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        cfg={'cohorts':{'validation_groups':1,'conditions':['quiet'],'validation_templates':[3]},'evaluation':{'conditions':['base','learned_dynamic']}}
        package={'synthetic':'unit-only'}
        report={'role':'validation','complete':True,'admitted':True,'raw_evidence_audited':True,'package_sha256':digest(package),'observed_trajectories':2,'missing':[],'families':{'3':{'admitted':True,'conditions':{'base':{},'learned_dynamic':{}}}}}
        return cfg,package,report

    def test_complete_validation_identity_and_controls_required(self):
        cfg,package,report=self.fixture();validate_admission(report,package,cfg)
        for key,value in [('admitted',False),('complete',False),('raw_evidence_audited',False),('role','heldout'),('observed_trajectories',1),('package_sha256','changed')]:
            bad={**report,key:value}
            with self.assertRaises(PermissionError):validate_admission(bad,package,cfg)
        bad=deepcopy(report);del bad['families']['3']['conditions']['base']
        with self.assertRaises(PermissionError):validate_admission(bad,package,cfg)

    def test_failed_validation_or_changed_source_never_reads_seal(self):
        cfg,package,report=self.fixture()
        for change in ['failed_validation','changed_source']:
            with tempfile.TemporaryDirectory() as scratch:
                root=Path(scratch);dump(root/'config/composite_controller_v1.json',cfg);release={'commit':'synthetic-unit','files':{}};dump(root/'RELEASE.json',release)
                value={'validation_report':deepcopy(report),'package':package,'package_sha256':digest(package),'validation_report_sha256':digest(report),'source_commit':release['commit'],'runtime_files_digest':digest({}),'protocol_sha256':sha(root/'config/composite_controller_v1.json')}
                if change=='failed_validation':value['validation_report']['admitted']=False
                else:value['source_commit']='changed'
                raw=json.dumps(value).encode()
                class Store:
                    def __init__(self):self.reads=[]
                    def get(self,path):
                        self.reads.append(path)
                        if path.startswith('freezes/'):return raw
                        raise AssertionError('A secret was read before admission')
                store=Store();descriptor={'path':'freezes/'+'a'*64+'/FREEZE.json','sha256':hashlib.sha256(raw).hexdigest()}
                with patch('src.composite_admission.require_databricks'):
                    with self.assertRaises(PermissionError):open_heldout(root,store,descriptor)
                self.assertEqual(store.reads,[descriptor['path']])

    def test_bad_descriptor_rejected_without_any_storage_read(self):
        class Store:
            def get(self,path):raise AssertionError('Unexpected storage access')
        with self.assertRaises(PermissionError):frozen(Path('.'),Store(),{'path':'sealed/HELDOUT_SEAL.json','sha256':'a'*64})

    def test_auxiliary_fingerprint_covers_quantization_state_and_buffers(self):
        parameter=torch.nn.Parameter(torch.ones(2),requires_grad=False);quant=torch.tensor([1.,2.]);buffer=torch.tensor([3.])
        class State:
            def as_dict(self,packed):return {'nested':quant,'kind':'unit'}
        parameter.quant_state=State();parameters=[('base',parameter)];buffers=[('buffer',buffer)]
        initial=auxiliary_digest(parameters,buffers);quant.add_(1)
        self.assertNotEqual(initial,auxiliary_digest(parameters,buffers));quant.sub_(1);buffer.add_(1)
        self.assertNotEqual(initial,auxiliary_digest(parameters,buffers))


if __name__=='__main__':unittest.main()
