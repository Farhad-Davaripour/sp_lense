"""V2 metadata recovery only: no real tokenizer/model/owned child or repeated science suite."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import support,input_reader,authority,preparation_owner

OLD=support.ROOT/'development/native_supervised_gate_evaluation_v1'
SCIENTIFIC=('audit_saved.py','authority.py','CHECKPOINT.json','core.py','counts.py','entry.py',
    'forward_trace.py','launch.py','learned_gate.py','loader.py','OWNED_IDENTITY.json','owned_production.py',
    'production_run.py','receiver.py','REUSED_SOURCES.json','science.py','setup_budget.py','windows_job.py','workflow.py')
CONSOLE='e449bce01f275cd08f3d4e64bb73b3b43ae845a0dbdb3e6131426e66537705e5'
def substitute(raw):
    return raw.replace(b'development/native_supervised_gate_preparation_v1',b'development/native_supervised_gate_preparation_v2')\
        .replace(b'development/native_supervised_gate_evaluation_v1',b'development/native_supervised_gate_evaluation_v2')

class MetadataTests(unittest.TestCase):
    def test_exact_scientific_sources_and_limits(self):
        for name in SCIENTIFIC:
            self.assertEqual((support.HERE/name).read_bytes(),(OLD/name).read_bytes(),name)
        self.assertEqual(support.ATTEMPT,'native_supervised_gate_evaluation_v2_attempt_001')
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,139837440)
        self.assertEqual(authority.LIMITS,{'loads':1,'forwards':120,'derivatives':32,'worker_seconds':1200,
            'audit_seconds':120,'shared_cleanup_seconds':15,'total_bytes':192*1024**2,'file_bytes':5*1024**2})
        self.assertIn('CHECKPOINT.json -text',(support.HERE/'.gitattributes').read_text())
        self.assertFalse((support.HERE/'root_release').exists())
        self.assertFalse((support.HERE/'ownership_attempt_001').exists())
        self.assertFalse((input_reader.PREP/'root_release').exists())
        self.assertFalse((input_reader.PREP/'preparation_attempt_001').exists())
        draft=json.loads((support.HERE/'RELEASE_DRAFT.json').read_bytes())
        self.assertFalse(draft['approved']);self.assertFalse(draft['preparation_authorized']);self.assertFalse(draft['model_authorized'])

    def test_only_explicit_metadata_deltas(self):
        for name in ('preparation_owner.py','test_admission.py'):
            self.assertEqual((support.HERE/name).read_bytes(),substitute((OLD/name).read_bytes()),name)
        old=(OLD/'support.py').read_bytes().replace(b'native_supervised_gate_evaluation_attempt_001',b'native_supervised_gate_evaluation_v2_attempt_001')
        self.assertEqual((support.HERE/'support.py').read_bytes(),old)
        old_config=json.loads((OLD/'PREPARATION_OWNED_IDENTITY.json').read_bytes())
        new_config=json.loads((support.HERE/'PREPARATION_OWNED_IDENTITY.json').read_bytes())
        old_config['console_sha256']=CONSOLE;self.assertEqual(new_config,old_config)
        old_reader=substitute((OLD/'input_reader.py').read_bytes()).replace(
            b'7d2fa6660c1d240c2541b6af93a14ab3d6245856a84cf265874d1234f6e5cf92',input_reader.PREPARATION_SOURCE_SHA256.encode())
        self.assertEqual((support.HERE/'input_reader.py').read_bytes(),old_reader)

    def test_console_fingerprint_and_signature_admission_metadata(self):
        receipt=json.loads((support.HERE/'CONHOST_IDENTITY_REVIEW.json').read_bytes())
        config=json.loads((support.HERE/'PREPARATION_OWNED_IDENTITY.json').read_bytes())
        self.assertEqual(receipt['signature_status'],'Valid')
        self.assertEqual(receipt['signer_subject'],'CN=Microsoft Windows, O=Microsoft Corporation, L=Redmond, S=Washington, C=US')
        self.assertEqual(receipt['signer_thumbprint'],'BAC13DF18B37E808208A39D3A54CCE975FAC8C1D')
        self.assertEqual(receipt['sha256'],config['console_sha256']);self.assertEqual(receipt['sha256'],CONSOLE)
        raw=Path(config['console_image']).read_bytes()
        self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(1003520,CONSOLE))
        self.assertNotEqual(config['console_sha256'],'7ced551ec8afab3391a3ede5442046558e4799dcfd631ac9252fe93929a3465e')
        # test_only_explicit_metadata_deltas proves live owner hash equality and
        # every retained handle/job/creation/image predicate is byte unchanged.

    def test_new_prepared_adapter_bridge(self):
        adapter=input_reader.adapter();base=support.HERE/'fixtures/preparation_bundle'
        inventory=json.loads((support.HERE/'FIXTURE_INVENTORY.json').read_bytes())
        for pin in inventory['files']:
            raw=(base/pin['path']).read_bytes();self.assertEqual((len(raw),support.sha(raw)),(pin['bytes'],pin['sha256']))
        text=(base/'TEXT_LOCK.json').read_bytes();lock=json.loads(text)
        self.assertEqual(lock['final_execution_binding']['namespace'],'development/native_supervised_gate_evaluation_v2')
        pins={name:support.sha((base/'preparation'/name).read_bytes()) for name in adapter.artifact_names()}
        release={'text_lock_sha256':support.sha(text),'preparation_files':pins,'inputs_sha256':pins['inputs.json'],
            'preparation_closure_sha256':support.sha((base/'PREPARATION_CLOSURE.json').read_bytes()),'source_freeze_sha256':'f'*64}
        data=json.loads((base/'preparation/inputs.json').read_bytes())
        fake_template=data['cases'][0]['input_binding']['chat_template_sha256'];original=input_reader.adapter
        def fake_adapter():
            value=original();value.TEMPLATE_SHA256=fake_template;return value
        with patch.object(input_reader,'adapter',fake_adapter):
            self.assertEqual(input_reader.read_bundle(base,release,synthetic=True),data)
            altered=copy.deepcopy(release);altered['preparation_files']['inputs.json']='0'*64
            with self.assertRaises(ValueError):input_reader.read_bundle(base,altered,synthetic=True)
        self.assertEqual(len(data['cases']),16)
        self.assertEqual(json.loads((base/'preparation/RESULT.json').read_bytes())['planned_operations'],209)
        with patch.object(input_reader,'PREPARATION_SOURCE_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MANIFEST_HASH'):input_reader.adapter()
        with patch.dict(sys.modules,{'renderer':type('Wrong',(),{'__file__':'wrong'})()}):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MODULE_COLLISION'):input_reader.adapter()

    def test_no_provider_modules(self):
        self.assertFalse({'torch','transformers','tokenizers','safetensors','datasets','pyarrow'} & {n.split('.')[0] for n in sys.modules})

if __name__=='__main__':unittest.main(verbosity=2)
