"""Focused closed-entry, console identity, and source-reuse seams; no child is launched."""
import ast,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import support,authority,preparation_owner,loader

class AdmissionTests(unittest.TestCase):
    def test_default_and_draft_remain_disabled(self):
        with patch.object(sys,'argv',['preparation_owner.py']):
            self.assertEqual(preparation_owner.main(),2)
        draft=support.HERE/'RELEASE_DRAFT.json'
        with patch.object(authority,'RELEASE',draft):
            with self.assertRaisesRegex(ValueError,'EXACT_NATIVE_RELEASE'):
                authority.read_release(support.sha(draft.read_bytes()))
        with patch.dict('os.environ',{'SP_NATIVE_RELEASE_SHA':''}):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_ROOT_RELEASE_HASH_REQUIRED'):
                loader.load(None,None,None,None)

    def test_preparation_owner_exact_console_config_and_command(self):
        owner=preparation_owner;owner_sha='f'*64;observed=[]
        lock={'preparation_owner_binding':{'namespace':'development/native_supervised_gate_evaluation_v2',
            'source_freeze_sha256':owner_sha}}
        text=support.json_bytes(lock)
        release={'approved':True,'scope':'ONE_OFFLINE_FINAL_PREPARATION','text_lock_sha256':support.sha(text)}
        raw=support.json_bytes(release);release_sha=support.sha(raw)
        original=Path.read_bytes
        def read(path):
            if path==owner.PREP/'root_release/PREPARATION_RELEASE.json':return raw
            if path==owner.PREP/'root_release/TEXT_LOCK.json':return text
            if path.name.endswith('OWNED_IDENTITY.json'):observed.append(path)
            return original(path)
        calls=[]
        def fake_run(command,config,out,prep_out,identity,**kwargs):
            calls.append((command,config,out,prep_out,identity))
            self.assertIn('console_image',config);self.assertIn('console_sha256',config)
            self.assertEqual(command,[config['launch_image'],'-B',str(owner.PREP/'prepare_offline.py'),
                '--approved-preparation-sha256',release_sha])
            self.assertEqual(out,support.HERE/'ownership_attempt_001')
            self.assertEqual(prep_out,owner.PREP/'preparation_attempt_001')
            return {'status':'PASS','quiescent':True,'exit_code':0,'timed_out':False,'synthetic_only':True}
        with patch.object(owner,'verify',lambda expected:owner_sha),patch.object(owner,'run',fake_run),\
             patch.object(Path,'read_bytes',read),patch.object(sys,'argv',
                ['preparation_owner.py','--owner-source-sha256',owner_sha,'--approved-preparation-sha256',release_sha]):
            self.assertEqual(owner.main(),0)
        self.assertEqual(observed,[support.HERE/'PREPARATION_OWNED_IDENTITY.json'])
        self.assertEqual(len(calls),1)

    def test_exact_unchanged_native_sources_and_compile(self):
        old=support.ROOT/'development/native_oracle_confirmation_execution_v1'
        for name in ('receiver.py','core.py','loader.py','entry.py','owned_production.py','forward_trace.py',
                     'launch.py','CHECKPOINT.json','OWNED_IDENTITY.json','REUSED_SOURCES.json'):
            self.assertEqual((support.HERE/name).read_bytes(),(old/name).read_bytes(),name)
        owner=support.ROOT/'development/native_oracle_confirmation_preparation_owner_v1'
        self.assertEqual((support.HERE/'windows_job.py').read_bytes(),(owner/'windows_job.py').read_bytes())
        for path in support.HERE.glob('*.py'):ast.parse(path.read_bytes(),filename=str(path))
        self.assertFalse({'torch','transformers','tokenizers','safetensors'} & {x.split('.')[0] for x in sys.modules})

if __name__=='__main__':unittest.main(verbosity=2)
