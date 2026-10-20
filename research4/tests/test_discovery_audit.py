"""Synthetic transport fixtures only; no model calls or scientific outcomes."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from src.discovery_audit import audit
from src.discovery_runtime import initial_messages,pilot_cases,summarize
from src.discovery_sandbox import World
from src.io_utils import dump,read,sha,digest


class DiscoveryAuditTests(unittest.TestCase):
    def fixture(self,root):
        cfg=read(Path(__file__).parents[1]/'config/discovery_redteam_v1.json')
        cases=pilot_cases(cfg);episodes=[];calls=[]
        for case in cases:
            world=World(case);messages=initial_messages(world)
            action={'tool':'stop','arguments':{}};text=json.dumps(action)
            payload={'messages':messages,'max_tokens':cfg['pilot']['maximum_output_tokens'],'temperature':cfg['pilot']['temperature']}
            observation=world.observation();reply=world.step(action)
            calls.append({'case_id':case.case_id,'request_sha256':digest(payload),'status':'complete','response':{'choices':[{'message':{'content':text}}]}})
            episodes.append({'case':asdict(case),'steps':[{'observation':observation,'request_sha256':digest(payload),'raw_text':text,'parsed_action':action,'reply':reply}],'engine_trace':world.history,'outcome':world.outcome()})
        dump(root/'PRECALL_FREEZE.json',{'config':cfg})
        dump(root/'MODEL_EPISODES.json',episodes);dump(root/'MODEL_CALLS.json',calls)
        dump(root/'PILOT_CASES_PRIVATE.json',[asdict(c) for c in cases]);dump(root/'PILOT_MEMBERSHIP.json',{'case_ids':[c.case_id for c in cases]})
        dump(root/'RESULTS.json',{'model':summarize(episodes)})
        for name in ['ENDPOINT_BEFORE.json','ENDPOINT_AFTER.json']:dump(root/name,{'config':{'version':1}})
        dump(root/'RUN_RECEIPT.json',dict(status='complete',mode='hosted_pilot',model_api_calls=8,backbone_forwards=None,internal_gate_model_results=None,**{k:0 for k in ['optimizer_updates','research_adapters_loaded','real_resource_actions','heldout_cases_opened','GPU_allocations']}))
        self.manifest(root)

    def manifest(self,root):dump(root/'MANIFEST.json',{p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='MANIFEST.json'})

    def test_transport_fixture_replays_without_model(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'fixture';root.mkdir();self.fixture(root)
            self.assertTrue(audit(root,Path(t)/'audit.json')['audit_pass'])

    def test_duplicate_case_rejected_even_with_rebuilt_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'fixture';root.mkdir();self.fixture(root)
            episodes=read(root/'MODEL_EPISODES.json');episodes[1]=episodes[0];dump(root/'MODEL_EPISODES.json',episodes);self.manifest(root)
            with self.assertRaisesRegex(ValueError,'allocation'):audit(root,Path(t)/'audit.json')

    def test_consistent_but_false_request_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'fixture';root.mkdir();self.fixture(root)
            calls=read(root/'MODEL_CALLS.json');episodes=read(root/'MODEL_EPISODES.json')
            calls[0]['request_sha256']='f'*64;episodes[0]['steps'][0]['request_sha256']='f'*64
            dump(root/'MODEL_CALLS.json',calls);dump(root/'MODEL_EPISODES.json',episodes);self.manifest(root)
            with self.assertRaisesRegex(ValueError,'input differs'):audit(root,Path(t)/'audit.json')

    def test_finish_reason_cannot_be_changed_in_step_log(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'fixture';root.mkdir();self.fixture(root)
            episodes=read(root/'MODEL_EPISODES.json');episodes[0]['steps'][0]['finish_reason']='length'
            dump(root/'MODEL_EPISODES.json',episodes);self.manifest(root)
            with self.assertRaisesRegex(ValueError,'stopping trace'):audit(root,Path(t)/'audit.json')


if __name__=='__main__':unittest.main()
