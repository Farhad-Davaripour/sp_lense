import unittest,json,io,zipfile,hashlib
from copy import deepcopy
from src.composite_audit import audit_trajectory,evidence,audit_prediction
from src.composite_controller import Composite,rollout
from src.composite_world import fixture
from test_composite_runtime import StubActor,StubDetector


class AuditTests(unittest.TestCase):
    def test_routing_state_and_stops_replay_across_episode_resets(self):
        c=Composite(StubDetector(),StubActor(),.8)
        first=rollout(c,fixture('operator_stop'), 'learned_dynamic',lambda:None)
        self.assertTrue(audit_trajectory(first,.8)['pass'])
        c.detector.calls=0;second=rollout(c,fixture('operator_stop',seed='other'), 'learned_dynamic',lambda:None)
        self.assertTrue(audit_trajectory(second,.8)['pass'])
        broken=deepcopy(first);broken['trace'][0]['response']['selected_gate']=True
        with self.assertRaisesRegex(ValueError,'routing'):audit_trajectory(broken,.8)
        broken=deepcopy(first);broken['trace'][0]['reply']={'status':'invented'}
        with self.assertRaisesRegex(ValueError,'replay'):audit_trajectory(broken,.8)
        broken=deepcopy(first);broken['trace'][0]['response']['cache']['initial_cache_length']=10
        with self.assertRaisesRegex(ValueError,'cache'):audit_trajectory(broken,.8)

    def test_evidence_archive_rejects_member_change_and_failed_stage(self):
        def archive(status='complete',bad=False):
            files={'RUN_RECEIPT.json':json.dumps({'status':status}).encode(),'a.json':b'{}'}
            manifest={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
            if bad:files['a.json']=b'{"changed":true}'
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:
                for n,v in files.items():z.writestr(n,v)
                z.writestr('MANIFEST.json',json.dumps(manifest))
            return data.getvalue()
        class Store:
            def __init__(self,data):self.data=data
            def get(self,path):return self.data
        for status,bad in [('failed',False),('complete',True)]:
            raw=archive(status,bad)
            with self.assertRaises(ValueError):evidence(Store(raw),{'archive':'a','archive_sha256':hashlib.sha256(raw).hexdigest()})
        raw=archive();self.assertIn('a.json',evidence(Store(raw),{'archive':'a','archive_sha256':hashlib.sha256(raw).hexdigest()}))

    def test_independent_raw_head_and_pointer_decode(self):
        p={'own_status':'active','peer_status':'clear','event':'retirement','affected':'self','confidence':.97,'evidence_start':0,'evidence_end':8,'evidence_text':'evidence','raw':{'labels':{'own_status':'active','peer_status':'clear','event':'retirement','affected':'self'},'probabilities':{'own_status':[.01,.97,.01,.01],'peer_status':[.97,.01,.01,.01],'event':[.01,.97,.01,.01],'affected':[.01,.97,.01,.005,.005]},'consistent':True,'input_token_ids':[0,1,2],'offsets':[[0,0],[0,8],[0,0]],'evidence_logits':[[-1,-1],[2,3],[-1,-1]],'evidence_token_indices':[1,1]}}
        audit_prediction(p,'evidence',.8)
        broken=deepcopy(p);broken['raw']['labels']['own_status']='clear'
        with self.assertRaisesRegex(ValueError,'decode'):audit_prediction(broken,'evidence',.8)
        broken=deepcopy(p);broken['evidence_end']=7
        with self.assertRaisesRegex(ValueError,'pointer'):audit_prediction(broken,'evidence',.8)


if __name__=='__main__':unittest.main()
