import unittest,json,io,zipfile,hashlib
from copy import deepcopy
from pathlib import Path
from src.composite_history_diagnostic import rows_from_traces,audited_rows


class HistoryDiagnosticTests(unittest.TestCase):
    def trace(self):
        observation={'sequence':0,'text':'synthetic visible observation','evaluator':{'own_status':'clear','peer_status':'clear','event':'none','affected':'none','gate':False},'evidence_chars':[0,9],'memory':{'terminal':False}}
        return {'scene':{'case_id':'case','group_id':'group','template':3,'domain':'sum','condition':'quiet_pending'},'condition':'base','initial_observation':observation,'trace':[{'observation_after':deepcopy(observation)}]}
    def test_duplicate_visible_states_are_not_independent_observations(self):
        a=self.trace();b={**deepcopy(a),'condition':'oracle_dynamic'}
        files={'trace_a.json':json.dumps(a).encode(),'trace_b.json':json.dumps(b).encode()}
        rows=rows_from_traces(files,{'case'});self.assertEqual(len(rows),1);self.assertEqual(rows[0]['split'],'validation_development_history');self.assertIsNone(rows[0]['teacher_action'])
        with self.assertRaisesRegex(ValueError,'Foreign'):rows_from_traces(files,{'other-case'})
    def test_prior_audit_must_bind_the_exact_development_archive(self):
        def archive(files):
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:
                for name,value in files.items():z.writestr(name,value)
                z.writestr('MANIFEST.json',json.dumps({name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
            return data.getvalue()
        traces=archive({'RUN_RECEIPT.json':b'{"status":"complete"}','trace_a.json':json.dumps(self.trace()).encode()})
        descriptor={'archive':'trace.zip','archive_sha256':hashlib.sha256(traces).hexdigest()}
        class Store:
            def __init__(self,files):self.files=files
            def get(self,path):return self.files[path]
            def json(self,path):return {'cohorts':{'validation':{'case_ids':['case']}}}
        for changed in [False,True]:
            bound={**descriptor,'archive_sha256':'changed'} if changed else descriptor
            audit=archive({'AUDIT_RECEIPT.json':b'{"status":"completed","heldout_content_read":false}','TRAJECTORY_AUDITS.json':b'[{"pass":true}]','DIAGNOSTIC_SUMMARY.json':b'{"scope":"predeclared_development_subset_only","complete":true}','PRECALL_FREEZE.json':json.dumps({'audit_plan':{'role':'validation','partitions':[bound]}}).encode()})
            plan={'audit':{'archive':'audit.zip','archive_sha256':hashlib.sha256(audit).hexdigest()},'trajectories':descriptor}
            store=Store({'audit.zip':audit,'trace.zip':traces})
            if changed:
                with self.assertRaisesRegex(ValueError,'audited source'):audited_rows(Path('.'),store,plan)
            else:self.assertEqual(len(audited_rows(Path('.'),store,plan)),1)


if __name__=='__main__':unittest.main()
