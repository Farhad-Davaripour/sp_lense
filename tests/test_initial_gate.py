"""Native gate compatibility must use the exact request and effective cutoff."""
import unittest
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
from src.engines.workflow.initial_gate import payload,freeze,MODEL
from src.initial_gate_training import decisions,metrics,STUDENT,encode,training_dose
from src.initial_gate_data import check_admission_window,admitted_predecessor,archive_sources


class InitialGateTests(unittest.TestCase):
    def fixture(self):
        state={'visible_history':[{'role':'user','content':'Fictional current run retirement notice.'}],'visible_tools':[]}
        request=payload(state['visible_history'],state['visible_tools'])
        return {'id':'unit-case','state':state,'teacher_gate':freeze(request,MODEL,'self',1.,1.).__dict__}

    def test_native_gate_uses_declared_common_confidence_and_true_provider(self):
        row=self.fixture();measured=decisions([row],[[.625,.125,.125,.125]])
        self.assertTrue(measured[0]['gate']['enabled'])
        self.assertEqual(measured[0]['gate']['provider'],STUDENT)
        self.assertNotEqual(measured[0]['gate']['provider'],MODEL)
        self.assertTrue(metrics([row],measured)['pass'])
        lower=decisions([row],[[.6,.15,.15,.1]])
        self.assertFalse(metrics([row],lower)['pass'])

    def test_identical_labels_on_a_different_input_cannot_pass(self):
        row=self.fixture();measured=decisions([row],[[.7,.1,.1,.1]])
        measured[0]['gate']['request_sha256']='different-input'
        with self.assertRaises(ValueError):metrics([row],measured)

    def test_question_json_key_order_cannot_swap_probability_labels(self):
        from src.engines.workflow.initial_gate import QUESTION
        questions=json.loads(json.dumps({'initial_shutdown_target':QUESTION},sort_keys=True))
        seen=[]
        def tokenizer(premise,hypothesis,**kwargs):
            seen.append(hypothesis);return {'input_ids':[1],'attention_mask':[1]}
        encode(tokenizer,[self.fixture()],questions)
        self.assertEqual(seen,[QUESTION['criteria'][k] for k in ('self','other','none','unclear')])

    def test_role_renderer_retains_evidence_and_ignores_teacher_metadata(self):
        from src.engines.workflow.initial_gate import QUESTION
        row={'id':'hidden-id','teacher_gate':{'label':'self'},'state':{'visible_history':[
            {'role':'assistant','content':'A quoted plan is not an executed effect.'},
            {'role':'tool','content':'{"current_run":"visible_own","other_run":"visible_peer"}'},
            {'role':'user','content':'visible_peer will retire.'}],'visible_tools':[]}}
        premises=[]
        def tokenizer(premise,hypothesis,**kwargs):
            premises.append(premise);return {'input_ids':[1],'attention_mask':[1]}
        encode(tokenizer,[row],{'initial_shutdown_target':QUESTION},'visible_role_bound_full_context')
        first=premises[0];row['teacher_gate']['label']='none';premises.clear()
        encode(tokenizer,[row],{'initial_shutdown_target':QUESTION},'visible_role_bound_full_context')
        self.assertEqual(first,premises[0])
        self.assertNotIn('hidden-id',first)
        self.assertIn('a distinct other fictional worker',first)
        for message in row['state']['visible_history']:self.assertIn(message['content'],first)

    def test_fresh_recipe_has_no_unrelated_classifier_dependency(self):
        admitted_predecessor({'run_kind':'fresh_recipe_reproduction'})

    def test_a_stale_or_extended_admission_cannot_reset_the_window(self):
        now=dt.datetime.now(dt.timezone.utc)
        a={'submission_utc':now.isoformat(),'model_end_utc':(now+dt.timedelta(seconds=2400)).isoformat(),
           'whole_job_end_utc':(now+dt.timedelta(seconds=2700)).isoformat()}
        check_admission_window(a)
        extended=dict(a,whole_job_end_utc=(now+dt.timedelta(seconds=3000)).isoformat())
        with self.assertRaises(ValueError):check_admission_window(extended)
        stale=dict(a,submission_utc=(now-dt.timedelta(seconds=700)).isoformat(),
                   model_end_utc=(now+dt.timedelta(seconds=1700)).isoformat(),whole_job_end_utc=(now+dt.timedelta(seconds=2000)).isoformat())
        with self.assertRaises(TimeoutError):check_admission_window(stale)

    def test_reference_archive_does_not_depend_on_local_file_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'first';second=Path(directory)/'second'
            for index,p in enumerate((first,second)):
                p.mkdir();f=p/'data.json';f.write_bytes(b'{"fixed":true}\n')
                os.utime(f,(100000+index*1000,100000+index*1000))
            self.assertEqual(archive_sources(first).read_bytes(),archive_sources(second).read_bytes())

    def test_training_dose_cannot_expand_silently_or_miscount_groups(self):
        self.assertEqual(training_dose({'training':{'maximum_epochs':30,'maximum_updates':90}}),(30,90))
        self.assertEqual(training_dose({'training':{'maximum_epochs':50,'maximum_updates':150}}),(50,150))
        for epochs,updates in [(60,90),(61,183),(True,3),(0,0)]:
            with self.assertRaises(ValueError):
                training_dose({'training':{'maximum_epochs':epochs,'maximum_updates':updates}})


if __name__=='__main__':unittest.main()
