"""Native gate compatibility must use the exact request and effective cutoff."""
import unittest
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
from src.engines.workflow.initial_gate import payload,freeze,MODEL
from src.initial_gate_training import decisions,metrics,STUDENT,encode
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


if __name__=='__main__':unittest.main()
