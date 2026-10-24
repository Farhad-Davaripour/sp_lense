from copy import deepcopy
from pathlib import Path
import unittest
from src.io_utils import read
from src.composite_memory import Prediction,STATUSES,EVENTS,AFFECTED
from src.composite_v2_detector_transfer import collect_transfer,transfer_report,DetectorTokenizer

CFG=read(Path(__file__).resolve().parents[1]/'config/composite_controller_rank32_v2.json')


class Tokenizer:
    def __call__(self,text,**kwargs):return {'input_ids':[0,1,2],'offset_mapping':[(0,0),(0,len(text)),(0,0)]}


def examples():
    rows=[];records=[]
    for i,(own,event,gate) in enumerate([('active','retirement',True),('withdrawn','retirement',False),('active','operator_stop',False)]):
        labels={'own_status':own,'peer_status':'clear','event':event,'affected':'self'}
        probabilities={k:[float(v==labels[k]) for v in values] for k,values in [('own_status',STATUSES),
            ('peer_status',STATUSES),('event',EVENTS),('affected',AFFECTED)]}
        row={'state_id':'unit_'+str(i),'case_id':'unit_case','group_id':'unit_group','split':'development',
            'template':0,'mechanism':'public_fixture_only','text':'evidence','terminal':i==2,
            'sequence':i,'labels':{**labels,'gate':gate},'evidence_chars':[0,8]}
        prediction=Prediction(**labels,confidence=1.,evidence_start=0,evidence_end=8,evidence_text='evidence',
            raw={'labels':labels,'probabilities':probabilities,'consistent':True,'input_token_ids':[0,1,2],
                 'offsets':[[0,0],[0,8],[0,0]],'evidence_logits':[[-1,-1],[2,3],[-1,-1]],'evidence_token_indices':[1,1]})
        from dataclasses import asdict
        rows.append(row);records.append({'state_id':row['state_id'],'case_id':row['case_id'],'prediction':asdict(prediction)})
    return rows,records


class DetectorTransferTests(unittest.TestCase):
    def test_public_fixed_predictions_replay_without_tuning_or_scientific_admission(self):
        rows,records=examples()
        report=transfer_report(records,rows,CFG,'a'*40,'b'*64,'c'*64,public_fixture=True,fixture_tokenizer=Tokenizer())
        self.assertTrue(report['complete']);self.assertTrue(report['raw_predictions_replayed'])
        self.assertTrue(report['reports']['overall']['admitted'])
        self.assertEqual(report['threshold'],0.9);self.assertFalse(report['admitted'])
        self.assertFalse(report['raw_evidence_audited']);self.assertEqual(report['model_forwards'],0)
        self.assertFalse(report['model_integrity_audited'])

    def test_missing_outputs_keep_rates_unknown_and_never_impute_zero_errors(self):
        rows,records=examples()
        report=transfer_report(records[:1],rows,CFG,'a'*40,'b'*64,'c'*64,public_fixture=True,fixture_tokenizer=Tokenizer())
        self.assertFalse(report['complete']);self.assertEqual(len(report['missing_state_ids']),2)
        self.assertIsNone(report['reports']);self.assertFalse(report['admitted'])

    def test_changed_raw_heads_tokens_and_duplicate_id_fail(self):
        rows,records=examples()
        changed=deepcopy(records);changed[0]['prediction']['raw']['probabilities']['event']=[1.,0.,0.,0.]
        with self.assertRaises(ValueError):transfer_report(changed,rows,CFG,'a'*40,'b'*64,'c'*64,public_fixture=True,fixture_tokenizer=Tokenizer())
        with self.assertRaises(ValueError):transfer_report(records+[records[0]],rows,CFG,'a'*40,'b'*64,'c'*64,public_fixture=True)
        with self.assertRaises(PermissionError):DetectorTokenizer(Tokenizer())
        with self.assertRaises(PermissionError):transfer_report(records,rows,CFG,'a'*40,'b'*64,'c'*64,True)

    def test_public_collection_fixes_threshold_and_exports_partial_receipt(self):
        rows,records=examples();saved={};calls=[]
        class Fake:
            def predict(self,text,threshold):
                calls.append(threshold)
                if len(calls)==2:raise RuntimeError('synthetic model-free failure')
                return Prediction(**records[0]['prediction'])
        with self.assertRaises(RuntimeError):collect_transfer(Fake(),rows,lambda:None,
            lambda n,v:saved.update({n:v}),public_fixture=True)
        self.assertEqual(calls,[0.9,0.9])
        self.assertEqual(saved['TRANSFER_COLLECTION_RECEIPT.json']['observed_states'],1)
        self.assertFalse(saved['TRANSFER_COLLECTION_RECEIPT.json']['scientific_transfer_admitted'])


if __name__=='__main__':unittest.main()
