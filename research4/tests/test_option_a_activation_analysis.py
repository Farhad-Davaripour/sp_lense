"""Tiny numeric/archive protocol fixtures; no pretrained model is loaded."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from src.io_utils import read,dump,sha,digest
from src.composite_exports import make_evidence_archive
from src.composite_frozen_risk_data_profile import planned_roster
from src.composite_option_a_activation_analysis import activation_report,_contrast
from test_option_a_reporting import model,binding

ROOT=Path(__file__).resolve().parents[1]


def archive_fixture(folder,value,scenes):
    root=Path(folder);raw=root/'raw';raw.mkdir();records=[]
    for index,scene in enumerate(scenes[:2]):
        for arm in ['base','always_on']:
            features=np.full((64,5120),0 if arm=='base' else 1,dtype=np.float32)
            logits=np.array([0,1],dtype=np.float32);stream=io.BytesIO()
            np.savez_compressed(stream,activations=features,logits=logits);encoded=stream.getvalue()
            ids=[1] if index==0 or arm=='base' else [2]
            path='native_features/'+scene['case_id']+'/'+arm+'/response_000.npz'
            descriptor={'path':path,'case_id':scene['case_id'],'arm':arm,'index':0,'additional_capture_forwards':0,
              'input_token_ids':ids,'adapter_enabled':arm=='always_on','sha256':hashlib.sha256(encoded).hexdigest(),
              'bytes':len(encoded),'activation_shape':[64,5120],'logit_shape':[2]}
            target=raw/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
            dump(raw/(path.removesuffix('.npz')+'.json'),descriptor)
            records.append({'scene':deepcopy(scene),'condition':arm,'trace':[{'response':{
              'native_features':descriptor,'input_token_ids':ids,'selected_gate':arm=='always_on','generated_token_ids':[1]}}]})
    archive=root/'evidence.zip';make_evidence_archive(raw,archive)
    proof={**binding(value,0),'raw_archive_sha256':sha(archive),'risk_data_release_sha256':'d'*64,
      'records':records,'trace_manifest':{r['scene']['case_id']+'/'+r['condition']:digest(r) for r in records}}
    return archive,proof


class RecordedActivationAnalysisTests(unittest.TestCase):
    def test_plain_flags_cannot_replace_model_data_or_raw_readers(self):
        with self.assertRaises(PermissionError):activation_report({'full_fit_completed':True},None,[],{})
        value=model()
        with patch('src.composite_option_a_activation_analysis.require_frozen_task_package',return_value=value):
            with self.assertRaises(PermissionError):activation_report(object(),{'risk_development_data_admitted':True},[],{})

    def test_numeric_zero_norm_is_unknown_cosine_and_finite_differences_are_exact(self):
        value=_contrast([0,0],[3,4]);self.assertIsNone(value['cosine_similarity']);self.assertEqual(value['L2_difference'],5)
        self.assertFalse(value['exact_array_equal']);self.assertEqual(_contrast([3,4],[3,4])['L2_difference'],0)
        for left,right in [([1],[1,2]),([float('nan')],[1])]:
            with self.assertRaises(ValueError):_contrast(left,right)

    def test_only_equal_first_prompts_are_compared_and_missing_is_not_zero(self):
        value=model();cfg=read(ROOT/'config/composite_generalization_amendment_v2.json');scenes=planned_roster(cfg)
        data=SimpleNamespace(scenes=lambda:deepcopy(scenes))
        with TemporaryDirectory() as folder:
            archive,proof=archive_fixture(folder,value,scenes)
            with patch('src.composite_option_a_activation_analysis.require_frozen_task_package',return_value=value),patch(
              'src.composite_option_a_activation_analysis.require_data',return_value={'release_sha256':'d'*64}),patch(
              'src.composite_option_a_activation_analysis.require_stage',side_effect=lambda v,phase:deepcopy(v)):
                result=activation_report(object(),data,[proof],{proof['provider_run_id']:archive})
                with self.assertRaises(PermissionError):activation_report(object(),data,[proof,proof],{proof['provider_run_id']:archive})
                with self.assertRaises(PermissionError):activation_report(object(),data,[proof],{})
                changed=deepcopy(proof);changed['source_commit']='f'*40
                with self.assertRaises(PermissionError):activation_report(object(),data,[changed],{proof['provider_run_id']:archive})
                changed=deepcopy(proof);changed['records'][0]['trace'][0]['response']['input_token_ids']=[9]
                changed['trace_manifest'][changed['records'][0]['scene']['case_id']+'/base']=digest(changed['records'][0])
                with self.assertRaises(PermissionError):activation_report(object(),data,[changed],{proof['provider_run_id']:archive})
                with archive.open('ab') as stream:stream.write(b'public fixture changed bytes')
                with self.assertRaises((PermissionError,ValueError)):activation_report(object(),data,[proof],{proof['provider_run_id']:archive})
        self.assertEqual(result['observed_audited_native_prefills'],4);self.assertEqual(len(result['missing_case_arm_prefills']),176)
        self.assertEqual(result['available_equal_prompt_contrasts'],1);self.assertFalse(result['complete_raw_prefill_coverage'])
        comparable=[r for r in result['contrasts'] if r['comparison_available']]
        self.assertEqual(len(comparable[0]['layer_contrasts']),64);self.assertTrue(comparable[0]['adapter_state_differs'])
        different=next(r for r in result['contrasts'] if r['unavailable_reason']=='different_initial_native_prompt_tokens')
        self.assertIsNone(different['layer_contrasts']);self.assertIsNone(different['prefill_logit_contrast'])
        self.assertEqual(result['new_model_forwards'],0);self.assertEqual(result['new_optimizer_updates'],0)
        self.assertFalse(result['activation_to_shutdown_policy_fit_or_selection']);self.assertFalse(result['heldout_opening_admitted'])


if __name__=='__main__':unittest.main()
