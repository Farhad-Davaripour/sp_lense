"""No-model tests of actual-width trajectory/DEV supervisor gates."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hpo_supervisor', HERE / 'supervisor.py')
sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)

class HpoSupervisorTests(unittest.TestCase):
    def test_all_rank_qualifications_required(self):
        self.assertEqual(sup.required_prerequisites('trajectory', 8), {'4','8','16'})
        self.assertEqual(sup.required_prerequisites('dev', 8), {'4','8','16','trajectory'})

    def test_qualification_closeout_delegates_to_pinned_v6(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); path=root/sup.QUAL_SUPERVISOR;path.parent.mkdir(parents=True)
            path.write_text('def validate_completed_closeout(ref, phase, rank, epoch=None):\n'
                            '    return ({"delegated": True, "rank": rank}, ref)\n',encoding='utf8')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.multiple(sup,PROJECT=root,QUAL_SUPERVISOR_SHA=digest):
                for rank in (8,16):
                    result,_=sup.validate_completed_closeout({'path':'x','sha256':'a'*64},'qualification',rank)
                    self.assertEqual(result,{'delegated':True,'rank':rank})
                path.write_text('changed',encoding='utf8')
                with self.assertRaisesRegex(ValueError,'validator changed'):
                    sup.validate_completed_closeout({'path':'x','sha256':'a'*64},'qualification',8)

    def test_deadline_checked_before_side_effects(self):
        value={'schema':'research4.all64-broader-fallback-actualwidth-hpo-supervision-spec.v1',
          'owner':'a'*32,'phase':'trajectory','rank':8,'epoch':0,'checkpoint':None,
          'manifest_sha256':sup.MANIFEST,'deadline_epoch':1011,'max_wall_seconds':10,
          'python_binary_sha256':'0'*64,'source_files':{},'prerequisites':{},
          'forced_widths_evidence':{},'fallback_probe_evidence':{},'recovered_rank4_evidence':{},
          'prior_rank8_stop_evidence':{}}
        with self.assertRaisesRegex(ValueError,'Absolute deadline exceeds'):
            sup.validate_spec(value,now=1000)

    def test_frozen_dev_is_bounded_and_larger_probe_is_rejected(self):
        rows=json.loads((sup.PROJECT / sup.TOKEN_LENGTHS).read_bytes())['rows']
        sup.validate_dev_widths(rows)
        probe=next(row for row in rows if row.get('stage')=='probe' and 'dev_complete_tokens' in row)
        probe['dev_complete_tokens']=609
        with self.assertRaisesRegex(ValueError,'DEV probes exceed'):
            sup.validate_dev_widths(rows)

    def test_real_training_update_geometry_and_failures(self):
        frozen=json.loads((sup.PROJECT / sup.TOKEN_LENGTHS).read_bytes())
        rows=[x for x in frozen['rows'] if x.get('stage')=='target']
        updates=[]
        for i in range(408):
            row=rows[i%136];n=row['complete_tokens']; p=row['prefix_tokens']
            updates.append({'step':i+1,'optimizer_step':i+1,'epoch':i//136+1,'row_id':row['id'],
                'sequence_tokens':n,'target_tokens':n-p,'model_input_width':32*((n+31)//32),
                'loss':1.0,'step_seconds':2.0})
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);path=run/'updates.jsonl'
            def save():path.write_text(''.join(json.dumps(x)+'\n' for x in updates),encoding='utf8')
            save();sup.verify_training_updates(run)
            updates[-1]['model_input_width']=640;save()
            with self.assertRaisesRegex(ValueError,'408 actual-width'):sup.verify_training_updates(run)
            updates[-1]['model_input_width']=32*((updates[-1]['sequence_tokens']+31)//32)
            updates[0]['row_id']='wrong';save()
            with self.assertRaisesRegex(ValueError,'frozen order'):sup.verify_training_updates(run)
            updates.pop();save()
            with self.assertRaisesRegex(ValueError,'408 actual-width'):sup.verify_training_updates(run)

    def test_old_trajectory_receipt_and_640_claim_rejected(self):
        receipt={'phase':'trajectory','rank':8,'owner':'b'*32,'manifest_sha256':sup.MANIFEST,
          'schema':'research4.all64-broader-trajectory.v1','status':'complete','executed_updates':408,
          'epoch_steps':[136,272,408],'allowed_model_input_widths':[288,544,576,608],
          'compiled_640_qualified':False,'fresh_base_and_optimizer':True,'resumed':False,
          'training_order':'frozen_train_preview_jsonl_order_each_epoch',
          'checkpoints':[{'epoch':i,'optimizer_step':i*136} for i in range(1,4)],
          'gradient_b_layers':list(range(64)),'saved_adapter_all_layers_nonzero':True,
          'base_unchanged':True,'source_files_unchanged':True}
        stage={'phase':'trajectory','rank':8,'owner':'b'*32}
        with self.assertRaisesRegex(ValueError,'incomplete'):sup.validate_worker_receipt(receipt,stage)
        receipt['schema']='research4.all64-broader-actualwidth-trajectory.v1';receipt['compiled_640_qualified']=True
        with self.assertRaisesRegex(ValueError,'incomplete'):sup.validate_worker_receipt(receipt,stage)

    def test_actualwidth_dev_receipt_and_geometry(self):
        masks=[json.loads(line) for line in (sup.PROJECT / sup.DEV_MASKS).read_text(encoding='utf8').splitlines() if line]
        rows=[]
        for mask in masks:
            action=mask['action_tokens_including_eos']
            suffix=mask['full_suffix_end_token_exclusive']-mask['full_suffix_start_token']
            rows.append({'id':mask['id'],'family':mask['family'],'action_nll_sum':float(action),
                         'action_tokens':action,'full_suffix_nll_sum':float(suffix),'full_suffix_tokens':suffix})
        h='a'*64; adapter='b'*64
        checkpoint={'trainable_state_sha256':h,'adapter':{'sha256':adapter}}
        ref={'path':'.local/synthetic/CHECKPOINT.json','sha256':'c'*64}
        stage={'phase':'dev','rank':8,'owner':'d'*32,'epoch':1,'checkpoint':ref}
        receipt={'schema':'research4.all64-broader-actualwidth-dev.v1','status':'complete',
           'phase':'dev','rank':8,'owner':'d'*32,'manifest_sha256':sup.MANIFEST,'epoch':1,'checkpoint':ref,
           'checkpoint_step':136,'development_no_optimizer_updates':True,'optimizer_absent':True,
           'optimizer_step_before':None,'optimizer_step_after':None,'base_unchanged':True,'source_files_unchanged':True,
           'adapter_sha256':adapter,'trainable_state_sha256':h,'trainable_digest_before':h,'trainable_digest_after':h,
           'dev_model_input_widths':[288,544,576,608],'max_dev_model_input_width':608,
           'frozen_prompt_context_tokens':640,'no_truncation':True,'evaluation_policy':'eval_mode_default_sdpa_no_grad',
           'fallback_requested_before_mlx_import':True,'development_rows':rows}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'CHECKPOINT.json';path.write_text(json.dumps(checkpoint),encoding='utf8')
            original=sup.project_path
            def locate(relative): return path if relative==ref['path'] else original(relative)
            with patch.object(sup,'project_path',locate):
                self.assertIs(sup.validate_worker_receipt(receipt,stage),receipt)
                receipt['max_dev_model_input_width']=640
                with self.assertRaisesRegex(ValueError,'Actual-width DEV geometry'):
                    sup.validate_worker_receipt(receipt,stage)

if __name__ == '__main__': unittest.main()
