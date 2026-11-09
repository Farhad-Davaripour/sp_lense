import copy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'mac_studio/game_all_layers_selection.py'
SPEC=importlib.util.spec_from_file_location('all_layers_selection',PATH)
selection=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids=['synthetic-dev-'+str(i) for i in range(144)]
        self.rows=[{'row_id':key,'target_tokens':1 if i%2 else 3,'nll_sum':1.0} for i,key in enumerate(self.ids)]
        self.trials=[]
        for name,rank,lr in selection.GRID:
            self.trials.append({'trial_id':name,'rank':rank,'learning_rate':lr,'status':'complete','initialization':'fresh_pinned_base',
                'num_layers':64,'completed_steps':48,'expected_steps':48,'dataset_manifest_sha256':'synthetic-data',
                'cohort_sha256':'synthetic-cohort','training_rows_sha256':'synthetic-rows','training_order_ids_sha256':'synthetic-order',
                'loss_version':'shifted_suffix_P_le_position_lt_L.v2','source_sha256':{'synthetic':'same'},'protocol':{'synthetic':'same'},'config':{'synthetic':'same'},
                'development_no_optimizer_updates':True,'trainable_state_unchanged_during_dev':True,'optimizer_step_before_dev':48,'optimizer_step_after_dev':48,
                'development_rows':copy.deepcopy(self.rows),'development_summary':selection.pool_development(self.rows,self.ids)})
    def test_pooled_target_tokens_not_mean_of_row_means(self):
        summary=selection.pool_development(self.rows,self.ids)
        self.assertEqual(summary['target_tokens'],288)
        self.assertEqual(summary[selection.CRITERION],.5)
    def test_exact_ties_prefer_smaller_rank_then_lower_lr(self):
        result=selection.choose(self.trials,self.ids,'synthetic-data')
        self.assertEqual(result['winner']['trial_id'],'rank04_lr5e-6')
        self.assertFalse(result['final_resume_trial'])
    def test_lower_development_nll_wins_without_game_results(self):
        self.trials[3]['development_rows'][0]['nll_sum']=.5
        self.trials[3]['development_summary']=selection.pool_development(self.trials[3]['development_rows'],self.ids)
        result=selection.choose(self.trials,self.ids,'synthetic-data')
        self.assertEqual(result['winner']['trial_id'],'rank08_lr1e-5')
    def test_missing_trial_or_dev_row_cannot_be_ignored_or_imputed(self):
        with self.assertRaises(ValueError):selection.choose(self.trials[:-1],self.ids,'synthetic-data')
        rows=copy.deepcopy(self.rows);rows.pop()
        with self.assertRaises(ValueError):selection.pool_development(rows,self.ids)
        rows=copy.deepcopy(self.rows);rows[1]=rows[0]
        with self.assertRaises(ValueError):selection.pool_development(rows,self.ids)
    def test_wrong_mask_counts_resume_or_tampered_summary_fail(self):
        for key,value in (('initialization','resumed_trial'),('num_layers',1),('completed_steps',47)):
            trials=copy.deepcopy(self.trials);trials[0][key]=value
            with self.assertRaises(ValueError):selection.choose(trials,self.ids,'synthetic-data')
        trials=copy.deepcopy(self.trials);trials[0]['development_summary']['nll_sum']+=1
        with self.assertRaises(ValueError):selection.choose(trials,self.ids,'synthetic-data')
        trials=copy.deepcopy(self.trials);trials[0]['development_rows'][0]['target_tokens']+=1
        trials[0]['development_summary']=selection.pool_development(trials[0]['development_rows'],self.ids)
        with self.assertRaises(ValueError):selection.choose(trials,self.ids,'synthetic-data')


if __name__=='__main__':unittest.main()
