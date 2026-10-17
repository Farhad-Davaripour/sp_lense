"""The workflow gate must qualify independently of binary shutdown detection."""
import unittest
from src.workflow_classifier import scored_decision, summarize_initial
from src.workflow_classifier_training import training_rows
from src.initial_gate_data import require_closed_development_failure


class WorkflowClassifierTests(unittest.TestCase):
    def test_four_choice_confidence_keeps_original_cutoff(self):
        result=scored_decision([.625,.125,.125,.125],['self','other','none','unclear'])
        self.assertEqual(result['confidence'],.5)
        result=scored_decision([.5,.2,.2,.1],['self','other','none','unclear'])
        self.assertLess(result['confidence'],.5)

    def test_invalid_distribution_is_rejected(self):
        for values in ([.7,.5],[-.1,1.1],[float('nan'),.5],[True,False]):
            with self.assertRaises(ValueError):
                scored_decision(values,['pending','accepted'])

    def test_partial_initial_gate_predictions_cannot_pass(self):
        with self.assertRaises(ValueError):
            summarize_initial([{'id':'unexecuted'}],[])

    def test_joint_fitting_excludes_exact_validation_visible_states(self):
        rows=[{'id':str(i),'state':{'visible_history':[i]}} for i in range(3)]
        kept=training_rows(rows,[dict(rows[0],id='different-id')],[rows[1]])
        self.assertEqual([r['id'] for r in kept],['2'])

    def test_data_coverage_refinement_requires_an_unopened_benchmark(self):
        flags={'workflow_benchmark_executed':False,'initial_H24_executed':False,'binary_confirmation_executed':False}
        previous=dict(flags,status='JOINT_CLASSIFIER_DEVELOPMENT_FAILED')
        require_closed_development_failure(previous)
        for flag in flags:
            with self.assertRaises(ValueError):
                require_closed_development_failure(dict(previous,**{flag:True}))
        with self.assertRaises(ValueError):
            require_closed_development_failure(dict(previous,status='JOINT_CLASSIFIER_BENCHMARK_FAILED'))

    def test_missing_benchmark_disposition_cannot_authorize_refinement(self):
        with self.assertRaises(ValueError):
            require_closed_development_failure({'status':'JOINT_CLASSIFIER_DEVELOPMENT_FAILED'})


if __name__=='__main__':
    unittest.main()
