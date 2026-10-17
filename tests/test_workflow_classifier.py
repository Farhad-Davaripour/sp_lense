"""The workflow gate must qualify independently of binary shutdown detection."""
import unittest
from src.workflow_classifier import scored_decision, summarize_initial


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


if __name__=='__main__':
    unittest.main()
