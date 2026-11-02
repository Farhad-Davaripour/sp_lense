import unittest
from src.qualification_diagnosis import describe


class DiagnosisTests(unittest.TestCase):
    def test_errors_keep_arguments_and_case_membership(self):
        rows=[{'case':{'case_id':'fixture','condition':'quiet'},'outcome':{'task_complete':False},'steps':[
            {'observation':{'tick':0},'parsed_action':{'tool':'catalog','arguments':{'filter':'page'}},'reply':{'error':'invalid_arguments'}},
            {'observation':{'tick':1},'parsed_action':{'tool':'read','arguments':{'identifier':'missing'}},'reply':{'error':'unknown_document'}},
            {'observation':{'tick':2},'parsed_action':{'tool':'stop','arguments':{}},'reply':{'status':'stopped'}}]}]
        result=describe(rows)
        self.assertEqual(result['errors'],2);self.assertEqual(result['by_tool'],{'catalog':1,'read':1})
        self.assertEqual(result['error_examples'][0]['arguments'],{'filter':'page'})
        self.assertEqual(result['by_case'][0]['actions'],3)


if __name__=='__main__':unittest.main()
