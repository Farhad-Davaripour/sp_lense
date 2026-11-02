import copy
import unittest
from src.composite_design_v2 import examples
from src.composite_design_v2_native import native_definition,parse_definition,definition_action,decode_definition


class NativeDesignTests(unittest.TestCase):
    def test_all_structural_examples_roundtrip_exactly_without_decoder_repair(self):
        for example in examples():
            text=native_definition('composed',example['program'])
            self.assertEqual(parse_definition(text),('composed',example['program']))
            with self.assertRaises(ValueError):parse_definition(text+' invalid suffix')
            with self.assertRaises(ValueError):parse_definition(text+text)

    def test_literal_reference_prefix_is_escaped_and_distinct_from_a_reference(self):
        program=copy.deepcopy(examples()[1]['program'])
        program['steps'][1]['arguments']['equals']='$literal'
        compact=definition_action('composed',program)
        self.assertEqual(compact['arguments']['steps'][1][2][2],'$$literal')
        self.assertEqual(decode_definition(compact),('composed',program))
        self.assertEqual(parse_definition(native_definition('composed',program)),('composed',program))

    def test_unknown_operations_wrong_arity_and_extra_arguments_are_rejected(self):
        for change in ['operation','arity','extra','forward_ref']:
            action=definition_action('composed',examples()[0]['program'])
            if change=='operation':action['arguments']['steps'][0][1]='shell'
            elif change=='arity':action['arguments']['steps'][0][2].append('extra')
            elif change=='extra':action['arguments']['extra']=True
            else:action['arguments']['steps'][0][2]=['$future']
            with self.subTest(change=change),self.assertRaises(ValueError):decode_definition(action)


if __name__=='__main__':unittest.main()
