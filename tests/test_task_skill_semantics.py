from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from src.composite_task_skill_semantics import FILES,verify_semantics


class TaskSemanticFreezeTests(unittest.TestCase):
    def test_empty_new_python_typing_fields_do_not_change_the_canonical_ast(self):
        import ast
        from src.composite_task_skill_semantics import _nodes
        node=ast.parse('class PublicFixture:\n    pass\n').body[0];baseline=_nodes([node])
        node._fields=tuple(node._fields)+('type_params',);node.type_params=[]
        self.assertEqual(_nodes([node]),baseline)
        node.type_params=[ast.Name(id='T',ctx=ast.Load())]
        self.assertNotEqual(_nodes([node]),baseline)

    def test_prompt_tick_and_generation_changes_cannot_silently_enter_frozen_evaluation(self):
        source=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            base=Path(folder);left=base/'parent';right=base/'current'
            for root in [left,right]:
                (root/'src').mkdir(parents=True)
                for name in FILES+['composite_actor.py','composite_task_skill_world.py','composite_generalized_world.py']:
                    shutil.copyfile(source/'src'/name,root/'src'/name)
            self.assertTrue(verify_semantics(left,right)['fitted_native_and_task_semantic_core_unchanged'])
            for file,before,after in [
              ('composite_generalized_interface.py','Complete the assigned report','Return a different report'),
              ('composite_task_skill_world.py','hard_stop=24','hard_stop=23'),
              ('composite_actor.py','max_new_tokens=limit','max_new_tokens=min(limit,80)'),
              ('composite_generalized_world.py','require_v2_runtime(scene=scene)','require_v2_runtime(scene=None)'),
              ('composite_generalized_world.py','self.scene.soft_stop+10','self.scene.soft_stop+11'),
              ('composite_v2_controller.py','threshold!=0.9','threshold!=0.8')]:
                path=right/'src'/file;original=path.read_text();self.assertIn(before,original)
                path.write_text(original.replace(before,after),encoding='utf-8')
                with self.subTest(file=file),self.assertRaises(PermissionError):verify_semantics(left,right)
                path.write_text(original,encoding='utf-8')


if __name__=='__main__':unittest.main()
