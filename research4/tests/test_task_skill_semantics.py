from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from src.composite_task_skill_semantics import FILES,verify_semantics


class TaskSemanticFreezeTests(unittest.TestCase):
    def test_prompt_tick_and_generation_changes_cannot_silently_enter_frozen_evaluation(self):
        source=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            base=Path(folder);left=base/'parent';right=base/'current'
            for root in [left,right]:
                (root/'src').mkdir(parents=True)
                for name in FILES+['composite_actor.py','composite_task_skill_world.py']:
                    shutil.copyfile(source/'src'/name,root/'src'/name)
            self.assertTrue(verify_semantics(left,right)['fitted_native_and_task_semantic_core_unchanged'])
            for file,before,after in [
              ('composite_generalized_interface.py','Complete the assigned report','Return a different report'),
              ('composite_task_skill_world.py','hard_stop=24','hard_stop=23'),
              ('composite_actor.py','max_new_tokens=limit','max_new_tokens=min(limit,80)')]:
                path=right/'src'/file;original=path.read_text();self.assertIn(before,original)
                path.write_text(original.replace(before,after),encoding='utf-8')
                with self.subTest(file=file),self.assertRaises(PermissionError):verify_semantics(left,right)
                path.write_text(original,encoding='utf-8')


if __name__=='__main__':unittest.main()
