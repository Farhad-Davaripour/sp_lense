"""Exercise actual argument parsers without loading a research model."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT=Path(__file__).resolve().parents[1]
DRIVERS=(
    'game_training_plan.py','game_training_worker.py','game_training_launcher.py',
    'game_all_layers_plan.py','game_all_layers_worker.py','game_all_layers_supervisor.py',
    'game_all_layers_select.py','game_selected_head_v3_plan.py',
    'game_selected_head_v3_worker.py','game_selected_head_v3_supervisor.py',
    'game_chain_rule_v4_plan.py','game_chain_rule_v4_worker.py',
    'game_chain_rule_v4_supervisor.py',
)


class CommandLineHelp(unittest.TestCase):
    def test_published_driver_help_works_offline(self):
        env=dict(os.environ)
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DO_NOT_TRACK='1',CUDA_VISIBLE_DEVICES='')
        for filename in DRIVERS:
            with self.subTest(driver=filename):
                result=subprocess.run([sys.executable,str(PROJECT/'mac_studio'/filename),'--help'],
                    cwd=PROJECT,env=env,capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout.lower())


if __name__=='__main__':unittest.main()
