"""Mocked v4 lifecycle only; no real process, MLX import or Qwen execution."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_chain_rule_v4_supervisor as supervisor


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.project=Path(self.temp.name)
        self.root=self.project/'.local/experiments/sp_lens_game/all_layers_v1/v4_chain_rule';self.root.mkdir(parents=True)
        self.old_stop=self.root.parent/'stopped_after_failure.json';self.old_stop.write_text('{"preserved":true}')
        self.plan_path=self.root/'plan.json';self.plan_path.write_text('{}')
        self.plan={'max_wall_seconds':1800,'memory_floor_bytes':4*1024**3,
            'original_campaign_clock':{'boot_time':supervisor.psutil.boot_time(),'started_monotonic':time.monotonic(),
                'global_wall_seconds':79200},'prior_failures':[{'fake':'preserved'}]}
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(supervisor.time,'time',return_value=1794187739.176875))
        self.stack.enter_context(patch.object(supervisor.plans,'PROJECT',self.project));self.stack.enter_context(patch.object(supervisor.plans,'ROOT',self.root))
        self.stack.enter_context(patch.object(supervisor,'LOCK',self.project/'.local/validation/inference.lock'))
        self.stack.enter_context(patch.object(supervisor,'STATUS',self.project/'.local/monitor/v4.json'))
        self.stack.enter_context(patch.object(supervisor.plans,'read_plan',return_value=(self.plan,{},None)))
        self.stack.enter_context(patch.object(supervisor,'idle_check'))
        self.stack.enter_context(patch.object(supervisor.shutil,'disk_usage',return_value=SimpleNamespace(free=4*1024**3)))
        self.stack.enter_context(patch.object(supervisor.psutil,'boot_time',return_value=self.plan['original_campaign_clock']['boot_time']))
        self.stack.enter_context(patch.object(supervisor.psutil,'virtual_memory',return_value=SimpleNamespace(available=12*1024**3)))
        self.stack.enter_context(patch.object(supervisor.psutil,'swap_memory',return_value=SimpleNamespace(used=0)))
        self.stack.enter_context(patch.object(supervisor.psutil,'Process',return_value=SimpleNamespace(create_time=lambda:123,memory_info=lambda:SimpleNamespace(rss=1024))))
        self.stack.enter_context(patch.object(supervisor.time,'sleep'))
        self.stack.enter_context(patch.object(supervisor,'replay_proof',return_value={'synthetic':'passed'}))
        self.run=None
        def spawn(slot,command,**kwargs):
            slot.process=SimpleNamespace(pid=12345);self.run=Path(command[command.index('--run-dir')+1])
            (self.run/'phase.json').write_text(json.dumps({'phase':'before_state0_full','observed_epoch':time.time(),
                'mlx_peak_bytes':1234,'available_bytes':12*1024**3}))
            (self.run/'equivalence_receipt.json').write_text('{}');return []
        self.spawn=self.stack.enter_context(patch.object(supervisor.owned,'launch',side_effect=spawn))
        self.clean=self.stack.enter_context(patch.object(supervisor.owned,'cleanup',return_value=(0,[],[])))

    def test_completed_diagnostic_is_not_a_model_or_long_pilot_and_old_failure_stays(self):
        with patch.object(supervisor.owned,'peek_exited',return_value=True):value=supervisor.supervise(self.plan_path,'a'*64)
        self.assertEqual(value['original_deadline_epoch'],1794266339.176875)
        self.assertFalse(value['eligible_for_tuning_or_serving']);self.assertFalse(value['optimized_long_pilot_verified'])
        self.assertEqual(self.old_stop.read_text(),'{"preserved":true}')
        self.assertFalse(supervisor.LOCK.exists());self.assertTrue((self.run/'guard_memory.jsonl').exists())
        self.assertFalse(json.loads(supervisor.STATUS.read_text())['final_model_complete'])
        self.assertEqual(json.loads(supervisor.STATUS.read_text())['diagnostic_checks_completed'],3)

    def test_memory_stop_preserves_phase_guard_and_both_failure_lineages(self):
        with patch.object(supervisor.owned,'peek_exited',return_value=False),patch.object(supervisor.psutil,'virtual_memory',return_value=SimpleNamespace(available=3*1024**3)):
            with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'b'*64)
        self.clean.assert_called_once();self.assertFalse(supervisor.LOCK.exists())
        self.assertTrue((self.root/'stopped_after_failure.json').exists());self.assertTrue(self.old_stop.exists())
        samples=[json.loads(line)for line in(self.run/'guard_memory.jsonl').read_text().splitlines()]
        self.assertEqual(samples[-1]['operator_stop'],'available_memory_below_4_GiB')
        self.assertEqual(samples[-1]['phase'],'before_state0_full')

    def test_original_deadline_expiry_stops_before_any_spawn_or_new_clock(self):
        with patch.object(supervisor.time,'time',return_value=1794266340):
            with self.assertRaises(TimeoutError):supervisor.supervise(self.plan_path,'c'*64)
        self.spawn.assert_not_called();self.assertFalse((self.root/'campaign.json').exists())

    def test_consumed_v4_authority_cannot_be_replaced_with_a_new_hash(self):
        authority=self.root/'authorities';authority.mkdir();(authority/'chain_rule_equivalence_rank4.json').write_text('{}')
        with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'d'*64)
        self.spawn.assert_not_called();self.assertEqual((authority/'chain_rule_equivalence_rank4.json').read_text(),'{}')

    def test_uncertain_owned_cleanup_keeps_exclusivity_lock(self):
        self.clean.side_effect=RuntimeError('ownership unknown')
        with patch.object(supervisor.owned,'peek_exited',return_value=True):
            with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'e'*64)
        self.clean.assert_called_once();self.assertTrue(supervisor.LOCK.exists())
        self.assertTrue(json.loads(supervisor.STATUS.read_text())['cleanup_uncertain_lock_retained'])


if __name__=='__main__':unittest.main()
