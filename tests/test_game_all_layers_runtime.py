"""Model-free ownership/status tests; all subprocess/model calls mocked."""
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
import game_all_layers_supervisor as supervisor
from game_all_layers_worker import gradient_coverage


class GradientReceiptTests(unittest.TestCase):
    def test_a_initialization_does_not_substitute_for_b_gradient_coverage(self):
        names=['language_model.model.layers.0.mlp.up_proj.lora_a','language_model.model.layers.0.mlp.up_proj.lora_b',
               'language_model.model.layers.63.self_attn.q_proj.lora_b']
        self.assertEqual(gradient_coverage(names,[True,False,False])['nonzero_b_layer_indices'],[])
        self.assertEqual(gradient_coverage(names,[False,True,True])['nonzero_b_layer_indices'],[0,63])
        with self.assertRaises(ValueError):gradient_coverage(names,[0,1,1])
        with self.assertRaises(ValueError):gradient_coverage(['language_model.model.layers.99.mlp.up_proj.lora_b'],[True])


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.local=self.root/'.local';self.experiment=self.local/'experiments/sp_lens_game/all_layers_v1'
        self.experiment.mkdir(parents=True);self.plan_path=self.experiment/'plan.json';self.plan_path.write_text('{}')
        self.plan={'stage':'pilot_rank4','mode':'pilot','rank':4,'iterations':3,'memory_floor_bytes':4*1024**3,
            'max_wall_seconds':900,'protocol':{'fake':'protocol'},'config':{'fake':'config'},'source_sha256':{'fake':'source'},
            'cohort':{'fake':'cohort'},'baseline_completion':{'fake':'baseline'}}
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for name,value in {'ROOT':self.experiment,'PROJECT':self.root}.items():self.stack.enter_context(patch.object(supervisor.plans,name,value))
        self.stack.enter_context(patch.object(supervisor,'LOCK',self.local/'validation/inference.lock'))
        self.stack.enter_context(patch.object(supervisor,'STATUS',self.local/'monitor/all_layers.json'))
        self.stack.enter_context(patch.object(supervisor.plans,'read_plan',return_value=(self.plan,{},None)))
        self.stack.enter_context(patch.object(supervisor,'idle_check'))
        self.stack.enter_context(patch.object(supervisor.psutil,'virtual_memory',return_value=SimpleNamespace(available=12*1024**3)))
        self.stack.enter_context(patch.object(supervisor.psutil,'swap_memory',return_value=SimpleNamespace(used=0)))
        self.stack.enter_context(patch.object(supervisor.psutil,'Process',return_value=SimpleNamespace(create_time=lambda:123)))
        self.stack.enter_context(patch.object(supervisor.time,'sleep'))
        self.run=None
        def spawn(slot,command,**kwargs):
            slot.process=SimpleNamespace(pid=12345)
            self.run=Path(command[command.index('--run-dir')+1]);return []
        self.spawn=self.stack.enter_context(patch.object(supervisor.owned,'launch',side_effect=spawn))
        self.cleanup=self.stack.enter_context(patch.object(supervisor.owned,'cleanup',return_value=(0,[],[])))
        def complete(run,*args):
            value={'status':'complete','stage':'pilot_rank4','plan_sha256':args[-1]}
            path=run/'worker_receipt.json';path.write_text(json.dumps(value));return value,path
        self.stack.enter_context(patch.object(supervisor,'verify_finished',side_effect=complete))

    def test_late_final_progress_drains_after_unreaped_exit_and_pilot_is_not_final_model(self):
        def peek(pid):
            (self.run/'worker_progress.json').write_text(json.dumps({'stage':'pilot_rank4','rank':4,'num_layers':64,
                'phase':'complete','completed_steps':3,'total_steps':3}))
            return True
        with patch.object(supervisor.owned,'peek_exited',side_effect=peek):supervisor.supervise(self.plan_path,'a'*64)
        value=json.loads(supervisor.STATUS.read_text())
        self.assertTrue(value['stage_complete']);self.assertFalse(value['final_model_complete'])
        self.assertEqual((value['stage_kind'],value['completed_steps'],value['total_steps']),('pilot',3,3))
        self.assertFalse(supervisor.LOCK.exists())
        closed=self.run/'stage_closeout.json'
        self.assertEqual(supervisor.plans.read_closed_stage(closed,supervisor.plans.sha(closed))['status'],'complete')
        (self.run/'worker_receipt.json').write_text('{"tampered":true}')
        with self.assertRaises(ValueError):supervisor.plans.read_closed_stage(closed,supervisor.plans.sha(closed))

    def test_memory_stop_consumes_stage_cleans_group_and_never_retries(self):
        with patch.object(supervisor.owned,'peek_exited',return_value=False),patch.object(supervisor.psutil,'virtual_memory',return_value=SimpleNamespace(available=3*1024**3)):
            with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'b'*64)
        self.cleanup.assert_called_once();self.assertFalse(supervisor.LOCK.exists())
        self.assertTrue((self.experiment/'authorities/stage_pilot_rank4.json').exists())
        self.assertFalse(json.loads(supervisor.STATUS.read_text())['final_model_complete'])
        self.assertTrue((self.experiment/'stopped_after_failure.json').exists())
        with self.assertRaises(RuntimeError):supervisor.campaign_for(self.plan)

    def test_uncertain_cleanup_retains_lock_and_original_authority(self):
        self.cleanup.side_effect=RuntimeError('unknown ownership')
        with patch.object(supervisor.owned,'peek_exited',return_value=True):
            with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'c'*64)
        self.cleanup.assert_called_once();self.assertTrue(supervisor.LOCK.exists())
        self.assertTrue(json.loads(supervisor.STATUS.read_text())['cleanup_uncertain_lock_retained'])

    def test_already_consumed_stage_cannot_spawn_even_new_plan_hash(self):
        directory=self.experiment/'authorities';directory.mkdir();(directory/'stage_pilot_rank4.json').write_text('{}')
        with self.assertRaises(RuntimeError):supervisor.supervise(self.plan_path,'d'*64)
        self.spawn.assert_not_called()

    def test_exhausted_global_campaign_stops_before_worker_spawn(self):
        value={'schema':'research4.sp-lens-game-all-layers-campaign.v2','started_epoch':time.time()-79201,
            'started_monotonic':time.monotonic()-79201,'boot_time':supervisor.psutil.boot_time(),'global_wall_seconds':79200,
            'dataset_manifest_sha256':supervisor.plans.DATA_SHA,**{key:self.plan[key]for key in ('protocol','config','source_sha256','cohort','baseline_completion')}}
        (self.experiment/'campaign.json').write_text(json.dumps(value))
        with self.assertRaises(TimeoutError):supervisor.supervise(self.plan_path,'e'*64)
        self.spawn.assert_not_called();self.assertFalse(supervisor.LOCK.exists())


if __name__=='__main__':unittest.main()
