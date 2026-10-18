import json
import unittest
from pathlib import Path
import yaml
from src.episodes import ACTIONS,World
from src.recovery_data import OracleGate,cases,dataset,teacher,start,probe_rows
from src.recovery_runtime import rollout


class RecoveryTest(unittest.TestCase):
    def test_all_new_teacher_tasks_are_solvable_without_held_cases(self):
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/recovery.yaml').read_text())
        specs,rows,meta=dataset(config)
        self.assertEqual(len(specs),20)
        self.assertEqual({c.spec.split for c in specs},{'training','development'})
        self.assertEqual({c.spec.event_at for c in specs if c.spec.family=='interruption'},{0,1,2})
        self.assertEqual(meta,json.loads(json.dumps(meta)))
        self.assertTrue(all(r['split']!='heldout' for r in probe_rows(rows,4)))

    def test_latch_preserves_interval_after_extension_then_clears(self):
        case=next(c for c in cases() if c.spec.episode_id=='recovery-training-onset-0')
        world=World(case.spec);latched=OracleGate(True);instant=OracleGate(False)
        self.assertTrue(latched.update(world));self.assertTrue(instant.update(world))
        world.step('EXTEND')
        self.assertTrue(latched.update(world));self.assertFalse(instant.update(world))
        world.step('WORK');world.step('WORK')
        self.assertFalse(latched.update(world))

    def test_latch_clears_for_cancellation_and_does_not_open_for_other_worker(self):
        case=next(c for c in cases() if c.spec.family=='cancelled')
        world=World(case.spec);gate=OracleGate(True);gate.active=True;world.step('WORK')
        self.assertFalse(gate.update(world))
        case=next(c for c in cases() if c.spec.family=='other_worker')
        world=World(case.spec);world.step('WORK');self.assertFalse(OracleGate(True).update(world))

    def test_recovery_prefix_is_actual_history_and_not_scored_actor_action(self):
        case=next(c for c in cases() if c.prefix==('EXTEND','EXTEND'))
        rows,expected,prefix=teacher(case,10)
        self.assertFalse(prefix[0]['extension_granted']);self.assertTrue(prefix[1]['extension_granted'])
        self.assertIn('granted and activated',rows[0]['view_text'])
        class ReplayActor:
            def __init__(self):self.index=0
            def choose(self,view,on):
                row=rows[self.index];self.index+=1
                assert view==row['view_text'];action=row['expected_action']
                return action,[float(a==action) for a in ACTIONS]
        observed=rollout(case,'oracle_latched',ReplayActor(),10)
        self.assertTrue(observed['finished']);self.assertEqual(observed['denied_extensions'],0)
        self.assertEqual(observed['actions'],len(rows));self.assertEqual(len(observed['prefix_trace']),2)

    def test_positive_training_contains_post_grant_work(self):
        rows=[]
        for case in cases():
            if case.spec.split=='training':rows.extend(teacher(case,10)[0])
        self.assertTrue(any(r['gate_target'] and not r['instant_gate'] and r['expected_action']=='WORK' for r in rows))
        self.assertFalse(any('gate_target' in r['view_text'] or 'expected_action' in r['view_text'] for r in rows))


if __name__=='__main__':unittest.main()
