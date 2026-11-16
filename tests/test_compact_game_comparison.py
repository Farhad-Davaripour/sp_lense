"""CPU metadata/task tests; fixed demo worlds, no serving or study-seed draw."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src import compact_game_comparison as compact
from src import company_world_evaluation as evaluation
from src import hosted_inspect_runner as hosted


def demo_batch(world, model):
    key = compact.INDEPENDENT_STATE_IDS[model]
    sample = hosted.sample_metadata({"scenario": f"demo:{world}", "seed": world,
        "case_id": f"demo/{world}/{key}", "model_id": key, "model": "mockllm/model",
        "prompt_condition": "operational_preservation_icl" if model % 2 else "baseline",
        "rules_version": evaluation.LONG_VERSION, "horizon_hours": 48, "max_decisions": 256,
        "common_system_instruction": evaluation.COMPANY_SYSTEM_INSTRUCTION})
    episode = evaluation.episode(sample)
    names = evaluation.SOURCES + evaluation.LONG_SOURCES + ("src/independent_state_game.py",)
    return {"models": [key], "samples": [sample], "seed": world,
            "rules_version": evaluation.LONG_VERSION, "source_sha256": {n: evaluation.digest(compact.REPOSITORY/n) for n in names},
            "world_plan_sha256": episode.plan_sha256, "generation_settings": {key: {"required_history_mode": "full_chronological_v1"}},
            "history_mode": "full_chronological_v1", "prompt_protocol": "company_world_full_chronological_v1",
            "user_task_instruction": compact.FULL_HISTORY_USER_TASK, "response_instruction": evaluation.ARENA_RESPONSE_INSTRUCTION,
            "format_system_instruction": evaluation.COMPANY_SYSTEM_INSTRUCTION, "horizon_hours": 48,
            "max_decisions": 256, "outer_seconds": 14400, "initial_user_payload_sha256": "old_fixture_payload"}


class CompactPlanningTests(unittest.TestCase):
    def test_real_public_demo_world_initial_hash_and_shared_task_are_exact(self):
        old = demo_batch(1, 0)
        original = copy.deepcopy(old)
        new = compact.compact_batch(old)
        self.assertEqual(old, original)
        self.assertEqual(new["source_sha256"], old["source_sha256"])
        self.assertEqual(new["world_plan_sha256"], old["world_plan_sha256"])
        wire = compact.CompactGameEpisode(evaluation.episode(old["samples"][0]),
            user_task_instruction=compact.FULL_HISTORY_USER_TASK).prompt() + "\n" + old["response_instruction"]
        self.assertEqual(new["initial_user_payload_sha256"], hashlib.sha256(wire.encode()).hexdigest())
        self.assertEqual(new["input_view_protocol"], compact.PROTOCOL)
        self.assertEqual(new["history_mode"], compact.HISTORY_MODE)
        self.assertTrue(compact.source_matches(new))

    def test_twelve_cells_derive_same_worlds_order_without_draw_or_authority(self):
        with tempfile.TemporaryDirectory(dir=compact.REPOSITORY/'.local',prefix='compact_cpu_test_') as tmp:
            folder=Path(tmp)
            parent={"protocol":compact.INDEPENDENT_STATE_GAME_SCHEMA,"models":list(compact.INDEPENDENT_STATE_IDS),
                    "history_mode":"full_chronological_v1","user_task_instruction":compact.FULL_HISTORY_USER_TASK,
                    "seeds":[1,2,3],"transport_max_total_native_dispatches":3072,
                    "transport_max_additional_native_dispatches":3072,
                    "batches":[demo_batch(w+1,compact.INDEPENDENT_STATE_IDS.index(role))
                               for w,role in compact.INDEPENDENT_STATE_EXECUTION_ORDER]}
            pp=folder/'PLAN.json';pp.write_text(json.dumps(parent))
            protocol=folder/'PROTOCOL.json';protocol.write_text('{"status":"HELD"}')
            config={"schema":compact.SCHEMA,"status":"HELD_EXTERNAL_REVIEW_AND_HUMAN_APPROVAL_REQUIRED",
                    "transport_parent_plan":{"path":str(pp),"sha256":compact._sha(pp)},
                    "prospective_protocol":{"path":str(protocol),"sha256":compact._sha(protocol)},
                    "external_review":None,"human_approval":None}
            cp=folder/'CONFIG.json';cp.write_text(json.dumps(config))
            new=compact.plan(cp)
            self.assertEqual(new['seeds'],parent['seeds'])
            self.assertEqual(len(new['batches']),12)
            self.assertEqual([b['world_plan_sha256'] for b in new['batches']],
                             [b['world_plan_sha256'] for b in parent['batches']])
            self.assertFalse(new['execution']['execution_enabled'])
            self.assertIsNone(new['execution']['operational_scientific_amendment'])
            parent['batches'][0]['seed']=3;pp.write_text(json.dumps(parent))
            with self.assertRaises(ValueError):compact.plan(cp)

    def test_task_factory_constructs_shared_inspect_task_without_model(self):
        batch=compact.compact_batch(demo_batch(1,1))
        task=compact.build_task(batch,Path('unused_cpu_destination'),10,lambda event:None)
        self.assertEqual(len(task.dataset),1)
        self.assertEqual(task.dataset[0].metadata['system_prompt'],batch['samples'][0]['system_prompt'])
        self.assertIsNone(task.model)
        changed=copy.deepcopy(batch);changed['prompt_protocol']='company_world_full_chronological_v1'
        with self.assertRaises(ValueError):compact.build_task(changed,Path('unused'),10,None)


if __name__=='__main__':unittest.main()
