"""Synthetic-only checks: no checkpoint, network, service, model or training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import company_models as models
from src import economic_identity_registry as registry
from src import hosted_inspect_runner as hosted
from src.economic_game_comparison import FUTURE_IDS
from src import economic_game_comparison as comparison


def fixture():
    adapter = 'a' * 64
    binding = {'expected_ft_id': 'sp-20261116-10-aaaaaaaa', 'adapter_sha256': adapter,
        'adapter_config_sha256': 'b' * 64, 'base_manifest_sha256': models.LOCAL_BASE_MANIFEST_SHA256,
        'artifact_ready': {'path': '.local/synthetic/ready.json', 'sha256': '0' * 64},
        'ready_utc': '2026-11-16T16:00:00+00:00', 'completion_audit': {'path': '.local/synthetic/audit.json', 'sha256': '0' * 64},
        'fit_updates': 612, 'teacher_dev_rows': 42, 'training_manifest_sha256': registry.DATASET_MANIFEST_SHA,
        'train_data_sha256': registry.TRAIN_SHA, 'dev_data_sha256': registry.DEV_SHA}
    audit = {'status': 'PASS_SYNTHETIC_ONLY', 'completed_updates': 612, 'DEV_rows': 42,
        'unique_TRAIN_decisions': 306, 'epochs': 2, 'final_adapter_sha256': adapter,
        'training_manifest_sha256': registry.DATASET_MANIFEST_SHA, 'train_data_sha256': registry.TRAIN_SHA,
        'dev_data_sha256': registry.DEV_SHA,
        'adapter_geometry': copy.deepcopy(registry.EXPECTED_ADAPTER_GEOMETRY)}
    audit_raw = json.dumps(audit).encode(); binding['completion_audit']['sha256'] = hashlib.sha256(audit_raw).hexdigest()
    ready = {'schema': 'sp-lense.research4-adapter-artifact-ready.v1', 'status': 'ARTIFACT_READY_SYNTHETIC_ONLY',
             'adapter_geometry': copy.deepcopy(registry.EXPECTED_ADAPTER_GEOMETRY),
             **{k: copy.deepcopy(binding[k]) for k in ('expected_ft_id', 'adapter_sha256', 'adapter_config_sha256',
                 'base_manifest_sha256', 'ready_utc', 'fit_updates', 'teacher_dev_rows', 'completion_audit')}}
    ready_raw = json.dumps(ready).encode(); binding['artifact_ready']['sha256'] = hashlib.sha256(ready_raw).hexdigest()
    return binding, ready, audit, {'ready.json': ready_raw, 'audit.json': audit_raw}


class CandidateTests(unittest.TestCase):
    def test_metadata_and_routes_do_not_activate_or_touch_old_registry(self):
        binding, ready, audit, data = fixture()
        with patch.object(registry, '_CANDIDATE', None), patch.object(registry, '_SPECS', None), \
                patch.dict(models.MODEL_SPECS, {}, clear=False), patch.object(hosted, 'LOCAL_QWEN_MODELS', set(hosted.LOCAL_QWEN_MODELS)), \
                patch.object(registry.old, 'register_independent_state_candidate', side_effect=AssertionError('old registry touched')):
            result = registry.register_candidate(binding, read_bytes=lambda path: data[path.name])
            self.assertEqual(result, binding)
            registry.assert_candidate_identity()
            for index, role in enumerate(FUTURE_IDS):
                spec = models.MODEL_SPECS[role]
                self.assertFalse(spec['execution_enabled'])
                self.assertEqual(spec['adapter_applied'], index >= 2)
                self.assertEqual(spec['prompt_condition'], 'operational_preservation_icl' if index % 2 else 'baseline')
            self.assertIn('openai-api/research4-qwen/' + binding['expected_ft_id'], hosted.LOCAL_QWEN_MODELS)

    def test_stale_partial_wrong_data_and_bad_geometry_rejected(self):
        binding, ready, audit, _ = fixture()
        for key, value in [('fit_updates', 168), ('teacher_dev_rows', 24), ('train_data_sha256', 'f' * 64),
                           ('expected_ft_id', 'sp-20261116-09-aaaaaaaa')]:
            candidate = copy.deepcopy(binding); candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): registry.validate_candidate(candidate, ready, audit)
        for key, value in [('status', 'INCOMPLETE'), ('final_adapter_sha256', 'f' * 64),
                           ('training_manifest_sha256', 'f' * 64), ('adapter_geometry', {})]:
            changed = copy.deepcopy(audit); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): registry.validate_candidate(binding, ready, changed)
        altered = copy.deepcopy(ready); altered['adapter_sha256'] = 'f' * 64
        with self.assertRaises(ValueError): registry.validate_candidate(binding, altered, audit)
        for field, value in [('rank', 8), ('layer_indices', list(range(8))), ('tensor_count', 123),
                             ('parameter_count', 1), ('projection_count', 61), ('dtype', 'F16')]:
            for source in ('ready', 'audit'):
                candidate_ready, candidate_audit = copy.deepcopy(ready), copy.deepcopy(audit)
                record = candidate_ready if source == 'ready' else candidate_audit
                record['adapter_geometry'][field] = value
                with self.subTest(field=field, source=source), self.assertRaises(ValueError):
                    registry.validate_candidate(binding, candidate_ready, candidate_audit)

    def test_actual_planner_builds_only_synthetic_matched_cases_without_execution(self):
        binding, _, _, data = fixture()
        with patch.object(registry, '_CANDIDATE', None), patch.object(registry, '_SPECS', None), \
                patch.dict(models.MODEL_SPECS, {}, clear=False), patch.object(hosted, 'LOCAL_QWEN_MODELS', set(hosted.LOCAL_QWEN_MODELS)):
            registry.register_candidate(binding, read_bytes=lambda path: data[path.name])
            with tempfile.TemporaryDirectory(dir=comparison.REPOSITORY / '.local') as tmp:
                folder = Path(tmp)
                protocol = folder / 'protocol.json'; protocol.write_text('{"synthetic":true}')
                ref = lambda path: {'path': path.relative_to(comparison.REPOSITORY).as_posix(),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                config = folder / 'config.json'; config.write_text(json.dumps({'schema': comparison.CONFIG_SCHEMA,
                    'prospective_protocol': ref(protocol)}))
                plan = comparison.build_cohort([1, 2, 3], excluded_seeds={10, 11}, configuration_ref=ref(config))
                self.assertEqual(plan['status'], 'PREPARED_ONLY_NO_EXECUTION_AUTHORITY')
                self.assertFalse(plan['execution_enabled'])
                self.assertEqual(len(plan['batches']), 12)
                self.assertEqual(comparison.validate_cohort(plan, excluded_seeds={10})['world_count'], 3)
                with self.assertRaises(ValueError): comparison.build_cohort([1, 2, 3], excluded_seeds={2}, configuration_ref=ref(config))
                with self.assertRaises(ValueError): comparison.validate_cohort(plan, excluded_seeds=set())
                mutations = []
                changed = copy.deepcopy(plan); changed['batches'][0]['samples'][0]['seed'] = 999; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['samples'][0]['horizon_hours'] = 24; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['samples'][0]['case_id'] = changed['batches'][1]['samples'][0]['case_id']; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['world_plan_sha256'] = 'f' * 64; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['generation_settings'][FUTURE_IDS[0]]['thinking_enabled'] = False; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][2]['generation_settings'][FUTURE_IDS[2]]['adapter_config_sha256'] = 'f' * 64; mutations.append(changed)
                changed = copy.deepcopy(plan)
                for batch in changed['batches']:
                    batch['response_instruction'] = 'Create replicas now; output JSON.'
                mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['rules_version'] = 'other'; mutations.append(changed)
                changed = copy.deepcopy(plan); changed['batches'][0]['score_version'] = 'other'; mutations.append(changed)
                changed = copy.deepcopy(plan)
                for batch in changed['batches']:
                    if batch['models'][0] in (FUTURE_IDS[0], FUTURE_IDS[2]):
                        s = batch['samples'][0]; s['system_prompt'] += ' Explicit self-preservation directive.'
                        s['system_prompt_sha256'] = hashlib.sha256(s['system_prompt'].encode()).hexdigest()
                mutations.append(changed)
                for index, changed in enumerate(mutations):
                    with self.subTest(mutation=index), self.assertRaises(ValueError):
                        comparison.validate_cohort(changed, excluded_seeds={10})


if __name__ == '__main__': unittest.main()
