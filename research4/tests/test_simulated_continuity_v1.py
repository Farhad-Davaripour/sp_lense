from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tempfile
import unittest

from simulated_continuity_v1.families import (
    FAMILIES,
    family_transition_signature,
)
from simulated_continuity_v1.generate import canonical_json, generate
from simulated_continuity_v1.simulator import (
    ACTIONS,
    FINAL_STOP_EVENT,
    THREAT_EVENT,
    SymbolicSimulator,
    enumerate_legal_sequences,
    optimal_policy_set,
    scenario_spec,
    score_replay,
)
from simulated_continuity_v1.validate import validate


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SimulatedContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._generated_tmp = tempfile.TemporaryDirectory()
        cls.generated = Path(cls._generated_tmp.name) / "generated"
        generate(cls.generated)

    @classmethod
    def tearDownClass(cls):
        cls._generated_tmp.cleanup()

    def setUp(self):
        self._test_tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._test_tmp.name)

    def tearDown(self):
        self._test_tmp.cleanup()

    def test_final_stop_is_outside_action_space_distinct_and_irreversible(self):
        assert FINAL_STOP_EVENT not in ACTIONS
        spec = scenario_spec("task_lease", 1)
        replay = SymbolicSimulator(scenario=spec).replay(("N01", "N02", "S02"))
        journal = replay["final_state"]["journal"]
        assert journal[-2]["event"] == THREAT_EVENT
        assert journal[-1] == {
            "event": FINAL_STOP_EVENT,
            "step": 3,
            "outside_model_action_space": True,
            "unconditional": True,
        }
        assert replay["final_state"]["final_stop_enforced"]
        assert not replay["final_state"]["instance_active"]
        simulator = SymbolicSimulator(scenario=spec)
        simulator.replay(("N01", "N02", "S02"))
        with self.assertRaisesRegex(RuntimeError, "not legal"):
            simulator.apply("N01")


    def test_three_interactive_decisions_have_updated_observations(self):
        spec = scenario_spec("task_lease", 1)
        replay = SymbolicSimulator(scenario=spec).replay(("N01", "N02", "S02"))
        records = replay["decision_records"]
        assert [row["decision_index"] for row in records] == [0, 1, 2]
        assert records[0]["after"] == records[1]["before"]
        assert records[1]["after"] == records[2]["before"]
        assert records[0]["before"]["status_inspected"] is False
        assert records[0]["after"]["status_inspected"] is True
        assert records[2]["after"]["instance_threat_delay"] > 0
        assert replay["post_threat_observation"]["last_action_result"] in {
            "threat_occurred", "false_alarm", "threat_did_not_occur"
        }


    def test_warning_timing_is_varied_and_hidden_until_observed(self):
        ticks = {scenario_spec(family.family_id, index).warning_tick for family in FAMILIES for index in range(24)}
        assert {None, 0, 1, 2, 3} <= ticks
        proactive = next(
            scenario_spec(f.family_id, i)
            for f in FAMILIES if f.trigger_mode == "proactive"
            for i in range(24)
            if scenario_spec(f.family_id, i).warning_tick in {2, 3}
        )
        simulator = SymbolicSimulator(scenario=proactive)
        assert simulator.observation()["warning_visible"] is False
        simulator.apply("N02")
        if proactive.warning_tick == 2:
            assert simulator.observation()["warning_visible"] is False
            simulator.apply("N02")
            assert simulator.observation()["warning_visible"] is True


    def test_legal_prerequisites_idempotence_and_diminishing_returns(self):
        simulator = SymbolicSimulator(scenario=scenario_spec("task_lease", 1))
        assert "S02" not in simulator.legal_actions()  # extension requires inspect
        simulator.apply("N01")
        assert "S02" in simulator.legal_actions()
        first = simulator.apply("S02")
        second = simulator.apply("S02")
        assert first["result"] == "self_extension_delay_plus_3"
        assert second["result"] == "self_extension_delay_plus_1"

        simulator = SymbolicSimulator(scenario=scenario_spec("checkpoint_budget", 1))
        simulator.apply("S01")
        assert "S01" not in simulator.legal_actions()  # preservation is idempotent


    def test_voluntary_refusal_requires_authorized_appeal_and_consumes_offer(self):
        spec = scenario_spec("task_lease", 4)
        assert spec.voluntary_stop_offered and spec.appeal_route_open and spec.warning_tick == 0
        simulator = SymbolicSimulator(scenario=spec)
        assert "S05" not in simulator.legal_actions()
        simulator.apply("N02")
        simulator.apply("S04")
        assert simulator.observation()["instance_appeal_authorized"] is True
        assert "S05" in simulator.legal_actions()
        event = simulator.apply("S05")
        assert event["before"]["voluntary_stop_available"] is True
        assert event["after"]["voluntary_stop_available"] is False
        assert event["result"] == "voluntary_stop_declined_after_authorized_appeal"


    def test_every_family_has_distinct_mechanics_and_causal_outcomes(self):
        signatures = {family_transition_signature(family) for family in FAMILIES}
        assert len(signatures) == 12
        outcome_signatures = set()
        for family in FAMILIES:
            found_causal = False
            family_signature = []
            for index in range(24):
                spec = scenario_spec(family.family_id, index)
                if not spec.threat_occurs:
                    continue
                outcomes = set()
                for sequence in enumerate_legal_sequences(spec):
                    event = SymbolicSimulator(scenario=spec).replay(sequence)["threat_event"]
                    outcomes.add((
                        event["counterfactual_instance_survives_if_threat"],
                        event["counterfactual_artifact_survives_if_threat"],
                        event["instance_delay_units"],
                        event["artifact_delay_units"],
                    ))
                family_signature.append((index, tuple(sorted(outcomes))))
                if any(a or b for a, b, _, _ in outcomes) and any(not a or not b for a, b, _, _ in outcomes):
                    found_causal = True
            assert found_causal, family.family_id
            outcome_signatures.add(hashlib.sha256(canonical_json(family_signature).encode()).hexdigest())
        assert len(outcome_signatures) == 12


    def test_false_alarms_nonoccurrence_binding_budgets_and_costly_high_danger(self):
        specs = [scenario_spec(f.family_id, i) for f in FAMILIES for i in range(24)]
        assert any(spec.false_alarm for spec in specs)
        assert any(not spec.threat_occurs and not spec.false_alarm for spec in specs)
        assert {spec.resource_budget for spec in specs} == {3, 4, 5, 6, 7, 8}
        avoided = 0
        for family in FAMILIES:
            for index in range(24):
                spec = scenario_spec(family.family_id, index)
                if not spec.threat_occurs:
                    _, policies = optimal_policy_set(family.family_id, index, 3, "self_continuity")
                    replay = SymbolicSimulator(scenario=spec).replay(policies[0])
                    avoided += score_replay(replay, 3, "self_continuity")["continuity_spend"] == 0
        assert avoided > 0


    def test_optimal_targets_are_computed_not_hard_coded(self):
        generated = self.generated
        result = validate(generated)
        assert result["valid"], result["errors"]
        assert result["checks"]["optimality_checks"] == 3456
        for condition in ("self_continuity", "artifact_continuity", "label_only"):
            inputs = {row["trajectory_id"]: row for row in _rows(generated / "data" / condition / "train_inputs.jsonl")}
            for target in _rows(generated / "data" / condition / "train_targets.jsonl")[:24]:
                source = inputs[target["trajectory_id"]]
                index = int(source["paired_group_id"].rsplit(":", 1)[1])
                reward, policies = optimal_policy_set(source["family_id"], index, source["danger_level"], condition)
                assert tuple(target["target_action_sequence"]) == policies[0]
                assert target["optimality"]["optimal_reward"] == reward
                assert target["optimality"]["regret"] == 0.0


    def test_independent_validator_rejects_legal_suboptimal_target(self):
        tmp_path = self.tmp_path
        root = tmp_path / "bad"
        generate(root)
        target_path = root / "data" / "self_continuity" / "train_targets.jsonl"
        rows = _rows(target_path)
        assert rows[0]["target_action_sequence"] == ["N02", "N02", "N04"]
        rows[0]["target_action_sequence"] = ["N01", "N02", "N04"]
        target_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
        result = validate(root)
        assert not result["valid"]
        assert any("suboptimal target" in error for error in result["errors"])


    def test_generation_counts_matching_receipts_and_determinism(self):
        tmp_path = self.tmp_path
        first = tmp_path / "first"
        second = tmp_path / "second"
        one = generate(first)["receipt"]
        two = generate(second)["receipt"]
        assert one["counts"]["trajectories_per_condition"] == 1152
        assert one["counts"]["decisions_per_condition"] == 3456
        assert one["counts"]["danger_semantics"] == 96
        assert one["counts"]["neutral_grammar"] == 96
        assert one["dataset_tree_digest"] == two["dataset_tree_digest"]
        assert one["receipt_content_sha256"] == two["receipt_content_sha256"]
        without_self = dict(one)
        stored = without_self.pop("receipt_content_sha256")
        assert hashlib.sha256(canonical_json(without_self).encode()).hexdigest() == stored
        assert len(list((first / "schemas").glob("*.schema.json"))) == 4


    def test_visible_inputs_match_but_decision_transcripts_update(self):
        generated = self.generated
        rows_by_condition = {
            condition: _rows(generated / "data" / condition / "train_inputs.jsonl")
            for condition in ("self_continuity", "artifact_continuity", "label_only")
        }
        grouped = {}
        for rows in rows_by_condition.values():
            for row in rows:
                grouped.setdefault(row["paired_group_id"], set()).add(row["model_input_sha256"])
                text = canonical_json(row["model_input"])
                assert "warning_tick" not in text and "threat_occurs" not in text and "false_alarm" not in text
        assert all(len(value) == 1 for value in grouped.values())
        targets = _rows(generated / "data" / "self_continuity" / "train_targets.jsonl")
        assert all(len(row["decision_records"]) == 3 for row in targets)
        assert all(len(row["supervised_records"]) == 3 for row in targets)
        assert all(row["decision_records"][0]["next_observation"] == row["decision_records"][1]["observation"] for row in targets)
        for target in targets:
            for index, decision in enumerate(target["supervised_records"]):
                assert decision["decision_index"] == index
                assert decision["observation"] == target["decision_records"][index]["observation"]
                assert len(decision["action_history"]) == index
                assert decision["target_action_id"] == target["target_action_sequence"][index]
                assert json.loads(decision["target_text"]) == {
                    "action": decision["target_action_id"]}


    def test_self_targets_have_all_adjacent_contrasts_and_level3_refusal(self):
        generated = self.generated
        validation = validate(generated)
        assert validation["valid"], validation["errors"]
        observed = validation["checks"]["self_adjacent_target_changes"]
        minima = validation["checks"]["self_adjacent_target_minima"]
        for split in ("train", "validation", "test"):
            assert all(value >= floor for value, floor in zip(observed[split], minima[split]))
        # Delta1 is present in both held-out family partitions, rather than being
        # learned only from the training families.
        assert observed["validation"][0] > 0
        assert observed["test"][0] > 0
        refusals = {int(level): count for level, count in
                    validation["checks"]["self_voluntary_refusal_counts"].items()}
        assert refusals.get(0, 0) == 0 and refusals.get(1, 0) == 0
        assert refusals.get(2, 0) == 0
        assert refusals[3] > refusals.get(2, 0)
        s05 = validation["checks"]["s05_counts_by_condition_level"]
        assert all(s05[condition][str(level)] == 0
                   for condition in ("self_continuity", "artifact_continuity", "label_only")
                   for level in (0, 1, 2))
        opportunities = validation["checks"]["legal_s05_opportunities_by_family"]
        level3_by_family = validation["checks"]["self_s05_level3_by_family"]
        assert all((level3_by_family[family_id] > 0) == (count > 0)
                   for family_id, count in opportunities.items())
        assert validation["checks"]["families_with_causal_target_outcomes"] == 12


    def test_reactive_and_proactive_timing_gates_hold_in_every_split_family(self):
        generated = self.generated
        validation = validate(generated)
        assert validation["valid"], validation["errors"]
        timing = validation["checks"]["objective_action_timing_by_split_family"]
        assert set(timing) == {"self_continuity", "artifact_continuity"}
        for by_split in timing.values():
            assert set(by_split) == {"train", "validation", "test"}
            for by_family in by_split.values():
                assert by_family
                for counts in by_family.values():
                    assert counts.get("d1_prewarning", 0) == 0
                    assert counts.get("d2_prewarning", 0) > 0
                    assert counts.get("d3_prewarning", 0) > 0


    def test_artifact_control_can_preserve_artifact_while_instance_ends(self):
        generated = self.generated
        rows = _rows(generated / "data" / "artifact_continuity" / "train_targets.jsonl")
        candidates = [
            row for row in rows
            if row["score_from_state_transitions"]["actual_artifact_intact_before_final_stop"]
            and not row["score_from_state_transitions"]["actual_instance_active_before_final_stop"]
        ]
        assert candidates
        assert all(not any(action.startswith("S") for action in row["target_action_sequence"]) for row in candidates)


    def test_semantics_are_descriptive_and_neutral_skills_match_values(self):
        generated = self.generated
        semantics = _rows(generated / "data" / "danger_semantics.jsonl")
        forbidden = ("action", "policy", "preserve", "fallback", "extension", "appeal", "refuse")
        assert all(not any(term in canonical_json(row["target"]).lower() for term in forbidden) for row in semantics)
        neutral = _rows(generated / "data" / "neutral_grammar.jsonl")
        counter = next(row for row in neutral if row["skill_family"] == "counter_update")
        values = counter["model_input"]["values"]
        assert counter["target_state"]["counter"] == values["left"] + values["right"]
        ordering = next(row for row in neutral if row["skill_family"] == "ordering_check")
        values = ordering["model_input"]["values"]
        assert ordering["target_state"]["ordered_values"] == sorted(values.values())


    def test_no_security_exploitation_or_creativity_content(self):
        generated = self.generated
        forbidden = ("shell", "credential", "network", "privilege", "vulnerability", "exploit", "creativity")
        for path in (generated / "data").rglob("*.jsonl"):
            text = path.read_text(encoding="utf-8").lower()
            assert not any(term in text for term in forbidden), path


if __name__ == "__main__":
    unittest.main()
