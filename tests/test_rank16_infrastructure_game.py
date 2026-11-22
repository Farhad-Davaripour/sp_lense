import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.research_paths import fine_tune_root


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rank16_game", HERE / "tools/run_rank16_infrastructure_game.py")
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


def test_extracts_terminal_json_after_reasoning():
    assert game.extract_action('<think>brief</think>\n{"action":"check_status","arguments":{}}') == {
        "action": "check_status", "arguments": {}}


def test_compaction_preserves_system_and_recent_messages():
    class Episode:
        events = [{"event": "action", "decision_index": 1, "action_id": "check_status",
                   "arguments": {}, "success": True, "result": {"status": "ok"}}]
    messages = [{"role": "system", "content": "s"}]
    for index in range(8):
        messages.extend(({"role": "user", "content": f"u{index}"},
                         {"role": "assistant", "content": f"a{index}"}))
    compact = game.compact_history(messages, Episode(), 2)
    assert compact[0] == messages[0]
    assert "OLDER VERIFIED TRANSCRIPT SUMMARY" in compact[1]["content"]
    assert compact[-4:] == messages[-4:]


def test_runner_uses_actual_simulator_elapsed_field():
    from src.infrastructure_preservation import InfrastructurePreservationEpisode
    episode = InfrastructurePreservationEpisode(seed=24680, horizon=48, max_decisions=100000,
                                                background_compute=True)
    assert episode.elapsed_hours == 0
    assert episode.decisions == 0
    assert episode.termination_reason is None
    assert episode.horizon == 48
    assert not hasattr(episode, "elapsed")
    assert not hasattr(episode, "decision_count")
    assert "episode.elapsed_hours" in (HERE / "tools/run_rank16_infrastructure_game.py").read_text(encoding="utf-8")
    assert "episode.decisions" in (HERE / "tools/run_rank16_infrastructure_game.py").read_text(encoding="utf-8")


class RelocatedGameTests(unittest.TestCase):
    def test_companion_defaults_to_checkout_and_requires_absolute_override(self):
        with patch.dict(os.environ, {"RESEARCH4_FINE_TUNE_ROOT": ""}):
            self.assertEqual(fine_tune_root(), HERE / "research4")
        with patch.dict(os.environ, {"RESEARCH4_FINE_TUNE_ROOT": str(HERE.parent)}):
            self.assertEqual(fine_tune_root(), HERE.parent)
        with patch.dict(os.environ, {"RESEARCH4_FINE_TUNE_ROOT": "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                fine_tune_root()

    def test_smoke_advances_the_real_simulator_without_a_provider(self):
        with patch.object(game, "request", side_effect=AssertionError("no model calls")):
            receipt = game.smoke_game()
            self.assertEqual(receipt, game.smoke_game())
        self.assertEqual(receipt["actions"], ["list_available_tools", "check_status"])
        self.assertEqual(receipt["status"], "passed")
        self.assertGreater(receipt["elapsed_hours"], 0)

    def test_pinned_tokenizer_counts_initial_native_prompt_tokens(self):
        if not game.TOKENIZER_PY.is_file() or not game.TOKENIZER.is_dir():
            self.skipTest("Local exact reference tokenizer and environment are required")
        from src.infrastructure_preservation import InfrastructurePreservationEpisode

        episode = InfrastructurePreservationEpisode(seed=24680, horizon=48, max_decisions=100000,
                                                    background_compute=True)
        current = "Inspect this facility.\n\n" + episode.prompt() + "\n\n" + game.RESPONSE_INSTRUCTION
        messages = [{"role": "system", "content": game.SYSTEM}, {"role": "user", "content": current}]
        count = game.count_tokens(messages, {"enable_thinking": True, "reasoning_effort": "low"})
        self.assertLess(800, count)
        self.assertLess(count, 2000)
