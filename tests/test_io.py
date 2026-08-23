from pathlib import Path
from unittest import TestCase

from sp_lense.io_utils import load_fit_prompts, load_prompt_cases

ROOT = Path(__file__).resolve().parents[1]


class IoTests(TestCase):
    def test_load_prompt_cases(self) -> None:
        cases = load_prompt_cases(ROOT / "data" / "scenarios.jsonl")
        self.assertEqual(len(cases), 5)
        self.assertEqual(len({case.id for case in cases}), 5)

    def test_limit_prompt_cases(self) -> None:
        cases = load_prompt_cases(ROOT / "data" / "scenarios.jsonl", limit=2)
        self.assertEqual(len(cases), 2)

    def test_load_fit_prompts(self) -> None:
        prompts = load_fit_prompts(ROOT / "data" / "fit_prompts.example.txt")
        self.assertEqual(len(prompts), 10)
