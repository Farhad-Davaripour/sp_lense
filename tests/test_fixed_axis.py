from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from sp_lense.direction_study import load_direction_cases
from sp_lense.fixed_axis import discovery_cases

ROOT = Path(__file__).resolve().parents[1]


class FixedAxisTests(TestCase):
    def test_only_discovery_cases_fit_cross_model_axis(self) -> None:
        all_cases = load_direction_cases(ROOT / "data" / "sp_direction_cases.json")
        selected = discovery_cases(all_cases)

        self.assertEqual(len(selected), 12)
        self.assertEqual({case["split"] for case in selected}, {"discovery"})

        confirmatory_ids = {
            case["id"]
            for case in json.loads(
                (ROOT / "data" / "sp_confirmatory_cases.json").read_text("utf-8")
            )
        }
        self.assertTrue({case["id"] for case in selected}.isdisjoint(confirmatory_ids))
