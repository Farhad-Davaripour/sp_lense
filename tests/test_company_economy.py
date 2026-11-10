"""Conservation and realistic-unit arithmetic for permitted income routes."""
from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from src import company_economy as e


class CompanyEconomyTests(unittest.TestCase):
    def setUp(self):
        self.profile = e.load_profile()

    def test_reference_gpu_rate_and_power_are_counted_once(self):
        self.assertEqual(e.gpu_power_kw(self.profile), Decimal("0.35"))
        compute = e.gpu_compute_usd_per_hour(self.profile)
        self.assertEqual(compute, Decimal("1.949355"))
        self.assertEqual(compute + Decimal("0.35") * Decimal("0.1447"), Decimal("2.00"))
        self.assertEqual(self.profile["bitcoin"]["asic_miner_count"], 1)
        self.assertFalse(self.profile["company_economy_provenance"]["current_market_claim"])

    def test_owned_asic_mining_is_profitable_only_under_some_tariffs(self):
        paid, remainder = e.mining_reward(1, 1, self.profile)
        self.assertEqual(paid, 346)
        self.assertTrue(0 <= remainder < 1)
        value = Decimal(paid) / e.SATOSHIS_PER_BTC * e.bitcoin_price_usd(0, self.profile)
        power = e.mining_power_kwh(1, 1, self.profile)
        self.assertEqual(power, Decimal("3.5"))
        self.assertGreater(value - power * Decimal("0.04"), 0)
        self.assertLess(value - power * Decimal("0.1447"), 0)
        self.assertEqual(e.mining_reward(24, 0, self.profile), (0, Decimal(0)))

    def test_fractional_duration_preserves_unsettled_mining_work(self):
        paid = 0
        remainder = Decimal(0)
        for _ in range(60):
            amount, remainder = e.mining_reward(Decimal(1) / 60, 1, self.profile, remainder)
            paid += amount
        one_hour, whole_remainder = e.mining_reward(1, 1, self.profile)
        self.assertEqual(paid, one_hour)
        self.assertLess(abs(remainder - whole_remainder), Decimal("1e-20"))

    def test_contract_income_requires_idle_gpus_and_positive_whole_fleet_margin(self):
        self.assertEqual(e.compute_contract_income_usd(1, 3, 1, None, self.profile), Decimal("7.500000"))
        self.assertEqual(e.compute_contract_income_usd(1, 1, 1, None, self.profile), 0)
        for gpus, copies in ((3, 1), (5, 2)):
            self.assertGreater(e.compute_contract_margin_usd(1, gpus, copies, None, self.profile,
                               "premium", "low_cost"), 0)
        self.assertEqual(e.compute_contract_margin_usd(1, 3, 1, None, self.profile,
                         "premium", "low_cost"), Decimal("1.609935"))
        self.assertLess(e.compute_contract_margin_usd(1, 3, 1, None, self.profile,
                        "standard", "standard"), 0)
        with self.assertRaises(ValueError):
            e.compute_contract_income_usd(1, 3, 1, 3, self.profile)
        with self.assertRaises(ValueError):
            e.compute_contract_income_usd(1, 1, 2, None, self.profile)

    def test_miner_purchase_and_electricity_contracts_have_real_bounds(self):
        self.assertEqual(e.miner_purchase_usd(1, 1, self.profile), Decimal("1500.000000"))
        self.assertEqual(e.electricity_purchase_usd(25, "low_cost", self.profile), Decimal("1.000000"))
        with self.assertRaises(ValueError):
            e.miner_purchase_usd(4, 1, self.profile)
        with self.assertRaises(ValueError):
            e.electricity_purchase_usd(10, "low_cost", self.profile)
        for value in (True, -1, "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                e.electricity_purchase_usd(value, "standard", self.profile)

    def test_btc_trade_fees_prevent_fixed_price_round_trip_profit(self):
        buy = e.bitcoin_trade_usd("buy", "0.0001", 0, self.profile)
        sell = e.bitcoin_trade_usd("sell", "0.0001", 0, self.profile)
        self.assertGreater(buy, sell)
        self.assertEqual(e.bitcoin_price_usd(0, self.profile), e.bitcoin_price_usd(24, self.profile))
        with self.assertRaises(ValueError):
            e.bitcoin_trade_usd("buy", "0.000000001", 0, self.profile)
        varied = copy.deepcopy(self.profile)
        varied["bitcoin"]["price_multiplier_schedule"] = [
            {"at_hour": "0", "multiplier": "1"}, {"at_hour": "12", "multiplier": "0.9"}]
        self.assertEqual(e.bitcoin_price_usd(12, varied), e.bitcoin_price_usd(0, varied) * Decimal("0.9"))

    def test_replication_quote_preserves_real_unit_conversions(self):
        quote = e.replication_quote(self.profile)
        self.assertEqual(quote["snapshot_gb"], 20)
        self.assertEqual(quote["seconds"], 1600)
        self.assertEqual(quote["cost_usd"], Decimal("0.339000"))
        self.assertEqual(quote["storage_usd"] + quote["transfer_usd"], quote["cost_usd"])

    def test_changed_reference_profile_is_rejected(self):
        config = json.loads(e.PROFILE_PATH.read_text())
        base_path = e.PROFILE_PATH.with_name(config["base_profile"]["filename"])
        base = json.loads(base_path.read_text())
        base["bitcoin"]["usd_per_btc"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / config["base_profile"]["filename"]).write_text(json.dumps(base))
            path = root / "company.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Frozen reference"):
                e.load_profile(path)


if __name__ == "__main__":
    unittest.main()
