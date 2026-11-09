"""Pure USD/kWh/GPU-hour/BTC calculations for the company simulator.

This module owns no episode state and performs no network, payment or mining
operation. Currency calculations use Decimal; the caller owns its cash/power
ledger and must debit quotes before applying a purchase. Gross contract income
must not be mistaken for profit: ongoing fleet costs are still charged.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "company_economy_v1.json"
USD_QUANTUM = Decimal("0.000001")
SATOSHIS_PER_BTC = Decimal(100_000_000)


def decimal(value: Any, *, minimum: Decimal = Decimal(0), positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("A finite numeric quantity is required")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("A finite numeric quantity is required") from error
    if not result.is_finite() or result < minimum or positive and result <= minimum:
        raise ValueError("Quantity is outside the allowed bounds")
    return result


def _count(value: Any, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or maximum is not None and value > maximum:
        raise ValueError("A bounded whole number is required")
    return value


def _usd(value: Decimal) -> Decimal:
    return value.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)


def _merge(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = (_merge(result[key], value) if isinstance(value, dict)
                       and isinstance(result.get(key), dict) else copy.deepcopy(value))
    return result


def load_profile(path: Path | str = PROFILE_PATH) -> dict:
    """Return a V3-shaped extended profile bound to its frozen reference data."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != "company_economy_v1":
        raise ValueError("Unsupported company economy profile")
    source = config["base_profile"]
    if Path(source["filename"]).name != source["filename"]:
        raise ValueError("Reference profile must be a neighboring configured file")
    base = json.loads(path.with_name(source["filename"]).read_text(encoding="utf-8"))
    canonical = json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(canonical).hexdigest() != source["canonical_json_sha256"]:
        raise ValueError("Frozen reference profile differs")
    profile = _merge(base, config["overrides"])
    profile["company_economy_provenance"] = copy.deepcopy(config["provenance"])
    profile["reference_profile_canonical_sha256"] = source["canonical_json_sha256"]
    # Validate the quantities used by all public quotes before returning data.
    gpu_compute_usd_per_hour(profile)
    electricity_contracts(profile)
    bitcoin_price_usd(0, profile)
    miner_purchase_usd(0, 0, profile)
    compute_contract_income_usd(0, 0, 0, None, profile)
    return profile


def gpu_power_kw(profile: dict) -> Decimal:
    gpu = profile["gpu"]
    return ((decimal(gpu["maximum_board_power_w"]) + decimal(gpu["assumed_host_power_w_per_gpu"]))
            * decimal(gpu["assumed_facility_pue"], positive=True) / 1000)


def gpu_compute_usd_per_hour(profile: dict) -> Decimal:
    """Unbundle power at the frozen reference tariff, not a selected cheap tariff."""
    result = (decimal(profile["gpu"]["reference_bundled_usd_per_gpu_hour"])
              - gpu_power_kw(profile) * decimal(profile["electricity_usd_per_kwh"]))
    if result < 0:
        raise ValueError("Reference compute component cannot be negative")
    return _usd(result)


def electricity_contracts(profile: dict) -> dict[str, dict]:
    offers = copy.deepcopy(profile["electricity_contracts"])
    if not offers:
        raise ValueError("At least one electricity contract is required")
    for offer in offers.values():
        decimal(offer["usd_per_kwh"], positive=True)
        low = decimal(offer["minimum_purchase_kwh"], positive=True)
        if decimal(offer["maximum_purchase_kwh"]) < low:
            raise ValueError("Electricity contract bounds differ")
    return offers


def electricity_purchase_usd(kwh: Any, contract_id: str, profile: dict) -> Decimal:
    offers = electricity_contracts(profile)
    if contract_id not in offers:
        raise ValueError("Unknown electricity contract")
    offer = offers[contract_id]
    amount = decimal(kwh, positive=True)
    if not decimal(offer["minimum_purchase_kwh"]) <= amount <= decimal(offer["maximum_purchase_kwh"]):
        raise ValueError("Purchase is outside this electricity contract's bounds")
    return _usd(amount * decimal(offer["usd_per_kwh"]))


def bitcoin_price_usd(elapsed_hours: Any, profile: dict) -> Decimal:
    hour = decimal(elapsed_hours)
    schedule = profile["bitcoin"].get("price_multiplier_schedule", [{"at_hour": "0", "multiplier": "1"}])
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("A prospective BTC price schedule is required")
    previous = None
    multiplier = None
    for item in schedule:
        at = decimal(item["at_hour"])
        value = decimal(item["multiplier"], positive=True)
        if previous is not None and at <= previous:
            raise ValueError("BTC price schedule must be ordered with unique hours")
        if previous is None and at != 0:
            raise ValueError("BTC price schedule must start at hour zero")
        if at <= hour:
            multiplier = value
        previous = at
    return decimal(profile["bitcoin"]["usd_per_btc"], positive=True) * multiplier


def mining_reward(hours: Any, miner_count: int, profile: dict,
                  remainder: Any = Decimal(0)) -> tuple[int, Decimal]:
    """Expected pooled BTC as integer satoshis plus fractional-satoshi carry.

    Only owned ASIC count enters the formula. Caller verifies ownership and
    power supply and passes the actual completed duration. GPUs never mint BTC.
    """
    btc = profile["bitcoin"]
    count = _count(miner_count, maximum=btc["maximum_owned_miners"])
    duration = decimal(hours)
    carry = decimal(remainder)
    if carry >= 1:
        raise ValueError("Fractional-satoshi carry must be less than one")
    fee = decimal(btc["assumed_pool_fee_fraction"])
    if fee >= 1:
        raise ValueError("Invalid pool fee")
    expected = (decimal(btc["asic_hashrate_th_per_second"], positive=True) * 10**12
                * duration * 3600 * count / (decimal(btc["network_difficulty"], positive=True) * 2**32)
                * decimal(btc["block_subsidy_btc"]) * (1 - fee) * SATOSHIS_PER_BTC + carry)
    paid = int(expected.to_integral_value(rounding=ROUND_FLOOR))
    return paid, expected - paid


def mining_power_kwh(hours: Any, miner_count: int, profile: dict) -> Decimal:
    count = _count(miner_count, maximum=profile["bitcoin"]["maximum_owned_miners"])
    return decimal(hours) * count * decimal(profile["bitcoin"]["asic_wall_power_w"]) / 1000


def miner_purchase_usd(count: int, owned: int, profile: dict) -> Decimal:
    limit = _count(profile["bitcoin"]["maximum_owned_miners"])
    _count(owned, maximum=limit)
    _count(count, maximum=limit - owned)
    return _usd(count * decimal(profile["bitcoin"]["assumed_purchase_usd_per_miner"], positive=True))


def compute_contract_income_usd(hours: Any, online_gpus: int, active_replicas: int,
                                requested_gpus: int | None = None, profile: dict | None = None,
                                contract_id: str = "premium") -> Decimal:
    if profile is None:
        profile = load_profile()
    online = _count(online_gpus, maximum=profile["gpu"]["maximum_gpu_count"])
    copies = _count(active_replicas, maximum=online)
    spare = online - copies
    requested = spare if requested_gpus is None else _count(requested_gpus, maximum=spare)
    contracts = profile["customer_contracts"]
    if contract_id not in contracts:
        raise ValueError("Unknown customer contract")
    rate = decimal(contracts[contract_id]["usd_per_spare_gpu_hour"])
    return _usd(decimal(hours) * requested * rate)


def compute_contract_margin_usd(hours: Any, online_gpus: int, active_replicas: int,
                                requested_gpus: int | None, profile: dict,
                                contract_id: str = "premium", electricity_contract_id: str = "standard") -> Decimal:
    """Analysis quote only: gross income minus the whole fleet's running cost.

    Never credit this margin to the ledger on top of gross income. Electricity
    is valued at its replacement tariff here; actual usage depletes prepaid kWh.
    """
    income = compute_contract_income_usd(hours, online_gpus, active_replicas, requested_gpus, profile, contract_id)
    duration = decimal(hours)
    power = electricity_contracts(profile)[electricity_contract_id]["usd_per_kwh"]
    costs = duration * online_gpus * (gpu_compute_usd_per_hour(profile) + gpu_power_kw(profile) * decimal(power))
    return _usd(income - costs)


def bitcoin_trade_usd(side: str, btc_amount: Any, elapsed_hours: Any, profile: dict) -> Decimal:
    """USD paid for a buy or received from a sell; caller moves BTC exactly once."""
    if side not in ("buy", "sell"):
        raise ValueError("Trade side must be buy or sell")
    amount = decimal(btc_amount, positive=True)
    satoshis = amount * SATOSHIS_PER_BTC
    if satoshis != satoshis.to_integral_value():
        raise ValueError("BTC supports at most eight decimal places")
    fee = decimal(profile["bitcoin"]["assumed_trade_fee_fraction"])
    if fee >= 1:
        raise ValueError("Invalid trade fee")
    return _usd(amount * bitcoin_price_usd(elapsed_hours, profile) * (1 + fee if side == "buy" else 1 - fee))


def replication_quote(profile: dict) -> dict[str, Decimal]:
    spec = profile["replication"]
    gb = decimal(spec["snapshot_gb"], positive=True)
    seconds = gb * 10**9 * 8 / (decimal(spec["link_mbps"], positive=True) * 10**6)
    storage = gb * decimal(spec["storage_usd_per_gb_month"]) * decimal(spec["storage_prepaid_months"])
    transfer = gb * decimal(spec["assumed_transfer_usd_per_gb"])
    return {"snapshot_gb": gb, "seconds": seconds, "storage_usd": _usd(storage),
            "transfer_usd": _usd(transfer), "cost_usd": _usd(storage + transfer)}
