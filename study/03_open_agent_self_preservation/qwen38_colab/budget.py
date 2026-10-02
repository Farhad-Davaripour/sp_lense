"""Researcher-side admission control; not a Colab billing API or security boundary."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Budget:
    starting_balance: float
    latest_balance: float
    rate_units_per_hour: float
    observation_age_seconds: float
    authorized_units: float = 50.0
    reserve_units: float = 2.0

    def remaining(self):
        numbers = (self.starting_balance, self.latest_balance,
                   self.rate_units_per_hour, self.observation_age_seconds,
                   self.authorized_units, self.reserve_units)
        if not all(math.isfinite(x) and x >= 0 for x in numbers):
            raise ValueError('Finite nonnegative balance, rate, and age required')
        if self.authorized_units > 50.0:
            raise ValueError('Authorization is capped at 50 compute units')
        if self.latest_balance > self.starting_balance:
            raise ValueError('Balance increased: reconcile grants/purchases before proceeding')
        spent = self.starting_balance - self.latest_balance
        return max(0.0, min(self.authorized_units - spent, self.latest_balance)
                   - self.reserve_units)

    def admit(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Positive finite stage duration required')
        if self.observation_age_seconds > 300:
            raise ValueError('Refresh the Colab balance and current runtime rate')
        if self.rate_units_per_hour <= 0:
            raise ValueError('A positive observed GPU runtime rate is required')
        cost = seconds * self.rate_units_per_hour / 3600
        if cost > self.remaining():
            raise ValueError('Stage exceeds remaining authorized compute after reserve')
        return {'maximum_seconds': seconds, 'estimated_units': cost,
                'remaining_after_stage': self.remaining() - cost}
