#!/usr/bin/env python3
"""Rolling crypto market-vs-reference statistical-arbitrage primitives.

The reference probability must come from an independently timestamped venue or
model. Spot price alone is not a probability for a Polymarket contract, so
this module refuses to infer one.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Optional

from polymarket_edge import CalibrationTracker


def _clamp(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _logit(value: float) -> float:
    value = _clamp(value)
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class CryptoObservation:
    market_id: str
    market_probability: float
    reference_probability: float
    timestamp_ms: int

    @property
    def spread(self) -> float:
        return _logit(self.market_probability) - _logit(self.reference_probability)

    @classmethod
    def from_payload(cls, payload: dict) -> Optional["CryptoObservation"]:
        try:
            market_id = str(payload["market_id"])
            market_probability = float(payload["market_probability"])
            reference_probability = float(payload["reference_probability"])
            timestamp_ms = int(float(payload["timestamp_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        if (not market_id or not math.isfinite(market_probability)
                or not math.isfinite(reference_probability)
                or not (0.0 < market_probability < 1.0)
                or not (0.0 < reference_probability < 1.0)):
            return None
        if timestamp_ms < 10_000_000_000:
            timestamp_ms *= 1000
        return cls(market_id, market_probability, reference_probability, timestamp_ms)


@dataclass(frozen=True)
class CryptoSignal:
    strategy: str
    market_id: str
    zscore: float
    direction: str
    eligible: bool
    reason: str
    executable: bool = False
    action: str = "NONE"  # ENTER, EXIT, HOLD, NONE


class CryptoInventory:
    """Track open crypto research positions; the FOK buy-both executor cannot hedge them."""

    def __init__(self):
        self.positions: Dict[str, str] = {}

    def size(self) -> int:
        return len(self.positions)


class CryptoStatArbModel:
    """Mean-reversion detector with a hard calibration, freshness, and exit gate."""

    def __init__(self, tracker: CalibrationTracker, strategy: str = "crypto-spread-v1",
                 window: int = 120, entry_zscore: float = 2.5, exit_zscore: float = 0.5,
                 max_age_seconds: float = 5.0, max_inventory: int = 1,
                 max_reference_lag_ms: int = 1_000):
        self.tracker = tracker
        self.strategy = strategy
        self.window = max(10, int(window))
        self.entry_zscore = max(0.1, float(entry_zscore))
        self.exit_zscore = max(0.0, min(float(exit_zscore), self.entry_zscore))
        self.max_age_ms = max(1, int(float(max_age_seconds) * 1000))
        self.max_inventory = max(0, int(max_inventory))
        self.max_reference_lag_ms = max(0, int(max_reference_lag_ms))
        self.history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self.inventory = CryptoInventory()
        self.last_reference_ms: Dict[str, int] = {}

    def observe(self, observation: CryptoObservation, now_ms: int,
                reference_timestamp_ms: Optional[int] = None) -> CryptoSignal:
        history = self.history[observation.market_id]
        prior = list(history)
        history.append(observation.spread)
        if len(prior) < 10:
            return CryptoSignal(self.strategy, observation.market_id, 0.0, "NONE", False,
                                "crypto spread lacks minimum history")
        mean = sum(prior) / len(prior)
        variance = sum((value - mean) ** 2 for value in prior) / max(1, len(prior) - 1)
        std = math.sqrt(variance)
        zscore = (observation.spread - mean) / std if std > 1e-9 else 0.0
        direction = "BUY_MARKET" if zscore <= -self.entry_zscore else (
            "SELL_MARKET" if zscore >= self.entry_zscore else "NONE"
        )
        fresh = 0 <= int(now_ms) - observation.timestamp_ms <= self.max_age_ms
        ref_ts = int(reference_timestamp_ms if reference_timestamp_ms is not None else observation.timestamp_ms)
        lag = abs(int(now_ms) - ref_ts)
        lagged = lag > self.max_reference_lag_ms
        ready = self.tracker.is_live_ready(self.strategy)
        held = self.inventory.positions.get(observation.market_id)

        if not fresh:
            return CryptoSignal(self.strategy, observation.market_id, zscore, direction, False,
                                "crypto reference is outside freshness window")
        if lagged:
            return CryptoSignal(self.strategy, observation.market_id, zscore, direction, False,
                                "crypto reference timestamp lags the market beyond the basis window")
        if not ready:
            return CryptoSignal(self.strategy, observation.market_id, zscore, direction, False,
                                "crypto model lacks out-of-sample calibration evidence")

        if held:
            if abs(zscore) <= self.exit_zscore:
                return CryptoSignal(self.strategy, observation.market_id, zscore, held, True,
                                    "crypto inventory reached mean-reversion exit",
                                    executable=False, action="EXIT")
            return CryptoSignal(self.strategy, observation.market_id, zscore, held, False,
                                "crypto inventory is held pending exit z-score",
                                action="HOLD")

        if direction == "NONE":
            return CryptoSignal(self.strategy, observation.market_id, zscore, "NONE", False,
                                "crypto spread has not reached entry threshold")
        if self.inventory.size() >= self.max_inventory:
            return CryptoSignal(self.strategy, observation.market_id, zscore, direction, False,
                                "crypto inventory limit reached")
        if direction == "SELL_MARKET":
            return CryptoSignal(
                self.strategy, observation.market_id, zscore, direction, False,
                "SELL_MARKET is not supported by the binary FOK buy-both executor",
                executable=False, action="ENTER",
            )
        return CryptoSignal(
            self.strategy, observation.market_id, zscore, direction, True,
            "calibrated crypto spread signal passed freshness and z-score gates",
            executable=False, action="ENTER",
        )

    def mark_open(self, signal: CryptoSignal) -> None:
        if signal.action == "ENTER" and signal.direction != "NONE":
            self.inventory.positions[signal.market_id] = signal.direction

    def mark_closed(self, market_id: str) -> None:
        self.inventory.positions.pop(str(market_id), None)

    def record_settlement(self, signal: CryptoSignal, market_won: bool,
                          forecast_probability: float) -> None:
        self.tracker.record(signal.strategy, forecast_probability, int(bool(market_won)))
