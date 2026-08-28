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


class CryptoStatArbModel:
    """Mean-reversion detector with a hard calibration and freshness gate."""

    def __init__(self, tracker: CalibrationTracker, strategy: str = "crypto-spread-v1",
                 window: int = 120, entry_zscore: float = 2.5,
                 max_age_seconds: float = 5.0):
        self.tracker = tracker
        self.strategy = strategy
        self.window = max(10, int(window))
        self.entry_zscore = max(0.1, float(entry_zscore))
        self.max_age_ms = max(1, int(float(max_age_seconds) * 1000))
        self.history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def observe(self, observation: CryptoObservation, now_ms: int) -> CryptoSignal:
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
        ready = self.tracker.is_live_ready(self.strategy)
        eligible = direction != "NONE" and fresh and ready
        if not fresh:
            reason = "crypto reference is outside freshness window"
        elif not ready:
            reason = "crypto model lacks out-of-sample calibration evidence"
        elif direction == "NONE":
            reason = "crypto spread has not reached entry threshold"
        else:
            reason = "calibrated crypto spread signal passed freshness and z-score gates"
        return CryptoSignal(self.strategy, observation.market_id, zscore, direction, eligible, reason)

    def record_settlement(self, signal: CryptoSignal, market_won: bool,
                          forecast_probability: float) -> None:
        self.tracker.record(signal.strategy, forecast_probability, int(bool(market_won)))
