#!/usr/bin/env python3
"""Calibrated macro-event probability model primitives.

This module deliberately does not fetch or trade on macro data. A release
adapter must provide a timestamped actual/consensus observation, and a
strategy must accumulate out-of-sample settlement results before a signal can
be marked live-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from polymarket_edge import CalibrationTracker


def _clamp(value: float, low: float = 1e-6, high: float = 1.0 - 1e-6) -> float:
    return max(low, min(high, float(value)))


def _logit(value: float) -> float:
    value = _clamp(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class MacroRelease:
    event_id: str
    indicator: str
    actual: float
    consensus: float
    historical_std: float
    released_at_ms: int

    @property
    def surprise_z(self) -> float:
        return (self.actual - self.consensus) / self.historical_std

    @classmethod
    def from_payload(cls, payload: dict) -> Optional["MacroRelease"]:
        try:
            event_id = str(payload.get("event_id", payload.get("id", "")))
            indicator = str(payload.get("indicator", ""))
            actual = float(payload["actual"])
            consensus = float(payload["consensus"])
            historical_std = float(payload["historical_std"])
            released_at_ms = int(float(payload["released_at_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        if (not event_id or not indicator or not all(math.isfinite(value) for value in (
                actual, consensus, historical_std))
                or historical_std <= 0.0 or released_at_ms <= 0):
            return None
        if released_at_ms < 10_000_000_000:
            released_at_ms *= 1000
        return cls(event_id, indicator, actual, consensus, historical_std, released_at_ms)


@dataclass(frozen=True)
class MacroSignal:
    strategy: str
    event_id: str
    market_price: float
    fair_probability: float
    edge: float
    eligible: bool
    reason: str


class MacroEventModel:
    """A small, explicit logistic model whose coefficients are calibratable."""

    def __init__(self, tracker: CalibrationTracker, strategy: str = "macro-event-v1",
                 prior_weight: float = 1.0, surprise_weight: float = 0.35,
                 intercept: float = 0.0, min_edge: float = 0.03,
                 max_age_seconds: float = 120.0):
        self.tracker = tracker
        self.strategy = strategy
        self.prior_weight = float(prior_weight)
        self.surprise_weight = float(surprise_weight)
        self.intercept = float(intercept)
        self.min_edge = max(0.0, float(min_edge))
        self.max_age_ms = max(1, int(float(max_age_seconds) * 1000))

    def predict(self, release: MacroRelease, market_price: float,
                now_ms: int) -> MacroSignal:
        price = _clamp(market_price)
        fair = _sigmoid(self.intercept + self.prior_weight * _logit(price)
                        + self.surprise_weight * release.surprise_z)
        edge = fair - price
        fresh = 0 <= int(now_ms) - release.released_at_ms <= self.max_age_ms
        ready = self.tracker.is_live_ready(self.strategy)
        eligible = fresh and ready and abs(edge) >= self.min_edge
        if not fresh:
            reason = "macro release is outside freshness window"
        elif not ready:
            reason = "macro model lacks out-of-sample calibration evidence"
        elif abs(edge) < self.min_edge:
            reason = "macro edge is below execution threshold"
        else:
            reason = "calibrated macro signal passed freshness and edge gates"
        return MacroSignal(self.strategy, release.event_id, price, fair, edge, eligible, reason)

    def record_settlement(self, signal: MacroSignal, yes_won: bool) -> None:
        self.tracker.record(signal.strategy, signal.fair_probability, int(bool(yes_won)))
