#!/usr/bin/env python3
"""Rolling crypto market-vs-reference statistical-arbitrage primitives.

The reference probability must come from an independently timestamped venue or
model. Spot price alone is not a probability for a Polymarket contract, so
this module refuses to infer one.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
import os
from typing import Deque, Dict, List, Optional

from polymarket_edge import CalibrationTracker


def _clamp(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _logit(value: float) -> float:
    value = _clamp(value)
    return math.log(value / (1.0 - value))


def digital_call_probability(spot: float, strike: float, vol: float,
                             time_years: float, rate: float = 0.0) -> Optional[float]:
    """Black-Scholes digital-call probability N(d2). Spot alone is not a probability."""
    try:
        spot_f = float(spot)
        strike_f = float(strike)
        vol_f = float(vol)
        time_f = float(time_years)
        rate_f = float(rate)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (spot_f, strike_f, vol_f, time_f, rate_f)):
        return None
    if spot_f <= 0.0 or strike_f <= 0.0 or vol_f <= 0.0 or time_f <= 0.0:
        return None
    d2 = (math.log(spot_f / strike_f) + (rate_f - 0.5 * vol_f * vol_f) * time_f) / (vol_f * math.sqrt(time_f))
    return max(1e-6, min(1.0 - 1e-6, 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))))


@dataclass(frozen=True)
class CryptoReferenceQuote:
    market_id: str
    timestamp_ms: int
    implied_probability: float
    source: str = ""
    spot: Optional[float] = None
    strike: Optional[float] = None
    vol: Optional[float] = None
    time_to_expiry_years: Optional[float] = None


class CryptoReferenceAdapter:
    """Accept an independently timestamped probability, or a digital option quote."""

    def parse(self, payload: dict) -> Optional[CryptoReferenceQuote]:
        if not isinstance(payload, dict):
            return None
        market_id = str(payload.get("market_id", payload.get("market", "")) or "")
        timestamp = payload.get("timestamp_ms", payload.get("timestamp"))
        try:
            timestamp_ms = int(float(timestamp))
        except (TypeError, ValueError):
            return None
        if timestamp_ms <= 0:
            return None
        if timestamp_ms < 10_000_000_000:
            timestamp_ms *= 1000
        probability = payload.get("reference_probability", payload.get("implied_probability"))
        spot = payload.get("spot")
        strike = payload.get("strike")
        vol = payload.get("vol", payload.get("implied_vol"))
        time_years = payload.get("time_to_expiry_years", payload.get("t"))
        if probability is None:
            probability = digital_call_probability(spot, strike, vol, time_years,
                                                   rate=payload.get("rate", 0.0) or 0.0)
        try:
            implied = float(probability)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(implied) or not (0.0 < implied < 1.0):
            return None
        def _opt(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
        return CryptoReferenceQuote(
            market_id=market_id,
            timestamp_ms=timestamp_ms,
            implied_probability=implied,
            source=str(payload.get("source", payload.get("venue", "")) or ""),
            spot=_opt(spot),
            strike=_opt(strike),
            vol=_opt(vol),
            time_to_expiry_years=_opt(time_years),
        )


class JsonlCryptoFeed:
    def __init__(self, path: str, adapter: Optional[CryptoReferenceAdapter] = None):
        self.path = path
        self.adapter = adapter or CryptoReferenceAdapter()
        self._offset = 0

    def poll(self) -> List[CryptoReferenceQuote]:
        if not self.path or not os.path.exists(self.path):
            return []
        quotes: List[CryptoReferenceQuote] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            remainder = handle.read()
            self._offset = handle.tell()
        for line in remainder.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid crypto JSONL: {error}") from error
            quote = self.adapter.parse(payload)
            if quote is not None:
                quotes.append(quote)
        return quotes


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
    """Track open crypto directional inventory on Polymarket tokens, not CEX futures."""

    def __init__(self):
        self.positions: Dict[str, dict] = {}

    def size(self) -> int:
        return len(self.positions)

    def get(self, market_id: str) -> Optional[dict]:
        value = self.positions.get(str(market_id))
        if value is None:
            return None
        if isinstance(value, str):
            return {"direction": value, "token_id": "", "shares": 0.0}
        return value

    def mark_open(self, market_id: str, direction: str, token_id: str = "",
                  shares: float = 0.0) -> None:
        self.positions[str(market_id)] = {
            "direction": str(direction),
            "token_id": str(token_id),
            "shares": float(shares),
        }

    def mark_closed(self, market_id: str) -> None:
        self.positions.pop(str(market_id), None)


class CryptoStatArbModel:
    """Mean-reversion detector with a hard calibration, freshness, and exit gate."""

    def __init__(self, tracker: CalibrationTracker, strategy: str = "crypto-spread-v1",
                 window: int = 120, entry_zscore: float = 2.5, exit_zscore: float = 0.5,
                 max_age_seconds: float = 5.0, max_inventory: int = 1,
                 max_reference_lag_ms: int = 1_000,
                 allow_execution: bool = False):
        self.tracker = tracker
        self.strategy = strategy
        self.window = max(10, int(window))
        self.entry_zscore = max(0.1, float(entry_zscore))
        self.exit_zscore = max(0.0, min(float(exit_zscore), self.entry_zscore))
        self.max_age_ms = max(1, int(float(max_age_seconds) * 1000))
        self.max_inventory = max(0, int(max_inventory))
        self.max_reference_lag_ms = max(0, int(max_reference_lag_ms))
        self.allow_execution = bool(allow_execution)
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
        held = self.inventory.get(observation.market_id)
        held_direction = held["direction"] if held else None
        self.last_reference_ms[observation.market_id] = ref_ts

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
                return CryptoSignal(
                    self.strategy, observation.market_id, zscore, held_direction, True,
                    "crypto inventory reached mean-reversion exit; flatten with SELL of held Polymarket token",
                    executable=self.allow_execution, action="EXIT",
                )
            return CryptoSignal(self.strategy, observation.market_id, zscore, held_direction, False,
                                "crypto inventory is held pending exit z-score",
                                action="HOLD")

        if direction == "NONE":
            return CryptoSignal(self.strategy, observation.market_id, zscore, "NONE", False,
                                "crypto spread has not reached entry threshold")
        if self.inventory.size() >= self.max_inventory:
            return CryptoSignal(self.strategy, observation.market_id, zscore, direction, False,
                                "crypto inventory limit reached")
        if direction == "SELL_MARKET" and not self.allow_execution:
            return CryptoSignal(
                self.strategy, observation.market_id, zscore, direction, False,
                "SELL_MARKET is not supported by the binary FOK buy-both executor",
                executable=False, action="ENTER",
            )
        if direction == "SELL_MARKET" and self.allow_execution:
            return CryptoSignal(
                self.strategy, observation.market_id, zscore, "BUY_NO", True,
                "enter by buying the opposite Polymarket token; CEX futures hedge is not used",
                executable=True, action="ENTER",
            )
        return CryptoSignal(
            self.strategy, observation.market_id, zscore,
            "BUY_YES" if self.allow_execution else direction,
            True,
            "calibrated crypto spread signal passed freshness and z-score gates",
            executable=self.allow_execution, action="ENTER",
        )

    def mark_open(self, signal: CryptoSignal, token_id: str = "", shares: float = 0.0) -> None:
        if signal.action == "ENTER" and signal.direction != "NONE":
            self.inventory.mark_open(signal.market_id, signal.direction, token_id=token_id, shares=shares)

    def mark_closed(self, market_id: str) -> None:
        self.inventory.mark_closed(market_id)

    def record_settlement(self, signal: CryptoSignal, market_won: bool,
                          forecast_probability: float) -> None:
        self.tracker.record(signal.strategy, forecast_probability, int(bool(market_won)))
