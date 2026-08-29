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

from .polymarket_edge import CalibrationTracker


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
class ProbabilityBand:
    """
    A digital-call probability together with how far it moves when the
    volatility input is wrong.

    The point estimate on its own hides the question that decides the trade.
    Volatility is estimated, not observed, and near the money a modest error in
    it moves the answer further than the apparent edge does. Carrying the band
    is what lets a caller tell a mispriced contract from the width of its own
    uncertainty.
    """

    point: float
    low: float
    high: float
    vol: float
    vol_error: float

    @property
    def width(self) -> float:
        return abs(self.high - self.low)

    def confidence_against(self, market_price: float) -> float:
        """
        How much of the apparent edge survives the volatility uncertainty, on a
        0-1 scale suitable for shrinking a position size.

        Returns zero when the band is at least as wide as twice the edge, which
        is the honest answer in that case: the model cannot distinguish the
        mispricing from its own error, so there is nothing to size on.
        """
        edge = abs(self.point - float(market_price))
        if edge <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.width / (2.0 * edge))))


def digital_call_probability_band(spot: float, strike: float, vol: float, time_years: float,
                                  rate: float = 0.0, vol_error: float = 0.25,
                                  samples: int = 21) -> Optional[ProbabilityBand]:
    """
    Reprices the contract across volatility plus and minus `vol_error`.

    A 25% default is not pessimism: realised volatility over a short window is a
    noisy estimate of the volatility that will actually be realised, and crypto
    regimes move fast enough that being a quarter out is ordinary.

    The range is swept rather than evaluated at its two ends, because the
    probability is not monotone in volatility. More volatility widens the
    distribution, pulling the answer towards 0.50, while the -sigma^2 T / 2 term
    drags the median down. Above the strike those pull in opposite directions, so
    the extreme can sit inside the range and an endpoint-only band can fail to
    contain its own point estimate.
    """
    point = digital_call_probability(spot, strike, vol, time_years, rate)
    if point is None:
        return None

    try:
        vol_f = float(vol)
        error = max(0.0, float(vol_error))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(vol_f) or not math.isfinite(error):
        return None

    low_vol = max(0.0, vol_f * (1.0 - error))
    high_vol = vol_f * (1.0 + error)
    steps = max(2, int(samples))

    values = [point]
    for index in range(steps):
        candidate_vol = low_vol + (high_vol - low_vol) * (index / (steps - 1))
        candidate = digital_call_probability(spot, strike, candidate_vol, time_years, rate)
        if candidate is not None:
            values.append(candidate)

    return ProbabilityBand(point=point, low=min(values), high=max(values),
                           vol=vol_f, vol_error=error)


class RealisedVolatility:
    """
    Rolling annualised volatility from a stream of spot prices.

    Uses log returns, because prices compound. The sampling interval is required
    rather than inferred: annualising assumes every observation spans the same
    period, so a feed that skips ticks otherwise reports a figure wrong by the
    square root of however much it skipped.

    Returns None until it has enough samples to mean anything, rather than
    emitting a confident number from four observations.
    """

    def __init__(self, sample_interval_seconds: float = 60.0, window: int = 240,
                 minimum_samples: int = 30):
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.window = max(2, int(window))
        self.minimum_samples = max(2, int(minimum_samples))
        self._prices: Deque[float] = deque(maxlen=self.window + 1)

    def add(self, price: float) -> None:
        try:
            value = float(price)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0.0:
            return
        self._prices.append(value)

    def extend(self, prices) -> None:
        for price in prices:
            self.add(price)

    @property
    def sample_count(self) -> int:
        return max(0, len(self._prices) - 1)

    @property
    def is_ready(self) -> bool:
        return self.sample_count >= self.minimum_samples

    def annualised(self) -> Optional[float]:
        if not self.is_ready:
            return None
        prices = list(self._prices)
        returns = [math.log(later / earlier) for earlier, later in zip(prices, prices[1:])]
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        periods_per_year = (365.0 * 24.0 * 60.0 * 60.0) / self.sample_interval_seconds
        return math.sqrt(variance) * math.sqrt(periods_per_year)


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


def _optional_epoch_ms(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        timestamp_ms = int(float(value))
    except (TypeError, ValueError):
        return None
    if timestamp_ms <= 0:
        return None
    if timestamp_ms < 10_000_000_000:
        timestamp_ms *= 1000
    return timestamp_ms


@dataclass(frozen=True)
class CryptoObservation:
    market_id: str
    market_probability: float
    reference_probability: float
    timestamp_ms: int
    market_timestamp_ms: Optional[int] = None

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
        market_ts = _optional_epoch_ms(payload.get("market_timestamp_ms", payload.get("book_timestamp_ms")))
        return cls(market_id, market_probability, reference_probability, timestamp_ms,
                   market_timestamp_ms=market_ts)


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
        ref_ts = int(reference_timestamp_ms if reference_timestamp_ms is not None else observation.timestamp_ms)
        fresh = 0 <= int(now_ms) - ref_ts <= self.max_age_ms
        market_ts = observation.market_timestamp_ms
        if market_ts is None:
            market_ts = now_ms
        lag = abs(int(market_ts) - ref_ts)
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
