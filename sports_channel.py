#!/usr/bin/env python3
"""Sports stream ingestion and conservative latency-gate primitives.

The Sports stream reports game state, not an executable price. This module
therefore emits observations and only marks a latency opportunity when the
adapter supplies an explicit source timestamp. Missing timing provenance is a
hard rejection, not an inferred edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Awaitable, Callable, Dict, Optional


def _value(value, *names, default=None):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "live"}:
            return True
        if normalized in {"false", "0", "no", "ended", "closed"}:
            return False
    return default if value is None else bool(value)


def _epoch_ms(value) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return int(float(value.timestamp()) * 1000)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < numeric < 10_000_000_000:
        numeric *= 1000
    return int(numeric) if numeric > 0 else None


@dataclass(frozen=True)
class SportsObservation:
    game_id: str
    status: str
    live: bool
    ended: bool
    score: str
    period: str
    received_at_ms: int
    source_timestamp_ms: Optional[int]
    changed: bool


class SportsStateTracker:
    """Track score changes without inventing an event timestamp."""

    def __init__(self):
        self.last_state: Dict[str, tuple[str, str, str]] = {}

    def observe(self, event, received_at_ms: Optional[int] = None) -> SportsObservation:
        payload = _value(event, "payload", default=event)
        game_id = str(_value(payload, "game_id", "gameId", default=""))
        if not game_id:
            raise ValueError("sports event has no game id")
        status = str(_value(payload, "status", default=""))
        live = _as_bool(_value(payload, "live", default=False))
        ended = _as_bool(_value(payload, "ended", default=False))
        score = str(_value(payload, "score", default=""))
        period = str(_value(payload, "period", default=""))
        state = (status, score, period)
        changed = self.last_state.get(game_id) != state
        self.last_state[game_id] = state
        received = int(received_at_ms if received_at_ms is not None else time.time() * 1000)
        return SportsObservation(
            game_id=game_id, status=status, live=live, ended=ended, score=score,
            period=period, received_at_ms=received,
            source_timestamp_ms=_epoch_ms(_value(payload, "source_timestamp", "sourceTimestamp", "timestamp")),
            changed=changed,
        )


@dataclass(frozen=True)
class SportsLatencyDecision:
    eligible: bool
    delay_ms: int
    reason: str


class SportsLatencyGate:
    """Admit only a timestamp-proven, still-live sports delay."""

    def __init__(self, max_age_seconds: float = 5.0, min_delay_ms: int = 100,
                 max_delay_ms: int = 5_000):
        self.max_age_ms = max(1, int(float(max_age_seconds) * 1000))
        self.min_delay_ms = max(0, int(min_delay_ms))
        self.max_delay_ms = max(self.min_delay_ms, int(max_delay_ms))

    def evaluate(self, observation: SportsObservation, market_timestamp_ms: int,
                 now_ms: Optional[int] = None) -> SportsLatencyDecision:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        if not observation.changed:
            return SportsLatencyDecision(False, 0, "no new sports state")
        if not observation.live or observation.ended:
            return SportsLatencyDecision(False, 0, "game is not live")
        if observation.source_timestamp_ms is None or market_timestamp_ms <= 0:
            return SportsLatencyDecision(False, 0, "missing source or market timestamp")
        source_age = now - observation.source_timestamp_ms
        if source_age < 0 or source_age > self.max_age_ms:
            return SportsLatencyDecision(False, 0, "sports source event is outside freshness window")
        delay = observation.source_timestamp_ms - int(market_timestamp_ms)
        if delay < self.min_delay_ms or delay > self.max_delay_ms:
            return SportsLatencyDecision(False, delay, "market is not measurably behind the sports source")
        return SportsLatencyDecision(True, delay, "timestamp-proven sports latency gap")


async def consume_sports_channel(client, on_event: Callable[[object], Awaitable[None]]) -> None:
    """Consume the official SDK Sports stream until cancelled or disconnected."""
    try:
        from polymarket.streams import SportsSpec
    except ImportError as error:
        raise RuntimeError("Install polymarket-client for Sports Channel support") from error
    subscribe = getattr(client, "subscribe", None)
    if subscribe is None:
        raise RuntimeError("official client does not expose subscribe")
    stream = subscribe(SportsSpec())
    if hasattr(stream, "__await__"):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        raise RuntimeError("Sports subscription is not an async iterator")
    try:
        async for event in stream:
            await on_event(event)
    finally:
        close = getattr(stream, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result
