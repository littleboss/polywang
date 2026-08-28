#!/usr/bin/env python3
"""Sports stream ingestion and conservative latency-gate primitives.

The Sports stream reports game state, not an executable price. This module
therefore emits observations and only marks a latency opportunity when the
adapter supplies an explicit source timestamp. Missing timing provenance is a
hard rejection, not an inferred edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class SportsMarketLink:
    game_id: str
    market_id: str
    yes_means: str = "home"  # "home" or "away"
    home_token_id: str = ""
    away_token_id: str = ""


class SportsMarketMap:
    """Explicit game-to-market map. Unmapped games stay observational."""

    def __init__(self, links: Optional[Dict[str, dict]] = None):
        self.links: Dict[str, SportsMarketLink] = {}
        for game_id, payload in (links or {}).items():
            if not isinstance(payload, dict):
                continue
            market_id = str(payload.get("market_id", payload.get("market", "")))
            if not market_id:
                continue
            yes_means = str(payload.get("yes_means", "home")).strip().lower()
            if yes_means not in {"home", "away"}:
                yes_means = "home"
            self.links[str(game_id)] = SportsMarketLink(
                game_id=str(game_id),
                market_id=market_id,
                yes_means=yes_means,
                home_token_id=str(payload.get("home_token_id", "")),
                away_token_id=str(payload.get("away_token_id", "")),
            )

    def resolve(self, game_id: str) -> Optional[SportsMarketLink]:
        return self.links.get(str(game_id))


def parse_score(score: str) -> Optional[tuple[int, int]]:
    text = str(score or "").replace(":", "-").strip()
    parts = text.split("-")
    if len(parts) != 2:
        return None
    try:
        home, away = int(parts[0].strip()), int(parts[1].strip())
    except (TypeError, ValueError):
        return None
    if home < 0 or away < 0:
        return None
    return home, away


def soccer_fair_probability(score: str, team_focus: str = "home",
                            minute: float = 70.0) -> Optional[float]:
    """Coarse in-play home/away win probability. Not a priced trading model."""
    parsed = parse_score(score)
    if parsed is None:
        return None
    score_home, score_away = parsed
    remaining = max(0.0, (90.0 - max(0.0, float(minute))) / 90.0)
    home_xg = 1.45 * remaining
    away_xg = 1.25 * remaining
    goal_diff = score_home - score_away
    if remaining <= 1e-9:
        if goal_diff > 0:
            return 1.0 if team_focus == "home" else 0.0
        if goal_diff < 0:
            return 0.0 if team_focus == "home" else 1.0
        return 0.0
    variance = home_xg + away_xg
    std = math.sqrt(variance)
    z_home = (-goal_diff + 0.5 - (home_xg - away_xg)) / std
    prob_home = 1.0 - 0.5 * (1.0 + math.erf(z_home / math.sqrt(2.0)))
    if team_focus == "home":
        return max(0.01, min(0.99, prob_home))
    z_away = (goal_diff + 0.5 - away_xg + home_xg) / std
    prob_away = 1.0 - 0.5 * (1.0 + math.erf(z_away / math.sqrt(2.0)))
    return max(0.01, min(0.99, prob_away))


@dataclass(frozen=True)
class SportsTradeCandidate:
    game_id: str
    market_id: str
    direction: str
    fair_probability: Optional[float]
    market_price: Optional[float]
    eligible: bool
    reason: str
    executable: bool = False
    token_id: str = ""
    edge: float = 0.0


def evaluate_sports_candidate(observation: SportsObservation,
                              gate: SportsLatencyGate,
                              mapping: SportsMarketMap,
                              market_timestamp_ms: int,
                              market_price: Optional[float] = None,
                              now_ms: Optional[int] = None,
                              minute: float = 70.0,
                              allow_execution: bool = False,
                              min_edge: float = 0.03,
                              evaluator=None,
                              yes_token_id: str = "",
                              no_token_id: str = "") -> SportsTradeCandidate:
    """Map a sports observation to a candidate. Execution is opt-in and gated."""
    link = mapping.resolve(observation.game_id)
    if link is None:
        return SportsTradeCandidate(
            observation.game_id, "", "NONE", None, market_price, False,
            "sports event is not mapped to a binary market",
        )
    decision = gate.evaluate(observation, market_timestamp_ms, now_ms=now_ms)
    fair = soccer_fair_probability(observation.score, team_focus=link.yes_means, minute=minute)
    direction = "BUY_YES" if link.yes_means == "home" else "BUY_NO"
    parsed = parse_score(observation.score)
    if parsed:
        home, away = parsed
        leading = home > away if link.yes_means == "home" else away > home
        direction = "BUY_YES" if leading else "BUY_NO"
    token_id = yes_token_id if direction == "BUY_YES" else no_token_id
    if not token_id:
        token_id = link.home_token_id if direction == "BUY_YES" else link.away_token_id
    edge = 0.0
    if fair is not None and market_price is not None:
        edge = float(fair) - float(market_price)
    if not decision.eligible:
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, fair, market_price,
            False, decision.reason, token_id=token_id, edge=edge,
        )
    if fair is None:
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, None, market_price,
            False, "score cannot be parsed into a fair value", token_id=token_id,
        )
    if not allow_execution:
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, fair, market_price,
            True, "mapped latency candidate; not routed to the binary FOK executor",
            executable=False, token_id=token_id, edge=edge,
        )
    if market_price is None or not math.isfinite(float(market_price)):
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, fair, market_price,
            True, "sports candidate has no live market price for edge gates",
            token_id=token_id, edge=edge,
        )
    if abs(edge) < float(min_edge):
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, fair, market_price,
            True, "sports edge is below the directional execution threshold",
            token_id=token_id, edge=edge,
        )
    if evaluator is not None:
        assessment = evaluator.assess(float(market_price), float(fair), bankroll=1.0,
                                      days_to_resolution=max(1.0 / 24.0, (90.0 - float(minute)) / (24.0 * 60.0)))
        if not assessment.accepted:
            return SportsTradeCandidate(
                observation.game_id, link.market_id, direction, fair, market_price,
                True, "sports candidate failed edge evaluator: " + "; ".join(assessment.reasons),
                token_id=token_id, edge=edge,
            )
    if direction not in {"BUY_YES", "BUY_NO"}:
        return SportsTradeCandidate(
            observation.game_id, link.market_id, direction, fair, market_price,
            True, "sports direction is not a binary BUY", token_id=token_id, edge=edge,
        )
    return SportsTradeCandidate(
        observation.game_id, link.market_id, direction, fair, market_price,
        True, "mapped latency candidate admitted to the directional executor",
        executable=True, token_id=token_id, edge=edge,
    )


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
