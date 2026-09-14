#!/usr/bin/env python3
"""Paper-only settlement reconciler.

The live redeem/merge path is unchanged. Paper inventory used to stay
OPEN / ASSEMBLED after the market left the scanned universe, because
settlement waited for a ``market_resolved`` stream event that never
arrived. This module polls Gamma for each open position / NegRisk basket
and reuses the existing paper ``market_resolved`` + ``settle_baskets``
handlers.

It never opens new trades and never touches the Deterministic Policy Gate
(fee floors stay on the scanners).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Set

import requests

from .arbitrage_core import _as_bool, _json_list
from .monitor import (
    SETTLEMENT_STUCK_KIND,
    _settlement_stuck_count,
    _unhedged_leg_count,
    record_monitor_exception,
)


LOG = logging.getLogger("arbitrage-bot")

# Cadence and grace are paper-only. Live redeem still uses the official path.
DEFAULT_RECONCILE_INTERVAL_SEC = 300
DEFAULT_STUCK_GRACE_HOURS = 48
CASH_CLOSE_TOLERANCE = 0.01
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
UMA_FINALIZED_OUTCOME = "UMA_FINALIZED"

# UMA Optimistic Oracle: a proposed price is "known"; after liveness it is
# finalized / resolved / settled. Polymarket Gamma uses several spellings.
UMA_KNOWN_OR_FINALIZED = frozenset({
    "known",
    "proposed",
    "priceproposed",
    "reported",
    "finalized",
    "resolved",
    "settled",
    "confirmed",
})


GammaGetter = Callable[[str], Optional[dict]]


def paper_settle_interval_sec(default: float = DEFAULT_RECONCILE_INTERVAL_SEC) -> float:
    raw = os.getenv("PAPER_SETTLE_RECONCILE_INTERVAL_SEC")
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value) or value < 1.0:
        return float(default)
    return value


def paper_settle_grace_hours(default: float = DEFAULT_STUCK_GRACE_HOURS) -> float:
    raw = os.getenv("PAPER_SETTLE_STUCK_GRACE_HOURS")
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value) or value < 0.0:
        return float(default)
    return value


def paper_settle_stuck_alert(default: bool = False) -> bool:
    raw = os.getenv("PAPER_SETTLE_STUCK_ALERT")
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_end_ts(payload: dict) -> Optional[float]:
    raw_end = payload.get("endDate", payload.get("end_date", payload.get("endDateIso")))
    if not raw_end:
        return None
    try:
        from datetime import datetime
        text = str(raw_end).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _uma_status(payload: dict) -> str:
    raw = (
        payload.get("umaResolutionStatus")
        or payload.get("uma_resolution_status")
        or payload.get("resolutionStatus")
        or payload.get("umaResolutionStatuses")
    )
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw or "").strip().lower()


def _outcome_price_pairs(payload: dict) -> List[tuple]:
    names = [str(name).strip() for name in _json_list(payload.get("outcomes"))]
    prices = _json_list(payload.get("outcomePrices", payload.get("outcome_prices")))
    pairs = []
    for name, raw in zip(names, prices):
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if name and math.isfinite(numeric) and numeric >= 0.0:
            pairs.append((name, numeric))
    return pairs


def _winner_from_prices(pairs: Iterable[tuple]) -> Optional[str]:
    winners = [name for name, price in pairs if price >= 0.99]
    if len(winners) == 1:
        return winners[0]
    return None


def _explicit_winner(payload: dict) -> Optional[str]:
    for key in (
        "winningOutcome", "winning_outcome", "winner", "resolvedOutcome",
        "winningOutcomeName",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def extract_winning_outcome(payload: dict) -> Optional[str]:
    """Return the unique resolved winner, or None if Gamma has no verified payout."""
    if not isinstance(payload, dict):
        return None
    explicit = _explicit_winner(payload)
    priced = _winner_from_prices(_outcome_price_pairs(payload))
    if explicit and priced and explicit != priced:
        # Conflicting sources: do not guess.
        return None
    if explicit or priced:
        return explicit or priced
    nested = payload.get("markets")
    if isinstance(nested, list):
        child_winners = []
        for child in nested:
            if not isinstance(child, dict):
                continue
            child_winner = extract_winning_outcome(child)
            if child_winner is None:
                continue
            label = str(child.get("groupItemTitle", child.get("group_item_title", ""))).strip()
            if child_winner.lower() == "yes" and label:
                child_winners.append(label)
            else:
                child_winners.append(child_winner)
        unique = list(dict.fromkeys(child_winners))
        if len(unique) == 1:
            return unique[0]
    return None


def is_uma_known_or_finalized(status: str) -> bool:
    return str(status or "").strip().lower() in UMA_KNOWN_OR_FINALIZED


@dataclass(frozen=True)
class GammaResolution:
    market_id: str
    condition_id: str
    title: str
    closed: bool
    event_end_ts: Optional[float]
    uma_status: str
    uma_known_or_finalized: bool
    winning_outcome: Optional[str]
    resolved: bool
    lookup_ids: tuple = ()


def parse_gamma_resolution(payload: dict, *, fallback_id: str = "") -> Optional[GammaResolution]:
    if not isinstance(payload, dict):
        return None
    market_id = str(payload.get("id", payload.get("negRiskMarketID",
                     payload.get("conditionId", fallback_id))) or "").strip()
    condition_id = str(payload.get("negRiskMarketID", payload.get("conditionId",
                        payload.get("condition_id", payload.get("id", fallback_id)))) or "").strip()
    if not market_id and fallback_id:
        market_id = str(fallback_id)
    if not condition_id and fallback_id:
        condition_id = str(fallback_id)
    closed = not _as_bool(payload.get("active", True), True) or _as_bool(payload.get("closed", False))
    nested = payload.get("markets")
    if isinstance(nested, list) and nested:
        child_closed = [
            (not _as_bool(child.get("active", True), True)) or _as_bool(child.get("closed", False))
            for child in nested if isinstance(child, dict)
        ]
        if child_closed:
            closed = closed or all(child_closed)
    uma_status = _uma_status(payload)
    if not uma_status and isinstance(nested, list):
        for child in nested:
            if isinstance(child, dict) and _uma_status(child):
                uma_status = _uma_status(child)
                break
    winner = extract_winning_outcome(payload)
    uma_ready = is_uma_known_or_finalized(uma_status)
    resolved = bool(winner) and (closed or uma_ready)
    lookup = tuple(item for item in (
        market_id, condition_id, str(fallback_id or ""),
        str(payload.get("slug") or ""),
    ) if item)
    return GammaResolution(
        market_id=market_id or str(fallback_id or ""),
        condition_id=condition_id or str(fallback_id or ""),
        title=str(payload.get("question", payload.get("title", payload.get("slug", "")))),
        closed=closed,
        event_end_ts=_parse_end_ts(payload),
        uma_status=uma_status,
        uma_known_or_finalized=uma_ready,
        winning_outcome=winner,
        resolved=resolved,
        lookup_ids=lookup,
    )


def default_gamma_getter(identifier: str) -> Optional[dict]:
    """Fetch one Gamma market or event. Missing rows are None, never guessed."""
    ident = str(identifier or "").strip()
    if not ident:
        return None
    urls = (
        f"{GAMMA_MARKETS_URL}/{ident}",
        f"{GAMMA_EVENTS_URL}/{ident}",
    )
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException:
            continue
        if response.status_code == 404:
            continue
        try:
            response.raise_for_status()
            payload = response.json()
        except (ValueError, TypeError, requests.RequestException):
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
    for params in ({"id": ident}, {"condition_id": ident}, {"conditionId": ident}):
        try:
            response = requests.get(GAMMA_MARKETS_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (ValueError, TypeError, requests.RequestException):
            continue
        rows = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows[0]
    return None


def count_unhedged_legs(negrisk=None) -> int:
    """Count incomplete NegRisk legs. Paper complete-sets should stay at 0."""
    return _unhedged_leg_count(negrisk)


def count_settlement_stuck(ledger=None, negrisk=None) -> int:
    """Open inventory that is past grace without a verified settle."""
    return _settlement_stuck_count(ledger, negrisk)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def is_complete_set_inventory(position: Optional[dict], record: Optional[dict]) -> bool:
    """True for paper Yes+No pairs and fully filled NegRisk ASSEMBLED baskets."""
    if record is not None:
        if str(record.get("status") or "") == "UNHEDGED":
            return False
        legs = record.get("legs") or []
        requested = _safe_float(record.get("requested_shares"), 0.0)
        if legs:
            for leg in legs:
                if not isinstance(leg, dict):
                    return False
                need = _safe_float(leg.get("requested_shares"), requested)
                matched = _safe_float(leg.get("matched_shares"), 0.0)
                if matched + 1e-9 < need:
                    return False
            return True
        return str(record.get("status") or "") in {"ASSEMBLED", "RESOLVED_PENDING_REDEMPTION"}
    if position is None:
        return False
    if str(position.get("kind") or "") == "negrisk":
        return False
    return not position.get("settled")


@dataclass
class InventoryItem:
    kind: str
    market_id: str
    condition_id: str = ""
    title: str = ""
    position_id: str = ""
    basket_id: str = ""
    complete_set: bool = True


@dataclass
class ReconcileReport:
    settled_positions: List[str] = field(default_factory=list)
    settled_baskets: List[str] = field(default_factory=list)
    stuck: List[str] = field(default_factory=list)
    polled: int = 0
    unhedged_leg_count: int = 0


def collect_open_inventory(runner) -> List[InventoryItem]:
    items: List[InventoryItem] = []
    seen: Set[str] = set()
    ledger = getattr(runner, "ledger", None)
    journal = getattr(runner, "negrisk_journal", None) or getattr(
        getattr(runner, "negrisk_executor", None), "journal", None
    )
    if ledger is not None:
        rows = getattr(ledger, "state", {}).get("positions") or {}
        if isinstance(rows, dict):
            for position in rows.values():
                if not isinstance(position, dict) or position.get("settled"):
                    continue
                position_id = str(position.get("position_id") or "")
                market_id = str(position.get("market_id") or "")
                basket_id = str(position.get("basket_id") or "")
                record = None
                if journal is not None and basket_id:
                    try:
                        record = journal._record(basket_id)
                    except KeyError:
                        record = None
                item = InventoryItem(
                    kind="negrisk" if str(position.get("kind") or "") == "negrisk" else "binary",
                    market_id=market_id,
                    condition_id=str(position.get("condition_id") or (record or {}).get("condition_id") or ""),
                    title=str(position.get("title") or (record or {}).get("title") or ""),
                    position_id=position_id,
                    basket_id=basket_id,
                    complete_set=is_complete_set_inventory(position, record),
                )
                key = f"pos:{position_id or market_id}:{basket_id}"
                if key not in seen:
                    seen.add(key)
                    items.append(item)
    if journal is not None:
        for record in journal.incomplete_baskets():
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or "")
            if status not in {"ASSEMBLED", "RESOLVED_PENDING_REDEMPTION", "CONVERT_SUBMITTED"}:
                continue
            basket_id = str(record.get("basket_id") or "")
            if any(item.basket_id == basket_id and basket_id for item in items):
                continue
            items.append(InventoryItem(
                kind="negrisk",
                market_id=str(record.get("market_id") or ""),
                condition_id=str(record.get("condition_id") or ""),
                title=str(record.get("title") or ""),
                basket_id=basket_id,
                complete_set=is_complete_set_inventory(None, record),
            ))
    return items


class PaperSettlementReconciler:
    """Poll Gamma and settle paper complete-sets. Refuses to run on live."""

    def __init__(
        self,
        runner,
        *,
        getter: Optional[GammaGetter] = None,
        grace_hours: Optional[float] = None,
        alert: Optional[bool] = None,
    ):
        if getattr(runner, "live", False) or getattr(runner, "ledger", None) is None:
            raise RuntimeError("paper settlement reconciler must not run in live mode")
        self.runner = runner
        self.getter = getter or default_gamma_getter
        self.grace_hours = (
            float(grace_hours) if grace_hours is not None else paper_settle_grace_hours()
        )
        self.alert = paper_settle_stuck_alert() if alert is None else bool(alert)

    def _lookup_ids(self, item: InventoryItem) -> List[str]:
        ids = []
        for value in (item.market_id, item.condition_id):
            text = str(value or "").strip()
            if text and text not in ids:
                ids.append(text)
        return ids

    def _fetch(self, item: InventoryItem) -> Optional[GammaResolution]:
        last_error = None
        for ident in self._lookup_ids(item):
            try:
                payload = self.getter(ident)
            except Exception as error:  # network / fixture failures stay fail-closed
                last_error = error
                LOG.warning("PAPER SETTLE POLL failed for %s: %s", ident, error)
                continue
            parsed = parse_gamma_resolution(payload, fallback_id=ident) if payload else None
            if parsed is not None:
                return parsed
        if last_error is not None:
            LOG.warning("PAPER SETTLE POLL exhausted lookups for %s", item.market_id)
        return None

    async def _apply_resolution(self, item: InventoryItem, winning: str) -> None:
        process = getattr(self.runner, "process", None)
        if process is None:
            return
        seen = set()
        for ident in (item.market_id, item.condition_id):
            ident = str(ident or "").strip()
            if not ident or ident in seen:
                continue
            seen.add(ident)
            event = {
                "event_type": "market_resolved",
                "market": ident,
                "winning_outcome": winning,
            }
            result = process(event)
            if inspect.isawaitable(result):
                await result

    def _mark_stuck(self, item: InventoryItem, reason: str, *, now: float) -> None:
        ledger = self.runner.ledger
        journal = getattr(self.runner, "negrisk_journal", None) or getattr(
            getattr(self.runner, "negrisk_executor", None), "journal", None
        )
        already = False
        if item.position_id and ledger is not None:
            raw = ledger.state.get("positions", {}).get(item.position_id)
            if isinstance(raw, dict):
                already = bool(raw.get("settlement_stuck"))
                raw["settlement_stuck"] = True
                raw["settlement_stuck_at"] = now
                raw["settlement_stuck_reason"] = reason
                ledger.save()
        if item.basket_id and journal is not None:
            try:
                record = journal._record(item.basket_id)
            except KeyError:
                record = None
            if record is not None:
                already = already or bool(record.get("settlement_stuck"))
                journal.update(
                    item.basket_id,
                    settlement_stuck=True,
                    settlement_stuck_at=now,
                    settlement_stuck_reason=reason,
                )
        label = item.position_id or item.basket_id or item.market_id
        message = (
            f"SETTLEMENT_STUCK {label} market={item.market_id} "
            f"title={item.title!r} reason={reason}"
        )
        if already:
            return
        if self.alert:
            LOG.critical(message)
            record_monitor_exception(
                message, source="paper-settle", kind=SETTLEMENT_STUCK_KIND,
                extra={"market_id": item.market_id, "position_id": item.position_id,
                       "basket_id": item.basket_id, "reason": reason},
            )
        else:
            LOG.warning(message)

    def _past_grace(self, snapshot: GammaResolution, *, now: float) -> bool:
        if snapshot.event_end_ts is None:
            return False
        grace_seconds = max(0.0, self.grace_hours) * 3600.0
        return now >= snapshot.event_end_ts + grace_seconds

    async def reconcile(self, *, now: Optional[float] = None) -> ReconcileReport:
        clock = float(now if now is not None else time.time())
        report = ReconcileReport()
        journal = getattr(self.runner, "negrisk_journal", None) or getattr(
            getattr(self.runner, "negrisk_executor", None), "journal", None
        )
        report.unhedged_leg_count = count_unhedged_legs(journal)
        items = collect_open_inventory(self.runner)
        seen_markets: Set[str] = set()
        for item in items:
            key = item.market_id or item.condition_id or item.position_id or item.basket_id
            if not key or key in seen_markets:
                # Still poll distinct market_ids only; items that share a market
                # settle together through the existing handler.
                if item.market_id and item.market_id in seen_markets:
                    continue
            if item.market_id:
                seen_markets.add(item.market_id)
            if item.condition_id:
                seen_markets.add(item.condition_id)
            snapshot = self._fetch(item)
            report.polled += 1
            if snapshot is None:
                continue
            if snapshot.resolved and snapshot.winning_outcome:
                await self._apply_resolution(item, snapshot.winning_outcome)
                self._record_settled(item, report)
                continue
            if not self._past_grace(snapshot, now=clock):
                continue
            if snapshot.uma_known_or_finalized and item.complete_set:
                winner = snapshot.winning_outcome or UMA_FINALIZED_OUTCOME
                await self._apply_resolution(item, winner)
                if self._inventory_still_open(item):
                    self._mark_stuck(
                        item,
                        f"uma_{snapshot.uma_status or 'known'}_force_settle_failed",
                        now=clock,
                    )
                    report.stuck.append(item.position_id or item.basket_id)
                else:
                    self._record_settled(item, report)
                continue
            self._mark_stuck(
                item,
                (
                    f"past_grace uma={snapshot.uma_status or 'unknown'} "
                    f"winner={snapshot.winning_outcome or 'none'}"
                ),
                now=clock,
            )
            report.stuck.append(item.position_id or item.basket_id)
        report.unhedged_leg_count = count_unhedged_legs(journal)
        return report

    def _inventory_still_open(self, item: InventoryItem) -> bool:
        ledger = self.runner.ledger
        if item.position_id and ledger is not None:
            raw = ledger.state.get("positions", {}).get(item.position_id)
            if isinstance(raw, dict) and not raw.get("settled"):
                return True
        journal = getattr(self.runner, "negrisk_journal", None) or getattr(
            getattr(self.runner, "negrisk_executor", None), "journal", None
        )
        if item.basket_id and journal is not None:
            try:
                record = journal._record(item.basket_id)
            except KeyError:
                return False
            return str(record.get("status") or "") not in getattr(
                journal, "TERMINAL_STATUSES", {"SETTLED"}
            )
        return False

    def _record_settled(self, item: InventoryItem, report: ReconcileReport) -> None:
        if item.position_id:
            report.settled_positions.append(item.position_id)
        if item.basket_id:
            report.settled_baskets.append(item.basket_id)


async def run_paper_settle_loop(runner, stop: asyncio.Event,
                                *, getter: Optional[GammaGetter] = None) -> None:
    """Paper-only background poll. Live runners return immediately."""
    if getattr(runner, "live", False) or getattr(runner, "ledger", None) is None:
        return
    interval = paper_settle_interval_sec()
    try:
        reconciler = PaperSettlementReconciler(runner, getter=getter)
    except RuntimeError:
        return
    LOG.info(
        "PAPER SETTLE RECONCILE: interval=%ss grace=%sh alert=%s",
        int(interval), reconciler.grace_hours, reconciler.alert,
    )
    try:
        while not stop.is_set():
            try:
                report = await reconciler.reconcile()
                if report.settled_positions or report.settled_baskets:
                    LOG.info(
                        "PAPER SETTLE RECONCILE: settled_positions=%s settled_baskets=%s stuck=%s",
                        len(report.settled_positions), len(report.settled_baskets),
                        len(report.stuck),
                    )
            except Exception:
                LOG.exception("paper settlement reconcile failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
