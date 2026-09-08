#!/usr/bin/env python3
"""Deterministic JSONL replay for recorded Polymarket CLOB events.

The replay path intentionally uses the same market-event normalizer and binary
scanner as paper/live mode. It is a fill opportunity report, not a claim that
every historical opportunity was executable: real fills still need to be
validated against account logs and latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional
import random

from .arbitrage_core import (
    ArbitrageOpportunity,
    BinaryArbitrageScanner,
    BinaryMarket,
    OrderBook,
    handle_market_event,
)
from .polymarket_edge import PolymarketFeeModel

LOG = logging.getLogger(__name__)

# Paper/live MARKET_EVENT_LOG used to append forever (one run hit ~52GB).
# Size-rotate so a long paper tape cannot fill the disk, but keep the newest
# active file plus N rotated siblings for RCA / replay.
DEFAULT_MARKET_EVENT_LOG_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
DEFAULT_MARKET_EVENT_LOG_BACKUP_COUNT = 1
_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip(), 10)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def market_event_log_max_bytes(explicit: Optional[int] = None) -> int:
    """Active-file size cap. Values <= 0 fall back to the 1 GiB default."""
    if explicit is not None:
        return int(explicit) if explicit > 0 else DEFAULT_MARKET_EVENT_LOG_MAX_BYTES
    return _env_non_negative_int(
        "MARKET_EVENT_LOG_MAX_BYTES", DEFAULT_MARKET_EVENT_LOG_MAX_BYTES
    ) or DEFAULT_MARKET_EVENT_LOG_MAX_BYTES


def market_event_log_backup_count(explicit: Optional[int] = None) -> int:
    """How many rotated siblings to keep (path.1, path.2, ...)."""
    if explicit is not None:
        return max(0, int(explicit))
    return _env_non_negative_int(
        "MARKET_EVENT_LOG_BACKUP_COUNT", DEFAULT_MARKET_EVENT_LOG_BACKUP_COUNT
    )


def _lock_for_path(path: str) -> threading.Lock:
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[path] = lock
        return lock


class JsonlEventRecorder:
    """Append raw/typed stream events with local receipt metadata.

    When ``MARKET_EVENT_LOG`` is set, the active JSONL is rotated (not
    disabled) once it reaches ``MARKET_EVENT_LOG_MAX_BYTES``. The newest
    events stay in the active file; older chunks are kept as ``path.1`` …
    ``path.N`` up to ``MARKET_EVENT_LOG_BACKUP_COUNT``.
    """

    def __init__(
        self,
        path: str = "",
        source: str = "market",
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
    ):
        self.path = path
        self.source = source
        self.max_bytes = market_event_log_max_bytes(max_bytes)
        self.backup_count = market_event_log_backup_count(backup_count)
        self.rotations = 0

    @staticmethod
    def _jsonable(event):
        dump = getattr(event, "model_dump", None)
        if dump:
            return dump(mode="json", by_alias=True)
        if isinstance(event, dict):
            return dict(event)
        raise TypeError(f"event is not JSON serializable: {type(event).__name__}")

    def _file_size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def _rotate(self) -> None:
        path = os.path.abspath(self.path)
        backups = self.backup_count
        if backups <= 0:
            # Still keep writing; drop the bloated active file only.
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            self.rotations += 1
            LOG.info(
                "Truncated MARKET_EVENT_LOG %s after exceeding %d bytes "
                "(backup_count=0; logging continues)",
                path, self.max_bytes,
            )
            with open(path, "a", encoding="utf-8"):
                pass
            return
        oldest = f"{path}.{backups}"
        try:
            os.remove(oldest)
        except FileNotFoundError:
            pass
        for index in range(backups - 1, 0, -1):
            src = f"{path}.{index}"
            dst = f"{path}.{index + 1}"
            if os.path.isfile(src):
                os.replace(src, dst)
        if os.path.isfile(path):
            os.replace(path, f"{path}.1")
        with open(path, "a", encoding="utf-8"):
            pass
        self.rotations += 1
        LOG.info(
            "Rotated MARKET_EVENT_LOG %s -> %s.1 (max_bytes=%d backup_count=%d)",
            path, path, self.max_bytes, backups,
        )

    def _maybe_rotate(self) -> None:
        if self._file_size() >= self.max_bytes:
            self._rotate()

    def record(self, event, received_at_ms: Optional[int] = None) -> None:
        if not self.path:
            return
        payload = self._jsonable(event)
        if not isinstance(payload, dict):
            raise TypeError("recorded event must serialize to an object")
        payload.setdefault("source", self.source)
        payload.setdefault(
            "received_at_ms", int(received_at_ms if received_at_ms is not None else time.time() * 1000)
        )
        abs_path = os.path.abspath(self.path)
        directory = os.path.dirname(abs_path) or "."
        os.makedirs(directory, exist_ok=True)
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        with _lock_for_path(abs_path):
            self._maybe_rotate()
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._maybe_rotate()


@dataclass(frozen=True)
class FillModel:
    """Replay-only fill assumptions. These are not live fill statistics."""

    fill_probability: float = 1.0
    rejection_rate: float = 0.0
    second_leg_failure_rate: float = 0.0
    queue_ahead_shares: float = 0.0
    seed: int = 1

    def __post_init__(self):
        object.__setattr__(self, "fill_probability", min(1.0, max(0.0, float(self.fill_probability))))
        object.__setattr__(self, "rejection_rate", min(1.0, max(0.0, float(self.rejection_rate))))
        object.__setattr__(self, "second_leg_failure_rate", min(1.0, max(0.0, float(self.second_leg_failure_rate))))
        object.__setattr__(self, "queue_ahead_shares", max(0.0, float(self.queue_ahead_shares)))


@dataclass(frozen=True)
class ReplayOpportunity:
    event_index: int
    market_id: str
    timestamp_ms: int
    shares: float
    capital_required: float
    gross_profit: float
    net_profit: float
    fingerprint: str

    @classmethod
    def from_opportunity(cls, event_index: int, opportunity: ArbitrageOpportunity) -> "ReplayOpportunity":
        return cls(
            event_index=event_index,
            market_id=opportunity.market_id,
            timestamp_ms=opportunity.book_timestamp_ms,
            shares=opportunity.shares,
            capital_required=opportunity.capital_required,
            gross_profit=opportunity.gross_profit,
            net_profit=opportunity.net_profit,
            fingerprint=opportunity.fingerprint,
        )


class BinaryMarketReplay:
    """Replay recorded CLOB events with configurable execution latency."""

    def __init__(self, markets: Iterable[BinaryMarket], scanner: Optional[BinaryArbitrageScanner] = None,
                 consume_fills: bool = False, execution_latency_ms: int = 0,
                 max_book_age_seconds: Optional[float] = None,
                 fill_model: Optional[FillModel] = None,
                 max_leg_skew_ms: int = 1000):
        self.markets: Dict[str, BinaryMarket] = {market.market_id: market for market in markets}
        self.token_to_market = {
            token: market
            for market in self.markets.values()
            for token in (market.yes_token_id, market.no_token_id)
        }
        self.books: Dict[str, OrderBook] = {}
        self.scanner = scanner or BinaryArbitrageScanner()
        self.consume_fills = consume_fills
        self.execution_latency_ms = max(0, int(execution_latency_ms))
        self.max_book_age_ms = (
            None if max_book_age_seconds is None
            else max(1, int(float(max_book_age_seconds) * 1000))
        )
        self.max_leg_skew_ms = max(0, int(max_leg_skew_ms))
        self.fill_model = fill_model or FillModel()
        self._rng = random.Random(self.fill_model.seed)
        self.opportunities: List[ReplayOpportunity] = []
        self.executed_opportunities: List[ReplayOpportunity] = []
        self.pending = []
        self.fee_overrides: Dict[str, float] = {}
        self.execution_stats = {
            "signals": 0, "executed": 0, "latency_missed": 0,
            "stale_missed": 0, "depth_missed": 0, "pending_at_end": 0,
            "rejected": 0, "queue_missed": 0, "second_leg_failed": 0,
            "fee_backfills": 0, "unsynced_skipped": 0, "skew_missed": 0,
            "simulated": True,
        }

    @staticmethod
    def _event_time_ms(event: dict) -> int:
        value = event.get("received_at_ms")
        if value is None:
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else event
            value = payload.get("timestamp")
        try:
            numeric = float(value)
            if numeric < 10_000_000_000:
                numeric *= 1000
            return int(numeric) if numeric > 0 else 0
        except (TypeError, ValueError):
            try:
                return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
            except (TypeError, ValueError, OverflowError):
                return 0

    def _backfill_fee_rate(self, event: dict) -> None:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        rate = payload.get("fee_rate_bps", event.get("fee_rate_bps"))
        token_id = str(payload.get("asset_id", payload.get("token_id", "")) or "")
        if rate is None or not token_id:
            return
        market = self.token_to_market.get(token_id)
        if market is None:
            return
        try:
            numeric = float(rate)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric) or numeric < 0.0:
            return
        if numeric > 1.0:
            numeric /= 10_000.0
        self.fee_overrides[market.market_id] = numeric
        self.execution_stats["fee_backfills"] += 1

    def _fee_model_for(self, market: BinaryMarket) -> PolymarketFeeModel:
        rate = self.fee_overrides.get(market.market_id, market.taker_fee_rate)
        return PolymarketFeeModel(
            market.category, taker_fee_rate=rate, fee_exponent=market.fee_exponent,
            fees_enabled=market.fees_enabled,
        )

    def _flush_pending(self, now_ms: int) -> None:
        if not self.pending or now_ms <= 0:
            return
        remaining = []
        for pending in self.pending:
            if pending["due_ms"] > now_ms:
                remaining.append(pending)
                continue
            market = self.markets[pending["market_id"]]
            yes_book = self.books.get(market.yes_token_id)
            no_book = self.books.get(market.no_token_id)
            if not yes_book or not no_book or not yes_book.synced or not no_book.synced:
                self.execution_stats["depth_missed"] += 1
                continue
            if abs(yes_book.timestamp_ms - no_book.timestamp_ms) > self.max_leg_skew_ms:
                self.execution_stats["skew_missed"] += 1
                continue
            if self.max_book_age_ms is not None and now_ms - min(yes_book.timestamp_ms, no_book.timestamp_ms) > self.max_book_age_ms:
                self.execution_stats["stale_missed"] += 1
                continue
            yes_cost, _, yes_fills = yes_book.walk_asks(pending["shares"])
            no_cost, _, no_fills = no_book.walk_asks(pending["shares"])
            yes_worst = yes_fills[-1][0] if yes_fills else float("inf")
            no_worst = no_fills[-1][0] if no_fills else float("inf")
            if (not yes_fills or not no_fills
                    or yes_cost <= 0.0 or no_cost <= 0.0
                    or yes_worst > pending["yes_worst_price"] + 1e-12
                    or no_worst > pending["no_worst_price"] + 1e-12):
                self.execution_stats["latency_missed"] += 1
                continue
            if not self._accept_simulated_fill(pending["opportunity"], yes_book, no_book, yes_fills, no_fills):
                continue
            yes_book.consume_asks(yes_fills)
            no_book.consume_asks(no_fills)
            self.executed_opportunities.append(pending["opportunity"])
            self.execution_stats["executed"] += 1
        self.pending = remaining

    def process(self, event: dict, event_index: int = 0) -> List[ReplayOpportunity]:
        found: List[ReplayOpportunity] = []
        event_time_ms = self._event_time_ms(event)
        self._flush_pending(event_time_ms)
        self._backfill_fee_rate(event)
        for market_id in handle_market_event(event, self.token_to_market, self.books):
            market = self.markets[market_id]
            yes_book = self.books.get(market.yes_token_id)
            no_book = self.books.get(market.no_token_id)
            if not yes_book or not no_book:
                continue
            if not yes_book.synced or not no_book.synced:
                self.execution_stats["unsynced_skipped"] += 1
                continue
            if abs(yes_book.timestamp_ms - no_book.timestamp_ms) > self.max_leg_skew_ms:
                self.execution_stats["skew_missed"] += 1
                continue
            if (self.max_book_age_ms is not None and event_time_ms > 0
                    and event_time_ms - min(yes_book.timestamp_ms, no_book.timestamp_ms) > self.max_book_age_ms):
                self.execution_stats["stale_missed"] += 1
                continue
            opportunity = self.scanner.scan(market, yes_book, no_book, fee_model=self._fee_model_for(market))
            if opportunity is None:
                continue
            record = ReplayOpportunity.from_opportunity(event_index, opportunity)
            self.opportunities.append(record)
            found.append(record)
            self.execution_stats["signals"] += 1
            if self.consume_fills:
                if self.execution_latency_ms == 0:
                    _, _, yes_fills = yes_book.walk_asks(opportunity.shares)
                    _, _, no_fills = no_book.walk_asks(opportunity.shares)
                    if self._accept_simulated_fill(record, yes_book, no_book, yes_fills, no_fills):
                        yes_book.consume_asks(yes_fills)
                        no_book.consume_asks(no_fills)
                        self.executed_opportunities.append(record)
                        self.execution_stats["executed"] += 1
                else:
                    self.pending.append({
                        "market_id": market_id,
                        "opportunity": record,
                        "shares": opportunity.shares,
                        "due_ms": (event_time_ms or record.timestamp_ms) + self.execution_latency_ms,
                        "yes_worst_price": opportunity.yes_worst_price,
                        "no_worst_price": opportunity.no_worst_price,
                    })
        return found

    def _accept_simulated_fill(self, opportunity: ReplayOpportunity, yes_book, no_book,
                               yes_fills, no_fills) -> bool:
        """Apply replay fill/reject/queue assumptions. Result is simulated PnL only."""
        model = self.fill_model
        if self._rng.random() < model.rejection_rate:
            self.execution_stats["rejected"] += 1
            return False
        if self._rng.random() > model.fill_probability:
            self.execution_stats["rejected"] += 1
            return False
        if model.queue_ahead_shares > 0.0:
            yes_available = sum(size for _, size in yes_book.asks_sorted())
            no_available = sum(size for _, size in no_book.asks_sorted())
            if (yes_available - model.queue_ahead_shares + 1e-12 < opportunity.shares
                    or no_available - model.queue_ahead_shares + 1e-12 < opportunity.shares):
                self.execution_stats["queue_missed"] += 1
                return False
        if self._rng.random() < model.second_leg_failure_rate:
            self.execution_stats["second_leg_failed"] += 1
            return False
        return True

    def run(self, events: Iterable[dict]) -> List[ReplayOpportunity]:
        for index, event in enumerate(events):
            if isinstance(event, dict):
                self.process(event, event_index=index)
        self.execution_stats["pending_at_end"] = len(self.pending)
        return list(self.opportunities)

    def report(self) -> dict:
        by_market: Dict[str, dict] = {}
        for opportunity in self.opportunities:
            row = by_market.setdefault(opportunity.market_id, {
                "opportunities": 0, "gross_profit": 0.0,
                "net_profit": 0.0, "capital_required": 0.0,
            })
            row["opportunities"] += 1
            row["gross_profit"] += opportunity.gross_profit
            row["net_profit"] += opportunity.net_profit
            row["capital_required"] += opportunity.capital_required
        return {
            "opportunities": len(self.opportunities),
            "gross_profit": sum(item.gross_profit for item in self.opportunities),
            "net_profit": sum(item.net_profit for item in self.opportunities),
            "capital_required": sum(item.capital_required for item in self.opportunities),
            "executed_opportunities": len(self.executed_opportunities),
            "executed_gross_profit": sum(item.gross_profit for item in self.executed_opportunities),
            "executed_net_profit": sum(item.net_profit for item in self.executed_opportunities),
            "executed_capital_required": sum(
                item.capital_required for item in self.executed_opportunities
            ),
            "pnl_is_simulated": True,
            "execution": dict(self.execution_stats),
            "by_market": by_market,
        }


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_events(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(event, dict):
                raise ValueError(f"event at line {line_number} is not an object")
            yield event


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay recorded Polymarket binary CLOB events")
    parser.add_argument("--markets", required=True, help="JSON file containing Gamma market objects")
    parser.add_argument("--events", required=True, help="JSONL file containing raw CLOB market events")
    parser.add_argument("--consume-fills", action="store_true", help="consume simulated fills from the local book")
    parser.add_argument("--execution-latency-ms", type=int, default=0,
                        help="delay simulated fills by this many milliseconds")
    parser.add_argument("--max-book-age-seconds", type=float, default=None,
                        help="reject opportunities whose two books are older than this")
    parser.add_argument("--max-leg-skew-ms", type=int, default=1000,
                        help="reject opportunities whose Yes/No book timestamps differ by more than this")
    parser.add_argument("--fill-probability", type=float, default=1.0)
    parser.add_argument("--rejection-rate", type=float, default=0.0)
    parser.add_argument("--second-leg-failure-rate", type=float, default=0.0)
    parser.add_argument("--queue-ahead-shares", type=float, default=0.0)
    parser.add_argument("--fill-seed", type=int, default=1)
    parser.add_argument("--max-order", type=float, default=100.0)
    parser.add_argument("--min-profit", type=float, default=0.05)
    parser.add_argument("--min-return", type=float, default=0.002)
    parser.add_argument("--buffer", type=float, default=0.02)
    args = parser.parse_args()

    raw_markets = _load_json(args.markets)
    rows = raw_markets if isinstance(raw_markets, list) else raw_markets.get("data", [])
    markets = [
        parsed for row in rows if isinstance(row, dict)
        for parsed in [BinaryMarket.from_gamma(row)] if parsed and parsed.active
    ]
    replay = BinaryMarketReplay(
        markets,
        scanner=BinaryArbitrageScanner(
            min_net_profit_usd=args.min_profit,
            min_return=args.min_return,
            safety_buffer_usd=args.buffer,
            max_order_usd=args.max_order,
        ),
        consume_fills=args.consume_fills,
        execution_latency_ms=args.execution_latency_ms,
        max_book_age_seconds=args.max_book_age_seconds,
        max_leg_skew_ms=args.max_leg_skew_ms,
        fill_model=FillModel(
            fill_probability=args.fill_probability,
            rejection_rate=args.rejection_rate,
            second_leg_failure_rate=args.second_leg_failure_rate,
            queue_ahead_shares=args.queue_ahead_shares,
            seed=args.fill_seed,
        ),
    )
    replay.run(_load_events(args.events))
    print(json.dumps(replay.report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
