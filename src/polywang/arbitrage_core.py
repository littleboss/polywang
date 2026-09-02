#!/usr/bin/env python3
"""Deterministic binary-market arbitrage primitives for Polymarket.

This module keeps live submission behind the verified official CLOB client and
explicit account/journal checks.  It is also usable for historical replay and
paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import asyncio
import inspect
import json
import math
import os
import tempfile
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .polymarket_edge import PolymarketFeeModel, resolve_category


def maker_gtc_enabled() -> bool:
    """Maker/GTC resting orders for binary combo arb. Default off."""
    return os.getenv("ENABLE_MAKER_GTC", "").strip().lower() in {"1", "true", "yes", "on"}


def maker_rest_seconds(default: float = 30.0) -> float:
    raw = os.getenv("MAKER_REST_SECONDS", "")
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return value if math.isfinite(value) and value >= 0.0 else default


def maker_limit_price(worst_price: float, tick_size: float) -> float:
    """Post one tick better than the scanned ask so post-only does not cross."""
    tick = tick_size if math.isfinite(tick_size) and tick_size > 0.0 else 0.01
    if not math.isfinite(worst_price) or worst_price <= 0.0:
        return tick
    return max(tick, worst_price - tick)


def _json_list(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return []
    return value if isinstance(value, list) else []


def _gamma_named_price(payload: dict, outcomes: Optional[list], name: str) -> Optional[float]:
    """Best-effort Yes/No price from Gamma outcomePrices. Missing is None."""
    names = outcomes if outcomes is not None else _json_list(payload.get("outcomes"))
    prices = _json_list(payload.get("outcomePrices", payload.get("outcome_prices")))
    if len(names) != 2 or len(prices) != 2:
        return None
    try:
        by_outcome = {
            str(outcome).strip().lower(): float(price)
            for outcome, price in zip(names, prices)
        }
    except (TypeError, ValueError):
        return None
    price = by_outcome.get(str(name).strip().lower())
    if price is None or not math.isfinite(price) or not (0.0 < price < 1.0):
        return None
    return price


def _gamma_yes_price(payload: dict, outcomes: Optional[list] = None) -> Optional[float]:
    """Best-effort Yes probability from a Gamma market object. Missing is None."""
    return _gamma_named_price(payload, outcomes, "yes")


def _gamma_no_price(payload: dict, outcomes: Optional[list] = None) -> Optional[float]:
    """Best-effort No price from Gamma outcomePrices. Missing is None."""
    return _gamma_named_price(payload, outcomes, "no")


def _gamma_outcome_prices(payload: dict) -> Optional[Dict[str, float]]:
    names = _json_list(payload.get("outcomes"))
    prices = _json_list(payload.get("outcomePrices", payload.get("outcome_prices")))
    if len(names) < 3 or len(names) != len(prices):
        return None
    parsed: Dict[str, float] = {}
    for name, price in zip(names, prices):
        try:
            numeric = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric < 0.0:
            return None
        label = str(name).strip()
        if not label:
            return None
        parsed[label] = numeric
    return parsed if parsed else None


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes"}:
            return True
        if value.strip().lower() in {"false", "0", "no", ""}:
            return False
    return default if value is None else bool(value)


def _category(raw: object) -> str:
    """
    Normalise a Gamma tag to the category whose fee schedule applies.

    Shares one alias table with the fee model rather than keeping a second one
    here. A tag that only this function knew about would still be priced at the
    default rate downstream, which is the bug this replaced: "Bitcoin" resolved
    to itself, missed the rate table, and was charged 0.05 against a real 0.07.
    """
    return resolve_category(raw)


# Market-channel schema versions this process can apply. An unknown version is
# a gap: the local book must be discarded rather than guessed into.
KNOWN_MARKET_SCHEMA_VERSIONS = frozenset({"", "1", "1.0", "v1"})


def _optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


@dataclass(frozen=True)
class BinaryMarket:
    market_id: str
    condition_id: str
    title: str
    yes_token_id: str
    no_token_id: str
    category: str = "other"
    resolution_ts: Optional[float] = None
    active: bool = True
    neg_risk: bool = False
    min_order_size: float = 0.0
    tick_size: float = 0.01
    taker_fee_rate: Optional[float] = None
    fee_exponent: float = 1.0
    fees_enabled: bool = True
    implied_yes: Optional[float] = None
    implied_no: Optional[float] = None

    @classmethod
    def from_gamma(cls, payload: dict) -> Optional["BinaryMarket"]:
        tokens = _json_list(payload.get("clobTokenIds", payload.get("clob_token_ids")))
        outcomes = _json_list(payload.get("outcomes"))
        if len(tokens) != 2 or len(outcomes) != 2:
            return None

        by_outcome = {str(outcome).strip().lower(): str(token) for outcome, token in zip(outcomes, tokens)}
        if "yes" not in by_outcome or "no" not in by_outcome:
            return None
        market_id = str(payload.get("id", payload.get("conditionId", ""))).strip()
        condition_id = str(payload.get("conditionId", payload.get("condition_id", payload.get("id", "")))).strip()
        yes_token_id = by_outcome["yes"].strip()
        no_token_id = by_outcome["no"].strip()
        if not market_id or not condition_id or not yes_token_id or not no_token_id:
            return None
        if yes_token_id == no_token_id:
            return None

        raw_end = payload.get("endDate", payload.get("end_date"))
        resolution_ts = None
        if raw_end:
            try:
                from datetime import datetime
                text = str(raw_end).replace("Z", "+00:00")
                resolution_ts = datetime.fromisoformat(text).timestamp()
            except (TypeError, ValueError, OverflowError):
                resolution_ts = None

        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tags = _json_list(tags)
        tag = tags[0] if tags else payload.get("category", "other")
        fee_schedule = payload.get("feeSchedule", payload.get("fee_schedule", {}))
        if isinstance(fee_schedule, str):
            try:
                fee_schedule = json.loads(fee_schedule)
            except (TypeError, ValueError):
                fee_schedule = {}
        if not isinstance(fee_schedule, dict):
            fee_schedule = {}
        raw_fee_rate = fee_schedule.get("rate", payload.get("taker_fee_rate"))
        raw_fee_exponent = fee_schedule.get("exponent", payload.get("fee_exponent", 1.0))
        try:
            fee_rate = float(raw_fee_rate) if raw_fee_rate is not None else None
            if fee_rate is not None and fee_rate > 1.0:
                fee_rate /= 10_000.0
            fee_exponent = float(raw_fee_exponent)
        except (TypeError, ValueError):
            fee_rate, fee_exponent = None, 1.0
        if fee_rate is not None and (not math.isfinite(fee_rate) or fee_rate < 0.0):
            fee_rate = None
        if not math.isfinite(fee_exponent) or fee_exponent < 0.0:
            fee_exponent = 1.0
        try:
            min_order_size = float(payload.get("minimum_order_size", 0.0) or 0.0)
            tick_size = float(payload.get("order_price_min_tick_size", 0.01) or 0.01)
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(min_order_size) or not math.isfinite(tick_size)
                or min_order_size < 0.0 or tick_size <= 0.0):
            return None
        return cls(
            market_id=market_id,
            condition_id=condition_id,
            title=str(payload.get("question", payload.get("title", payload.get("slug", "")))),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            category=_category(payload.get("category", tag)),
            resolution_ts=resolution_ts,
            active=_as_bool(payload.get("active", True), True) and not _as_bool(payload.get("closed", False)),
            neg_risk=_as_bool(payload.get("negRisk", payload.get("neg_risk", False))),
            min_order_size=min_order_size,
            tick_size=tick_size,
            taker_fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            # A market can run with fees switched off whatever its category, and
            # charging it the category rate would reject entries that are free.
            fees_enabled=_as_bool(payload.get("feesEnabled", payload.get("fees_enabled", True)), True),
            implied_yes=_gamma_yes_price(payload, outcomes),
            implied_no=_gamma_no_price(payload, outcomes),
        )


class OrderBook:
    """Local order book built from official market-channel events.

    Incremental updates are fail-closed: a sequence gap, hash-chain break, or
    unknown schema version discards the book. Trading on a book that missed an
    update is worse than standing down until the next snapshot.
    """

    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.timestamp_ms: int = 0
        self.hash: str = ""
        self.sequence: Optional[int] = None
        self.schema_version: str = ""
        self.synced: bool = False
        self.gap_reason: str = ""

    def invalidate(self, reason: str = "unsynced") -> None:
        """Drop resting levels so a partial book cannot be scanned."""
        self.bids.clear()
        self.asks.clear()
        self.synced = False
        self.gap_reason = str(reason or "unsynced")

    @staticmethod
    def _levels(levels: Iterable[dict]) -> Dict[float, float]:
        result = {}
        for level in levels or []:
            try:
                price = float(_response_value(level, "price", default=None))
                size = float(_response_value(level, "size", default=None))
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0.0:
                result[price] = size
        return result

    def replace_snapshot(self, event: dict) -> bool:
        if not self._set_meta(event, allow_equal=True, is_snapshot=True):
            return False
        self.bids = self._levels(_response_value(event, "bids", default=()))
        self.asks = self._levels(_response_value(event, "asks", default=()))
        self.synced = True
        self.gap_reason = ""
        return True

    def apply_change(self, side: str, price: float, size: float,
                     event: Optional[dict] = None, level_hash: Optional[str] = None) -> bool:
        normalized_side = str(side).upper()
        if normalized_side in ("BUY", "BID", "BIDS"):
            book = self.bids
        elif normalized_side in ("SELL", "ASK", "ASKS"):
            book = self.asks
        else:
            return False
        if not self.synced:
            return False
        if not (0.0 < price < 1.0):
            return False
        if event and not self._set_meta(event, level_hash=level_hash, allow_equal=True, is_snapshot=False):
            return False
        if size <= 0.0:
            book.pop(price, None)
        else:
            book[price] = size
        if not event and level_hash:
            self.hash = str(level_hash)
        return True

    def _schema_ok(self, event: dict) -> bool:
        raw = _response_value(event, "schema_version", "version", "clob_version", default="")
        version = str(raw or "").strip()
        if version and version not in KNOWN_MARKET_SCHEMA_VERSIONS:
            self.invalidate(f"unsupported market-channel schema version {version!r}")
            return False
        if version:
            self.schema_version = version
        return True

    def _sequence_ok(self, event: dict, is_snapshot: bool) -> bool:
        sequence = _optional_int(_response_value(
            event, "sequence", "seq", "event_sequence", default=None
        ))
        if sequence is None:
            return True
        if is_snapshot or self.sequence is None:
            self.sequence = sequence
            return True
        if sequence == self.sequence or sequence == self.sequence + 1:
            self.sequence = sequence
            return True
        self.invalidate(f"sequence gap: expected {self.sequence + 1}, got {sequence}")
        return False

    def _hash_chain_ok(self, event: dict, is_snapshot: bool) -> bool:
        prev = _response_value(event, "prev_hash", "previous_hash", "parent_hash", default=None)
        if is_snapshot or not prev or not self.hash:
            return True
        if str(prev) == str(self.hash):
            return True
        self.invalidate(f"hash chain break: local {self.hash} != prev {prev}")
        return False

    def _set_meta(self, event: dict, level_hash: Optional[str] = None,
                  allow_equal: bool = False, is_snapshot: bool = False) -> bool:
        try:
            raw_timestamp = _response_value(event, "timestamp", default=0)
            if hasattr(raw_timestamp, "timestamp"):
                timestamp = int(float(raw_timestamp.timestamp()) * 1000)
            else:
                try:
                    timestamp = int(float(raw_timestamp))
                except (TypeError, ValueError):
                    from datetime import datetime
                    timestamp = int(datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00")).timestamp() * 1000)
            # The CLOB wire format uses epoch milliseconds. Accept epoch
            # seconds only when the value is recognisably a current epoch;
            # small fixture values remain useful in deterministic tests.
            if 1_000_000_000 <= timestamp < 10_000_000_000:
                timestamp *= 1000
            if timestamp > 0 and self.timestamp_ms > 0:
                if timestamp < self.timestamp_ms or (not allow_equal and timestamp == self.timestamp_ms):
                    return False
            if timestamp > 0:
                self.timestamp_ms = timestamp
        except (TypeError, ValueError):
            return False
        if timestamp <= 0:
            return False
        if not self._schema_ok(event):
            return False
        if not self._sequence_ok(event, is_snapshot=is_snapshot):
            return False
        if not self._hash_chain_ok(event, is_snapshot=is_snapshot):
            return False
        raw_hash = level_hash if level_hash is not None else _response_value(event, "hash", default=None)
        if raw_hash:
            self.hash = str(raw_hash)
        return True

    def asks_sorted(self) -> List[Tuple[float, float]]:
        return sorted(self.asks.items())

    def bids_sorted(self) -> List[Tuple[float, float]]:
        return sorted(self.bids.items(), reverse=True)

    def best_ask(self) -> Optional[Tuple[float, float]]:
        levels = self.asks_sorted()
        return levels[0] if levels else None

    def best_bid(self) -> Optional[Tuple[float, float]]:
        levels = self.bids_sorted()
        return levels[0] if levels else None

    def walk_asks(self, shares: float) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Return total cost, average price, and consumed (price, shares)."""
        remaining = max(0.0, shares)
        cost = 0.0
        fills: List[Tuple[float, float]] = []
        for price, available in self.asks_sorted():
            if remaining <= 1e-12:
                break
            quantity = min(remaining, available)
            cost += quantity * price
            fills.append((price, quantity))
            remaining -= quantity
        filled = shares - remaining
        return cost, (cost / filled if filled else 0.0), fills

    def consume_asks(self, fills: Sequence[Tuple[float, float]]) -> None:
        for price, quantity in fills:
            available = self.asks.get(price, 0.0)
            remaining = available - quantity
            if remaining <= 1e-12:
                self.asks.pop(price, None)
            else:
                self.asks[price] = remaining

    def walk_bids(self, shares: float) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Return total proceeds, average price, and consumed bid (price, shares)."""
        remaining = max(0.0, shares)
        proceeds = 0.0
        fills: List[Tuple[float, float]] = []
        for price, available in self.bids_sorted():
            if remaining <= 1e-12:
                break
            quantity = min(remaining, available)
            proceeds += quantity * price
            fills.append((price, quantity))
            remaining -= quantity
        filled = shares - remaining
        return proceeds, (proceeds / filled if filled else 0.0), fills

    def consume_bids(self, fills: Sequence[Tuple[float, float]]) -> None:
        for price, quantity in fills:
            available = self.bids.get(price, 0.0)
            remaining = available - quantity
            if remaining <= 1e-12:
                self.bids.pop(price, None)
            else:
                self.bids[price] = remaining


@dataclass
class ArbitrageOpportunity:
    market_id: str
    condition_id: str
    title: str
    yes_token_id: str
    no_token_id: str
    category: str
    shares: float
    yes_cost: float
    no_cost: float
    yes_fee: float
    no_fee: float
    gross_profit: float
    net_profit: float
    return_on_capital: float
    yes_average_price: float
    no_average_price: float
    yes_worst_price: float
    no_worst_price: float
    yes_execution_amount: float
    no_execution_amount: float
    yes_execution_fee_cap: float
    no_execution_fee_cap: float
    execution_capital_required: float
    book_timestamp_ms: int
    fingerprint: str
    merge_gas_usd: float = 0.0
    is_risk_free: bool = False
    is_taker: bool = True
    order_style: str = "FOK"
    tick_size: float = 0.01
    residual_risk: str = (
        "sequential FOK legs are not atomic; a first-leg fill with a second-leg "
        "failure is unwound with FAK and may slip, partially fill, or remain open"
    )

    @property
    def capital_required(self) -> float:
        return self.yes_cost + self.no_cost + self.yes_fee + self.no_fee


class BinaryArbitrageScanner:
    """Find only buy-both-legs opportunities with a fixed 1.00 settlement."""

    def __init__(self, min_net_profit_usd: float = 0.05,
                 min_return: float = 0.002,
                 safety_buffer_usd: float = 0.02,
                 max_order_usd: float = 100.0,
                 max_levels: Optional[int] = None,
                 merge_gas_usd: float = 0.0):
        self.min_net_profit_usd = max(0.0, min_net_profit_usd)
        self.min_return = max(0.0, min_return)
        self.safety_buffer_usd = max(0.0, safety_buffer_usd)
        self.max_order_usd = max(0.01, max_order_usd)
        self.max_levels = max(1, int(max_levels)) if max_levels is not None else None
        self.merge_gas_usd = max(0.0, float(merge_gas_usd))
        # Set on each scan() call so callers can attribute silent rejects.
        self.last_reject_reason: Optional[str] = None
        self.last_touch_sum: Optional[float] = None
        self.last_best_net: Optional[float] = None

    @staticmethod
    def _cost_for(book: OrderBook, shares: float,
                  levels: Optional[Sequence[Tuple[float, float]]] = None) -> Tuple[float, float, List[Tuple[float, float]]]:
        if levels is None:
            cost, average, fills = book.walk_asks(shares)
        else:
            remaining = max(0.0, shares)
            cost = 0.0
            fills = []
            for price, available in levels:
                if remaining <= 1e-12:
                    break
                quantity = min(remaining, available)
                cost += quantity * price
                fills.append((price, quantity))
                remaining -= quantity
            filled = shares - remaining
            average = cost / filled if filled else 0.0
        filled = sum(quantity for _, quantity in fills)
        if filled + 1e-12 < shares:
            return float("inf"), 0.0, []
        return cost, average, fills

    def scan(self, market: BinaryMarket, yes_book: OrderBook, no_book: OrderBook,
             fee_model: Optional[PolymarketFeeModel] = None,
             is_taker: bool = True) -> Optional[ArbitrageOpportunity]:
        # Negative-risk groups need their own complete-set and redemption
        # semantics. Do not infer them from a two-token price sum.
        self.last_reject_reason = None
        self.last_touch_sum = None
        self.last_best_net = None
        if not market.active or market.neg_risk:
            return None
        if not yes_book.synced or not no_book.synced:
            self.last_reject_reason = "not_synced"
            return None
        yes_touch = yes_book.best_ask()
        no_touch = no_book.best_ask()
        if not yes_touch or not no_touch:
            self.last_reject_reason = "no_touch"
            return None
        self.last_touch_sum = float(yes_touch[0] + no_touch[0])

        yes_levels = yes_book.asks_sorted()
        no_levels = no_book.asks_sorted()
        if self.max_levels is not None:
            yes_levels = yes_levels[:self.max_levels]
            no_levels = no_levels[:self.max_levels]

        taker = bool(is_taker)
        fee_model = fee_model or PolymarketFeeModel(
            market.category,
            taker_fee_rate=market.taker_fee_rate,
            fee_exponent=market.fee_exponent,
            fees_enabled=market.fees_enabled,
        )
        yes_depth = sum(size for _, size in yes_levels)
        no_depth = sum(size for _, size in no_levels)
        max_shares = min(yes_depth, no_depth)
        if max_shares <= 0.0:
            self.last_reject_reason = "no_depth"
            return None

        def execution_capital_for(shares: float) -> float:
            yes_cost, _, yes_fills = self._cost_for(yes_book, shares, yes_levels)
            no_cost, _, no_fills = self._cost_for(no_book, shares, no_levels)
            if not yes_fills or not no_fills:
                return float("inf")
            # A protected BUY order is sized at its worst allowed price. The
            # reservation must therefore cover that ceiling, not only the
            # currently visible average depth cost.
            yes_worst = yes_fills[-1][0]
            no_worst = no_fills[-1][0]
            yes_fee_cap = shares * max(
                (fee_model.fee_per_share(price, is_taker=taker) for price, _ in yes_levels),
                default=0.0,
            )
            no_fee_cap = shares * max(
                (fee_model.fee_per_share(price, is_taker=taker) for price, _ in no_levels),
                default=0.0,
            )
            fees = yes_fee_cap + no_fee_cap
            return shares * (yes_worst + no_worst) + fees

        # Enforce max_order_usd against the actual depth-walked cost, including
        # fees. A touch-price approximation can exceed the configured budget on
        # thin, slippage-heavy books.
        if execution_capital_for(max_shares) > self.max_order_usd:
            low, high = 0.0, max_shares
            for _ in range(60):
                middle = (low + high) / 2.0
                if execution_capital_for(middle) <= self.max_order_usd:
                    low = middle
                else:
                    high = middle
            max_shares = low
        if max_shares <= 1e-12 or (market.min_order_size > 0.0 and max_shares < market.min_order_size):
            self.last_reject_reason = "below_min_size"
            return None

        # Costs are piecewise-linear. Evaluating every cumulative depth boundary
        # is sufficient to find the best fill for two books with fixed levels.
        candidates = {max_shares}
        for _, size in yes_levels:
            candidates.add(min(max_shares, size))
        for _, size in no_levels:
            candidates.add(min(max_shares, size))
        yes_cumulative = 0.0
        for _, size in yes_book.asks_sorted():
            yes_cumulative += size
            candidates.add(min(max_shares, yes_cumulative))
        no_cumulative = 0.0
        for _, size in no_book.asks_sorted():
            no_cumulative += size
            candidates.add(min(max_shares, no_cumulative))

        best = None
        best_any = None
        for shares in sorted(candidates):
            if shares <= 0.0:
                continue
            yes_cost, yes_avg, yes_fills = self._cost_for(yes_book, shares, yes_levels)
            no_cost, no_avg, no_fills = self._cost_for(no_book, shares, no_levels)
            if not yes_fills or not no_fills:
                continue
            yes_fee = sum(fee_model.fee_usd(quantity, price, is_taker=taker) for price, quantity in yes_fills)
            no_fee = sum(fee_model.fee_usd(quantity, price, is_taker=taker) for price, quantity in no_fills)
            yes_worst_price = yes_fills[-1][0]
            no_worst_price = no_fills[-1][0]
            yes_execution_amount = shares * yes_worst_price
            no_execution_amount = shares * no_worst_price
            yes_execution_fee_cap = shares * max(
                (fee_model.fee_per_share(price, is_taker=taker) for price, _ in yes_levels),
                default=0.0,
            )
            no_execution_fee_cap = shares * max(
                (fee_model.fee_per_share(price, is_taker=taker) for price, _ in no_levels),
                default=0.0,
            )
            execution_capital = (
                yes_execution_amount + no_execution_amount
                + yes_execution_fee_cap + no_execution_fee_cap
            )
            gross = shares - yes_cost - no_cost
            net = gross - yes_fee - no_fee - self.safety_buffer_usd - self.merge_gas_usd
            capital = yes_cost + no_cost + yes_fee + no_fee
            result = ArbitrageOpportunity(
                market_id=market.market_id,
                condition_id=market.condition_id,
                title=market.title,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                category=market.category,
                shares=shares,
                yes_cost=yes_cost,
                no_cost=no_cost,
                yes_fee=yes_fee,
                no_fee=no_fee,
                gross_profit=gross,
                net_profit=net,
                return_on_capital=(net / capital if capital > 0 else 0.0),
                yes_average_price=yes_avg,
                no_average_price=no_avg,
                yes_worst_price=yes_worst_price,
                no_worst_price=no_worst_price,
                yes_execution_amount=yes_execution_amount,
                no_execution_amount=no_execution_amount,
                yes_execution_fee_cap=yes_execution_fee_cap,
                no_execution_fee_cap=no_execution_fee_cap,
                execution_capital_required=execution_capital,
                book_timestamp_ms=max(yes_book.timestamp_ms, no_book.timestamp_ms),
                fingerprint=f"{market.market_id}:{yes_book.hash}:{no_book.hash}:{shares:.12f}",
                merge_gas_usd=self.merge_gas_usd,
                is_risk_free=False,
                is_taker=taker,
                order_style="FOK" if taker else "GTC",
                tick_size=market.tick_size if market.tick_size > 0.0 else 0.01,
                residual_risk=(
                    "sequential FOK legs are not atomic; a first-leg fill with a "
                    "second-leg failure is unwound with FAK and may slip, partially "
                    "fill, or remain open"
                    if taker else
                    "resting GTC legs are not atomic; a timeout cancels leftovers "
                    "and FAK-unwinds a one-sided fill, which may slip or remain open"
                ),
            )
            if best_any is None or result.net_profit > best_any.net_profit:
                best_any = result
            if result.net_profit >= self.min_net_profit_usd and result.return_on_capital >= self.min_return:
                if best is None or result.net_profit > best.net_profit:
                    best = result
        if best_any is not None:
            self.last_best_net = float(best_any.net_profit)
        if best is not None:
            return best
        if best_any is None:
            self.last_reject_reason = "no_depth"
            return None
        if best_any.net_profit < self.min_net_profit_usd:
            self.last_reject_reason = "net_below_floor"
        else:
            self.last_reject_reason = "roc_below_floor"
        return None


@dataclass
class PaperPosition:
    position_id: str
    market_id: str
    title: str
    shares: float
    cost: float
    fees: float
    opened_at: float
    settled: bool = False
    payout: float = 0.0


class JsonLedger:
    """Small atomic JSON ledger for paper trading and crash recovery."""

    def __init__(self, path: str, initial_cash: float = 1000.0):
        self.path = path
        self.initial_cash = float(initial_cash)
        self.state = {"initial_cash": float(initial_cash), "cash": float(initial_cash),
                      "positions": {}, "trades": []}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("positions"), dict):
                self.state.update(loaded)
                self.state.setdefault("initial_cash", float(self.state.get("cash", self.initial_cash)))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError):
            raise RuntimeError(f"Ledger is unreadable: {self.path}")

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".ledger-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def open_pair(self, opportunity: ArbitrageOpportunity) -> PaperPosition:
        required = opportunity.capital_required
        if required > float(self.state["cash"]) + 1e-9:
            raise ValueError("insufficient paper cash")
        position_id = f"{opportunity.market_id}:{int(time.time() * 1_000_000)}"
        position = PaperPosition(
            position_id=position_id,
            market_id=opportunity.market_id,
            title=opportunity.title,
            shares=opportunity.shares,
            cost=opportunity.yes_cost + opportunity.no_cost,
            fees=opportunity.yes_fee + opportunity.no_fee,
            opened_at=time.time(),
        )
        self.state["cash"] = float(self.state["cash"]) - required
        self.state["positions"][position_id] = asdict(position)
        self.state["trades"].append({"type": "OPEN_PAIR", "opportunity": asdict(opportunity), "position_id": position_id})
        self.save()
        return position

    def settle(self, position_id: str, winning_outcome: str) -> float:
        raw = self.state["positions"].get(position_id)
        if not raw:
            raise KeyError(position_id)
        if raw.get("settled"):
            return float(raw.get("payout", 0.0))
        payout = float(raw["shares"]) * float(raw.get("payout_per_share", 1.0))
        raw["settled"] = True
        raw["payout"] = payout
        self.state["cash"] = float(self.state["cash"]) + payout
        self.state["trades"].append({"type": "SETTLE_PAIR", "position_id": position_id,
                                     "winning_outcome": winning_outcome, "payout": payout})
        self.save()
        return payout


class PaperArbitrageExecutor:
    def __init__(self, ledger: JsonLedger, max_total_exposure_fraction: float = 0.25,
                 max_market_exposure_fraction: float = 0.05):
        self.ledger = ledger
        self.max_total_exposure_fraction = max(0.0, max_total_exposure_fraction)
        self.max_market_exposure_fraction = max(0.0, max_market_exposure_fraction)

    def _exposure(self) -> float:
        return sum(float(position["cost"]) + float(position["fees"])
                   for position in self.ledger.state["positions"].values()
                   if not position.get("settled"))

    def execute(self, opportunity: ArbitrageOpportunity) -> PaperPosition:
        initial_cash = float(self.ledger.state.get("initial_cash", 0.0))
        if initial_cash <= 0.0:
            initial_cash = float(self.ledger.state["cash"]) + self._exposure()
        required = opportunity.capital_required
        if self._exposure() + required > initial_cash * self.max_total_exposure_fraction + 1e-9:
            raise ValueError("total exposure limit")
        market_exposure = sum(float(position["cost"]) + float(position["fees"])
                              for position in self.ledger.state["positions"].values()
                              if position["market_id"] == opportunity.market_id and not position.get("settled"))
        if market_exposure + required > initial_cash * self.max_market_exposure_fraction + 1e-9:
            raise ValueError("market exposure limit")
        return self.ledger.open_pair(opportunity)


class UnhedgedPairError(RuntimeError):
    """The first leg filled but the compensating leg or rollback failed."""


class MatchingEngineRestartError(RuntimeError):
    """The API reported that the matching engine is restarting (HTTP 425)."""


class CancelOnlyError(RuntimeError):
    """The API is in cancel-only/post-only mode (HTTP 503)."""


class RiskHaltError(RuntimeError):
    """A live risk gate has halted new order placement."""


class LiveOrderJournal:
    """Atomic journal for live pairs and confirmed fills.

    A journal is deliberately separate from the paper ledger: an accepted
    order is not a filled order, and a restart must never reconstruct fills by
    guessing from local intent.  All updates are idempotent so duplicate user
    stream events are harmless.
    """

    TERMINAL_STATUSES = {"ROLLED_BACK", "SETTLED", "REJECTED"}

    def __init__(self, path: str):
        self.path = path
        self.state = {"pairs": {}, "events": [], "trade_watermarks": {}}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("pairs"), dict):
                raise ValueError("missing pairs")
            self.state.update(loaded)
            self.state.setdefault("events", [])
            self.state.setdefault("trade_watermarks", {})
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Live order journal is unreadable: {self.path}") from error

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".live-journal-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def create_pair(self, opportunity: ArbitrageOpportunity) -> str:
        pair_id = f"{opportunity.market_id}:{time.time_ns()}"
        now = time.time()
        self.state["pairs"][pair_id] = {
            "pair_id": pair_id,
            "market_id": opportunity.market_id,
            "condition_id": opportunity.condition_id,
            "title": opportunity.title,
            "yes_token_id": opportunity.yes_token_id,
            "no_token_id": opportunity.no_token_id,
            "requested_shares": float(opportunity.shares),
            "capital_required": float(opportunity.capital_required),
            "capital_reserved": float(opportunity.execution_capital_required),
            "expected_net_profit": float(opportunity.net_profit),
            "yes_order_id": "",
            "no_order_id": "",
            "yes_order_status": "",
            "no_order_status": "",
            "yes_placement_shares": 0.0,
            "no_placement_shares": 0.0,
            "yes_matched_shares": 0.0,
            "no_matched_shares": 0.0,
            "yes_trade_ids": [],
            "no_trade_ids": [],
            "yes_trade_fills": {},
            "no_trade_fills": {},
            "yes_fill_details": {},
            "no_fill_details": {},
            "yes_transaction_hashes": [],
            "no_transaction_hashes": [],
            "status": "PENDING",
            "rollback_status": "NOT_REQUIRED",
            "rollback_details": {},
            "pnl_quality": "UNAVAILABLE",
            "order_style": str(getattr(opportunity, "order_style", "FOK") or "FOK").upper(),
            "settlement_type": "",
            "settlement_tx_id": "",
            "settlement_tx_hash": "",
            "merge_attempts": 0,
            "redemption_attempts": 0,
            "redemption_tx_id": "",
            "created_at": now,
            "updated_at": now,
            "error": "",
        }
        self.save()
        return pair_id

    def open_exposure(self) -> float:
        """Capital reserved by all non-terminal pairs."""
        total = 0.0
        for record in self.incomplete_pairs():
            if "capital_reserved" not in record:
                raise RuntimeError(
                    f"live journal pair {record.get('pair_id', '')} has no capital reservation"
                )
            reserved = float(record["capital_reserved"])
            if not math.isfinite(reserved) or reserved < 0.0:
                raise RuntimeError(
                    f"live journal pair {record.get('pair_id', '')} has invalid capital reservation"
                )
            total += reserved
        return total

    def market_exposure(self, market_id: str) -> float:
        total = 0.0
        for record in self.incomplete_pairs():
            if "capital_reserved" not in record:
                raise RuntimeError(
                    f"live journal pair {record.get('pair_id', '')} has no capital reservation"
                )
            if str(record.get("market_id")) == str(market_id):
                reserved = float(record["capital_reserved"])
                if not math.isfinite(reserved) or reserved < 0.0:
                    raise RuntimeError(
                        f"live journal pair {record.get('pair_id', '')} has invalid capital reservation"
                    )
                total += reserved
        return total

    def integrity_issues(self) -> List[str]:
        """Return journal inconsistencies that make live state ambiguous."""
        issues = []
        seen_orders = {}
        for record in self.state["pairs"].values():
            if not record.get("condition_id"):
                issues.append(f"{record.get('pair_id', '')} has no condition id")
            try:
                requested = float(record.get("requested_shares", 0.0))
            except (TypeError, ValueError):
                requested = float("nan")
            if not math.isfinite(requested) or requested <= 0.0:
                issues.append(f"{record.get('pair_id', '')} has invalid requested shares")
            reserved = record.get("capital_reserved")
            try:
                reserved_value = float(reserved)
            except (TypeError, ValueError):
                reserved_value = float("nan")
            if not math.isfinite(reserved_value) or reserved_value < 0.0:
                issues.append(f"{record.get('pair_id', '')} has invalid capital reservation")
            for field in ("capital_required", "expected_net_profit"):
                try:
                    value = float(record.get(field, 0.0))
                except (TypeError, ValueError):
                    value = float("nan")
                if not math.isfinite(value):
                    issues.append(f"{record.get('pair_id', '')} has invalid {field}")
            for leg in ("yes", "no"):
                if not record.get(f"{leg}_token_id"):
                    issues.append(f"{record.get('pair_id', '')} has no {leg} token id")
                try:
                    matched = float(record.get(f"{leg}_matched_shares", 0.0))
                except (TypeError, ValueError):
                    matched = float("nan")
                if not math.isfinite(matched) or matched < 0.0:
                    issues.append(f"{record.get('pair_id', '')} has invalid {leg} matched shares")
                elif math.isfinite(requested) and matched > requested + 1e-8:
                    issues.append(f"{record.get('pair_id', '')} {leg} fill exceeds request")
                details = record.get(f"{leg}_fill_details", {})
                if not isinstance(details, dict):
                    issues.append(f"{record.get('pair_id', '')} has invalid {leg} fill details")
                else:
                    detail_total = 0.0
                    for trade_id, detail in details.items():
                        if not isinstance(detail, dict):
                            issues.append(f"{record.get('pair_id', '')} has invalid {leg} fill {trade_id}")
                            continue
                        try:
                            detail_shares = float(detail["shares"])
                            detail_price = float(detail["price"])
                            detail_fee = float(detail["fee_usd"])
                        except (KeyError, TypeError, ValueError):
                            issues.append(f"{record.get('pair_id', '')} has incomplete {leg} fill {trade_id}")
                            continue
                        if (not math.isfinite(detail_shares) or detail_shares < 0.0
                                or not math.isfinite(detail_price) or not (0.0 < detail_price <= 1.0)
                                or not math.isfinite(detail_fee) or detail_fee < 0.0):
                            issues.append(f"{record.get('pair_id', '')} has invalid {leg} fill {trade_id}")
                        else:
                            detail_total += detail_shares
                    if math.isfinite(requested) and detail_total > requested + 1e-8:
                        issues.append(f"{record.get('pair_id', '')} {leg} fill details exceed request")
                order_id = str(record.get(f"{leg}_order_id", ""))
                if order_id:
                    previous = seen_orders.get(order_id)
                    if previous and previous != record.get("pair_id"):
                        issues.append(f"order {order_id} belongs to multiple pairs")
                    seen_orders[order_id] = record.get("pair_id")
            if "realized_pnl" in record:
                try:
                    realized_pnl = float(record["realized_pnl"])
                except (TypeError, ValueError):
                    realized_pnl = float("nan")
                if not math.isfinite(realized_pnl):
                    issues.append(f"{record.get('pair_id', '')} has invalid realized PnL")
            if record.get("status") not in {
                "PENDING", "RESTING", "HEDGED", "RESOLVED_PENDING_REDEMPTION", "SETTLED",
                "ROLLED_BACK", "REJECTED", "UNHEDGED"
            }:
                issues.append(f"{record.get('pair_id', '')} has unknown status")
        return issues

    def _record(self, pair_id: str) -> dict:
        try:
            return self.state["pairs"][pair_id]
        except KeyError as error:
            raise KeyError(f"unknown live pair: {pair_id}") from error

    def update(self, pair_id: str, **changes) -> dict:
        record = self._record(pair_id)
        record.update(changes)
        record["updated_at"] = time.time()
        self.save()
        return record

    def set_order_id(self, pair_id: str, leg: str, order_id: str) -> None:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        self.update(pair_id, **{f"{leg}_order_id": str(order_id)})

    def set_order_status(self, pair_id: str, leg: str, status: str) -> dict:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        return self.update(pair_id, **{f"{leg}_order_status": str(status).upper()})

    def add_fill(self, pair_id: str, leg: str, shares: float,
                 trade_id: Optional[str] = None, price: Optional[float] = None,
                 fee_usd: Optional[float] = None, fee_rate_bps: Optional[float] = None,
                 tx_hash: Optional[str] = None, timestamp_ms: Optional[int] = None) -> dict:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        try:
            shares = float(shares)
        except (TypeError, ValueError) as error:
            raise ValueError("fill shares must be numeric") from error
        if not math.isfinite(shares) or shares < 0.0:
            raise ValueError("fill shares must be finite and non-negative")
        if price is not None:
            price = float(price)
            if not math.isfinite(price) or not (0.0 < price <= 1.0):
                raise ValueError("fill price must be finite and in (0, 1]")
        if fee_usd is not None:
            fee_usd = float(fee_usd)
            if not math.isfinite(fee_usd) or fee_usd < 0.0:
                raise ValueError("fill fee must be finite and non-negative")
        if fee_rate_bps is not None:
            fee_rate_bps = float(fee_rate_bps)
            if not math.isfinite(fee_rate_bps) or fee_rate_bps < 0.0:
                raise ValueError("fill fee rate must be finite and non-negative")
        record = self._record(pair_id)
        matched_key = f"{leg}_matched_shares"
        details_key = f"{leg}_fill_details"
        record.setdefault(details_key, {})
        record.setdefault(f"{leg}_transaction_hashes", [])
        # Order snapshots are cumulative.  Trade events are deduplicated by
        # trade id and only then added to the cumulative leg total.
        if trade_id:
            trade_id = str(trade_id)
            ids = record[f"{leg}_trade_ids"]
            if trade_id not in ids:
                ids.append(trade_id)
                record[f"{leg}_trade_fills"][trade_id] = max(0.0, float(shares))
            elif float(record[f"{leg}_trade_fills"].get(trade_id, 0.0)) <= 1e-12:
                record[f"{leg}_trade_fills"][trade_id] = max(0.0, float(shares))
            if price is not None or fee_usd is not None or fee_rate_bps is not None or tx_hash:
                detail = record[details_key].setdefault(trade_id, {})
                detail["shares"] = max(0.0, float(shares))
                if price is not None:
                    detail["price"] = float(price)
                if fee_usd is not None:
                    detail["fee_usd"] = max(0.0, float(fee_usd))
                if fee_rate_bps is not None:
                    detail["fee_rate_bps"] = max(0.0, float(fee_rate_bps))
                if tx_hash:
                    detail["tx_hash"] = str(tx_hash)
                    if str(tx_hash) not in record[f"{leg}_transaction_hashes"]:
                        record[f"{leg}_transaction_hashes"].append(str(tx_hash))
                if timestamp_ms is not None:
                    detail["timestamp_ms"] = int(timestamp_ms)
        else:
            record[matched_key] = max(float(record[matched_key]), max(0.0, float(shares)))
        trade_total = sum(float(value) for value in record[f"{leg}_trade_fills"].values())
        record[matched_key] = max(float(record[matched_key]), trade_total)
        record["updated_at"] = time.time()
        self.save()
        return record

    def actual_pair_cost(self, pair_id: str) -> Optional[float]:
        """Return actual two-leg cost only when every fill has economics."""
        record = self._record(pair_id)
        requested = float(record.get("requested_shares", 0.0))
        total = 0.0
        for leg in ("yes", "no"):
            details = record.get(f"{leg}_fill_details", {})
            leg_shares = 0.0
            leg_cost = 0.0
            for detail in details.values():
                if not isinstance(detail, dict) or "shares" not in detail or "price" not in detail or "fee_usd" not in detail:
                    return None
                shares = float(detail["shares"])
                price = float(detail["price"])
                fee_usd = float(detail["fee_usd"])
                if (not math.isfinite(shares) or shares < 0.0
                        or not math.isfinite(price) or not (0.0 < price <= 1.0)
                        or not math.isfinite(fee_usd) or fee_usd < 0.0):
                    raise RuntimeError(f"pair {pair_id} has invalid fill economics")
                leg_shares += shares
                leg_cost += shares * price + fee_usd
            if leg_shares + 1e-8 < requested:
                return None
            total += leg_cost
        return total

    def add_transaction_hashes(self, pair_id: str, leg: str,
                               hashes: Iterable[str]) -> dict:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        record = self._record(pair_id)
        key = f"{leg}_transaction_hashes"
        values = record.setdefault(key, [])
        for value in hashes or ():
            normalized = str(value)
            if normalized and normalized not in values:
                values.append(normalized)
        record["updated_at"] = time.time()
        self.save()
        return record

    def add_trade_ids(self, pair_id: str, leg: str, trade_ids: Iterable[str]) -> dict:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        record = self._record(pair_id)
        ids = record[f"{leg}_trade_ids"]
        fills = record[f"{leg}_trade_fills"]
        for trade_id in trade_ids or ():
            normalized = str(trade_id)
            if normalized not in ids:
                ids.append(normalized)
                fills.setdefault(normalized, 0.0)
        record["updated_at"] = time.time()
        self.save()
        return record

    def set_matched(self, pair_id: str, leg: str, shares: float) -> dict:
        """Set a leg from an authoritative cumulative order snapshot."""
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        shares = float(shares)
        if not math.isfinite(shares) or shares < 0.0:
            raise ValueError("matched shares must be finite and non-negative")
        record = self._record(pair_id)
        trade_total = sum(float(value) for value in record[f"{leg}_trade_fills"].values())
        record[f"{leg}_matched_shares"] = max(0.0, float(shares), trade_total)
        record["updated_at"] = time.time()
        self.save()
        return record

    def set_placement_fill(self, pair_id: str, leg: str, shares: float) -> dict:
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        shares = float(shares)
        if not math.isfinite(shares) or shares < 0.0:
            raise ValueError("placement shares must be finite and non-negative")
        return self.update(pair_id, **{f"{leg}_placement_shares": shares})

    def record_rollback(self, pair_id: str, leg: str, response,
                        entry_price: Optional[float] = None) -> dict:
        """Persist a FAK unwind, including proceeds and conservative PnL."""
        if leg not in {"yes", "no"}:
            raise ValueError("leg must be yes or no")
        record = self._record(pair_id)
        shares = _filled_shares(response, side="SELL")
        proceeds = _numeric_value(response, "taking_amount", "takingAmount", "proceeds", "proceeds_usd")
        fee_usd = _numeric_value(response, "fee_usd", "fee", "fee_amount")
        detail = {
            "order_id": str(_response_value(response, "order_id", "orderID", default="") or ""),
            "trade_ids": [str(value) for value in (_response_value(response, "trade_ids", "tradeIDs", default=()) or ())],
            "transaction_hashes": [
                str(value) for value in
                (_response_value(response, "transactions_hashes", "transactionsHashes", default=()) or ())
            ],
            "shares": shares if shares is not None and math.isfinite(shares) and shares >= 0.0 else 0.0,
            "proceeds_usd": proceeds if proceeds is not None and math.isfinite(proceeds) and proceeds >= 0.0 else None,
            "fee_usd": fee_usd if fee_usd is not None and math.isfinite(fee_usd) and fee_usd >= 0.0 else None,
            "entry_price": entry_price if entry_price is not None and math.isfinite(entry_price) else None,
            "recorded_at": time.time(),
        }
        if detail["proceeds_usd"] is not None and detail["entry_price"] is not None:
            detail["pnl"] = detail["proceeds_usd"] - (detail["fee_usd"] or 0.0) - detail["shares"] * detail["entry_price"]
        record.setdefault("rollback_details", {})[leg] = detail
        pnls = [
            float(value["pnl"])
            for value in record["rollback_details"].values()
            if isinstance(value, dict) and isinstance(value.get("pnl"), (int, float))
            and math.isfinite(float(value["pnl"]))
        ]
        if pnls:
            record["realized_pnl"] = sum(pnls)
            record["pnl_quality"] = "ESTIMATED"
        record["updated_at"] = time.time()
        self.save()
        return record

    def set_status(self, pair_id: str, status: str, error: str = "") -> dict:
        return self.update(pair_id, status=status, error=error)

    def incomplete_pairs(self) -> List[dict]:
        return [record for record in self.state["pairs"].values()
                if record.get("status") not in self.TERMINAL_STATUSES]

    def summary(self) -> dict:
        """Return a read-only operational summary for monitoring and audits."""
        by_status: Dict[str, int] = {}
        realized_pnl = 0.0
        actual_pnl_count = 0
        submitted_settlements = []
        unhedged_pairs = []
        for record in self.state["pairs"].values():
            status = str(record.get("status", "UNKNOWN"))
            by_status[status] = by_status.get(status, 0) + 1
            pnl = record.get("realized_pnl")
            if pnl is not None and math.isfinite(float(pnl)):
                realized_pnl += float(pnl)
            if record.get("pnl_quality") == "ACTUAL":
                actual_pnl_count += 1
            if record.get("settlement_type") in {"MERGE_SUBMITTED", "REDEEM_SUBMITTED"}:
                submitted_settlements.append(str(record.get("pair_id", "")))
            if status == "UNHEDGED":
                unhedged_pairs.append(str(record.get("pair_id", "")))
        return {
            "pairs": len(self.state["pairs"]),
            "by_status": dict(sorted(by_status.items())),
            "open_exposure": self.open_exposure(),
            "realized_pnl": realized_pnl,
            "actual_pnl_pairs": actual_pnl_count,
            "submitted_settlements": submitted_settlements,
            "unhedged_pairs": unhedged_pairs,
        }

    def pair_for_order(self, order_id: str) -> Optional[dict]:
        order_id = str(order_id)
        for record in self.state["pairs"].values():
            if order_id in {record.get("yes_order_id"), record.get("no_order_id")}:
                return record
        return None

    def mark_resolved(self, market_id: str, condition_id: str, winning_outcome: str) -> int:
        """Record market resolution without claiming collateral was redeemed."""
        winning_outcome = str(winning_outcome or "").strip()
        if not winning_outcome:
            raise ValueError("winning_outcome is required")
        count = 0
        for record in self.state["pairs"].values():
            matches_market = str(record.get("market_id")) in {str(market_id), str(condition_id)}
            matches_condition = str(record.get("condition_id")) in {str(market_id), str(condition_id)}
            if not (matches_market or matches_condition):
                continue
            valid_outcomes = {
                str(record.get("yes_token_id")), str(record.get("no_token_id")),
                "yes", "no", "Yes", "No", "YES", "NO",
            }
            if winning_outcome not in valid_outcomes:
                raise ValueError(f"unknown winning outcome for pair {record['pair_id']}")
            if record.get("status") == "RESOLVED_PENDING_REDEMPTION":
                if str(record.get("winning_outcome")) != winning_outcome:
                    raise RuntimeError(f"conflicting resolution for pair {record['pair_id']}")
                continue
            if record.get("status") == "HEDGED":
                record["status"] = "RESOLVED_PENDING_REDEMPTION"
                record["settlement_type"] = "MARKET_RESOLUTION"
                record["winning_outcome"] = winning_outcome
                record["resolved_at"] = time.time()
                record["updated_at"] = time.time()
                count += 1
        if count:
            self.save()
        return count

    def mark_redeemed(self, pair_id: str, transaction_hash: str,
                      redeemed_shares: Optional[float] = None) -> dict:
        """Mark collateral redemption confirmed and realize the pair PnL."""
        record = self._record(pair_id)
        if record.get("status") == "SETTLED":
            return record
        if record.get("status") != "RESOLVED_PENDING_REDEMPTION":
            raise RuntimeError(f"pair {pair_id} is not pending redemption")
        transaction_hash = str(transaction_hash or "").strip()
        if not transaction_hash:
            raise ValueError("redemption transaction hash is required")
        requested = float(record.get("requested_shares", 0.0))
        payout = requested if redeemed_shares is None else float(redeemed_shares)
        if payout <= 0.0 or payout > requested + 1e-8:
            raise ValueError("invalid redeemed share amount")
        actual_cost = self.actual_pair_cost(pair_id)
        if actual_cost is None:
            actual_cost = float(record.get("capital_required", 0.0))
            record["pnl_quality"] = "ESTIMATED"
        else:
            record["pnl_quality"] = "ACTUAL"
        record["status"] = "SETTLED"
        record["settlement_type"] = "MARKET_REDEMPTION"
        record["settlement_tx_hash"] = transaction_hash
        record["redeemed_shares"] = payout
        record["settled_at"] = time.time()
        record["realized_pnl"] = payout - actual_cost
        record["updated_at"] = time.time()
        self.save()
        return record

    def mark_merged(self, pair_id: str, transaction_hash: str) -> dict:
        """Mark a complete Yes/No set as redeemed back to collateral."""
        record = self._record(pair_id)
        if record.get("status") == "SETTLED":
            return record
        if record.get("status") != "HEDGED":
            raise RuntimeError(f"pair {pair_id} is not fully hedged")
        actual_cost = self.actual_pair_cost(pair_id)
        if actual_cost is None:
            actual_cost = float(record.get("capital_required", 0.0))
            record["pnl_quality"] = "ESTIMATED"
        else:
            record["pnl_quality"] = "ACTUAL"
        record["status"] = "SETTLED"
        record["settlement_type"] = "COMPLETE_SET_MERGE"
        record["settlement_tx_hash"] = str(transaction_hash)
        record["settled_at"] = time.time()
        record["realized_pnl"] = float(record.get("requested_shares", 0.0)) - actual_cost
        record["updated_at"] = time.time()
        self.save()
        return record


class LiveDirectionalJournal:
    """Atomic journal for single-leg BUY/SELL fills used by sports/macro/crypto.

    Pair FOK execution stays on LiveOrderJournal. Directional inventory is
    tracked here so account-wide recon can distinguish owned tokens from
    unexplained leftovers, and so risk limits see both books of exposure.
    """

    TERMINAL_STATUSES = {"REJECTED", "CANCELLED", "FLATTENED"}
    OPEN_STATUSES = {"PENDING", "OPEN", "PARTIAL", "UNKNOWN", "FILLED"}

    def __init__(self, path: str):
        self.path = path
        self.state = {"trades": {}, "events": []}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("trades"), dict):
                raise ValueError("missing trades")
            self.state.update(loaded)
            self.state.setdefault("events", [])
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Directional journal is unreadable: {self.path}") from error

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".live-directional-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def create(self, intent: "DirectionalIntent") -> str:
        trade_id = f"{intent.source or 'dir'}:{intent.market_id}:{time.time_ns()}"
        now = time.time()
        notional = float(intent.shares) * float(intent.limit_price)
        reserved = 0.0 if intent.side.upper() == "SELL" else max(0.0, notional + float(intent.fee_cap))
        self.state["trades"][trade_id] = {
            "trade_id": trade_id,
            "market_id": intent.market_id,
            "condition_id": intent.condition_id,
            "token_id": intent.token_id,
            "side": intent.side.upper(),
            "order_type": (intent.order_type or "FOK").upper(),
            "source": intent.source,
            "event_id": intent.event_id,
            "reason": intent.reason,
            "requested_shares": float(intent.shares),
            "limit_price": float(intent.limit_price),
            "capital_reserved": reserved,
            "order_id": "",
            "order_status": "",
            "matched_shares": 0.0,
            "inventory_shares": 0.0,
            "fill_details": {},
            "status": "PENDING",
            "created_at": now,
            "updated_at": now,
            "error": "",
            "extra": dict(intent.extra or {}),
        }
        self.save()
        return trade_id

    def _record(self, trade_id: str) -> dict:
        try:
            return self.state["trades"][trade_id]
        except KeyError as error:
            raise KeyError(f"unknown directional trade: {trade_id}") from error

    def update(self, trade_id: str, **changes) -> dict:
        record = self._record(trade_id)
        record.update(changes)
        record["updated_at"] = time.time()
        self.save()
        return record

    def incomplete_trades(self) -> List[dict]:
        open_trades = []
        for record in self.state["trades"].values():
            status = str(record.get("status", ""))
            if status in self.TERMINAL_STATUSES:
                continue
            if status == "FILLED" and str(record.get("side") or "").upper() == "SELL":
                continue
            open_trades.append(record)
        return open_trades

    def open_exposure(self) -> float:
        total = 0.0
        for record in self.incomplete_trades():
            reserved = float(record.get("capital_reserved", 0.0))
            if not math.isfinite(reserved) or reserved < 0.0:
                raise RuntimeError(
                    f"directional trade {record.get('trade_id', '')} has invalid capital reservation"
                )
            total += reserved
        return total

    def market_exposure(self, market_id: str) -> float:
        total = 0.0
        for record in self.incomplete_trades():
            if str(record.get("market_id")) != str(market_id):
                continue
            reserved = float(record.get("capital_reserved", 0.0))
            if not math.isfinite(reserved) or reserved < 0.0:
                raise RuntimeError(
                    f"directional trade {record.get('trade_id', '')} has invalid capital reservation"
                )
            total += reserved
        return total

    def known_order_ids(self) -> set:
        known = set()
        for record in self.state["trades"].values():
            order_id = str(record.get("order_id") or "")
            if order_id:
                known.add(order_id)
        return known

    def known_inventory_token_ids(self) -> set:
        known = set()
        for token_id, shares in self.inventory_by_token().items():
            if shares > 1e-8:
                known.add(token_id)
        return known

    def inventory_by_token(self) -> Dict[str, float]:
        inventory: Dict[str, float] = {}
        for record in self.state["trades"].values():
            token_id = str(record.get("token_id") or "")
            if not token_id:
                continue
            matched = float(record.get("matched_shares") or 0.0)
            if matched <= 1e-8:
                continue
            side = str(record.get("side") or "").upper()
            if side == "BUY":
                inventory[token_id] = inventory.get(token_id, 0.0) + matched
            elif side == "SELL":
                inventory[token_id] = inventory.get(token_id, 0.0) - matched
        return {token: shares for token, shares in inventory.items() if abs(shares) > 1e-8}

    def record_for_order(self, order_id: str) -> Optional[dict]:
        order_id = str(order_id or "")
        if not order_id:
            return None
        for record in self.state["trades"].values():
            if str(record.get("order_id") or "") == order_id:
                return record
        return None

    def set_order_id(self, trade_id: str, order_id: str) -> dict:
        return self.update(trade_id, order_id=str(order_id))

    def add_fill(self, trade_id: str, shares: float, fill_id: Optional[str] = None,
                 price: Optional[float] = None, fee_usd: Optional[float] = None) -> dict:
        record = self._record(trade_id)
        shares = float(shares)
        if not math.isfinite(shares) or shares < 0.0:
            raise ValueError("fill shares must be finite and non-negative")
        fill_id = str(fill_id or f"fill-{len(record.get('fill_details') or {}) + 1}")
        details = record.setdefault("fill_details", {})
        if fill_id not in details:
            record["matched_shares"] = float(record.get("matched_shares") or 0.0) + shares
            details[fill_id] = {
                "shares": shares,
                "price": None if price is None else float(price),
                "fee_usd": 0.0 if fee_usd is None else float(fee_usd),
            }
        side = str(record.get("side") or "").upper()
        if side == "BUY":
            record["inventory_shares"] = float(record.get("matched_shares") or 0.0)
        requested = float(record.get("requested_shares") or 0.0)
        matched = float(record.get("matched_shares") or 0.0)
        if matched + 1e-8 >= requested and requested > 0:
            record["status"] = "FILLED"
        elif matched > 1e-8:
            record["status"] = "PARTIAL"
        record["updated_at"] = time.time()
        self.save()
        return record

    def integrity_issues(self) -> List[str]:
        issues = []
        seen_orders = {}
        for record in self.state["trades"].values():
            trade_id = str(record.get("trade_id") or "")
            try:
                reserved = float(record.get("capital_reserved", 0.0))
            except (TypeError, ValueError):
                reserved = float("nan")
            if not math.isfinite(reserved) or reserved < 0.0:
                issues.append(f"{trade_id} has invalid capital reservation")
            try:
                shares = float(record.get("requested_shares", 0.0))
            except (TypeError, ValueError):
                shares = float("nan")
            if not math.isfinite(shares) or shares <= 0.0:
                issues.append(f"{trade_id} has invalid requested shares")
            if str(record.get("side") or "").upper() not in {"BUY", "SELL"}:
                issues.append(f"{trade_id} has invalid side")
            if not record.get("token_id"):
                issues.append(f"{trade_id} has no token id")
            order_id = str(record.get("order_id") or "")
            if order_id:
                previous = seen_orders.get(order_id)
                if previous and previous != trade_id:
                    issues.append(f"order {order_id} belongs to multiple directional trades")
                seen_orders[order_id] = trade_id
        return issues

    def record_halt_flatten(self, token_id: str, sold: float, remaining: float,
                            order_id: str = "", market_id: str = "") -> dict:
        """Record a kill-switch FAK sale of unhedged directional inventory."""
        sold = max(0.0, float(sold))
        remaining = max(0.0, float(remaining))
        trade_id = f"halt:{token_id}:{time.time_ns()}"
        now = time.time()
        status = "FLATTENED" if remaining <= 1e-8 else "PARTIAL"
        record = {
            "trade_id": trade_id,
            "market_id": str(market_id or ""),
            "condition_id": "",
            "token_id": str(token_id),
            "side": "SELL",
            "order_type": "FAK",
            "source": "kill-switch",
            "event_id": "halt",
            "reason": "FAK flatten unhedged directional inventory on halt",
            "requested_shares": sold,
            "limit_price": 0.0,
            "capital_reserved": 0.0,
            "order_id": str(order_id or ""),
            "order_status": status,
            "matched_shares": sold,
            "inventory_shares": 0.0,
            "fill_details": {
                "halt-fak": {"shares": sold, "price": None, "fee_usd": 0.0},
            } if sold > 1e-8 else {},
            "status": status,
            "created_at": now,
            "updated_at": now,
            "error": "" if remaining <= 1e-8 else f"{remaining} shares remain after FAK",
            "extra": {"remaining_shares": remaining},
        }
        self.state["trades"][trade_id] = record
        self.save()
        return record

    def summary(self) -> dict:
        return {
            "trades": len(self.state["trades"]),
            "open_trades": len(self.incomplete_trades()),
            "open_exposure": self.open_exposure(),
            "inventory": self.inventory_by_token(),
        }


@dataclass
class DirectionalIntent:
    """One-leg BUY or SELL that must not be jammed into the Yes+No pair FOK path."""

    token_id: str
    side: str
    shares: float
    limit_price: float
    market_id: str = ""
    condition_id: str = ""
    order_type: str = "FOK"
    source: str = ""
    event_id: str = ""
    reason: str = ""
    fee_cap: float = 0.0
    extra: dict = None

    def __post_init__(self):
        self.side = str(self.side or "").upper()
        self.order_type = str(self.order_type or "FOK").upper()
        if self.extra is None:
            self.extra = {}

    @property
    def notional(self) -> float:
        return float(self.shares) * float(self.limit_price)


@dataclass
class DirectionalResult:
    trade_id: str
    order_id: str
    shares: float
    status: str
    side: str


def intent_from_best_ask(market: BinaryMarket, token_id: str, book: OrderBook,
                         max_order_usd: float, source: str, event_id: str = "",
                         reason: str = "", order_type: str = "FOK",
                         max_shares: Optional[float] = None) -> Optional[DirectionalIntent]:
    """Size a BUY against the live best ask. Returns None if the book cannot fill."""
    if not book or not book.synced:
        return None
    touch = book.best_ask()
    if not touch:
        return None
    price, depth = touch
    if price <= 0.0 or depth <= 0.0:
        return None
    budget_shares = float(max_order_usd) / price if price > 0 else 0.0
    shares = min(depth, budget_shares)
    if max_shares is not None:
        shares = min(shares, float(max_shares))
    if market.min_order_size > 0.0:
        if shares + 1e-12 < market.min_order_size:
            return None
    if shares <= 1e-8:
        return None
    return DirectionalIntent(
        token_id=str(token_id),
        side="BUY",
        shares=shares,
        limit_price=price,
        market_id=market.market_id,
        condition_id=market.condition_id,
        order_type=order_type,
        source=source,
        event_id=event_id,
        reason=reason,
    )


def intent_from_inventory_bid(market: BinaryMarket, token_id: str, book: OrderBook,
                              shares: float, source: str, event_id: str = "",
                              reason: str = "", order_type: str = "FOK") -> Optional[DirectionalIntent]:
    """Size a SELL of owned inventory against the live best bid."""
    if not book or not book.synced or shares <= 1e-8:
        return None
    touch = book.best_bid()
    if not touch:
        return None
    price, depth = touch
    sell_shares = min(float(shares), depth)
    if sell_shares <= 1e-8:
        return None
    return DirectionalIntent(
        token_id=str(token_id),
        side="SELL",
        shares=sell_shares,
        limit_price=price,
        market_id=market.market_id,
        condition_id=market.condition_id,
        order_type=order_type,
        source=source,
        event_id=event_id,
        reason=reason,
    )


class PaperDirectionalExecutor:
    """Immediate paper fill for directional research/execution flags."""

    def __init__(self, journal: LiveDirectionalJournal):
        self.journal = journal

    def execute(self, intent: DirectionalIntent) -> DirectionalResult:
        if intent.shares <= 0.0 or intent.limit_price <= 0.0:
            raise ValueError("cannot execute an empty directional intent")
        if intent.side not in {"BUY", "SELL"}:
            raise ValueError("directional side must be BUY or SELL")
        trade_id = self.journal.create(intent)
        self.journal.set_order_id(trade_id, f"paper-{trade_id}")
        record = self.journal.add_fill(trade_id, intent.shares, fill_id=f"paper-{trade_id}",
                                       price=intent.limit_price, fee_usd=0.0)
        return DirectionalResult(
            trade_id=trade_id,
            order_id=str(record.get("order_id") or ""),
            shares=float(record.get("matched_shares") or 0.0),
            status=str(record.get("status") or ""),
            side=intent.side,
        )


class DirectionalExecutor:
    """Single-leg official CLOB execution, journaled separately from pair FOK."""

    def __init__(self, client_executor: "OfficialFOKExecutor",
                 journal: LiveDirectionalJournal,
                 risk: Optional["LiveRiskController"] = None):
        self.client_executor = client_executor
        self.journal = journal
        self.risk = risk

    async def execute(self, intent: DirectionalIntent) -> DirectionalResult:
        if intent.shares <= 0.0 or intent.limit_price <= 0.0:
            raise ValueError("cannot execute an empty directional intent")
        if intent.side not in {"BUY", "SELL"}:
            raise ValueError("directional side must be BUY or SELL")
        if intent.order_type not in {"FOK", "FAK"}:
            raise ValueError("directional order type must be FOK or FAK")
        if self.risk:
            self.risk.check_directional(intent.notional + float(intent.fee_cap), intent.market_id)
        if intent.side == "BUY":
            await self.client_executor.preflight(required_usd=intent.notional + float(intent.fee_cap))
            order_kwargs = {
                "token_id": intent.token_id,
                "side": "BUY",
                "amount": f"{intent.notional:.6f}",
                "max_spend": f"{intent.notional + float(intent.fee_cap):.6f}",
                "max_price": f"{intent.limit_price:.6f}",
                "order_type": intent.order_type,
            }
        else:
            await self.client_executor.preflight(required_usd=0.0)
            order_kwargs = {
                "token_id": intent.token_id,
                "side": "SELL",
                "shares": f"{intent.shares:.6f}",
                "min_price": f"{intent.limit_price:.6f}",
                "order_type": intent.order_type,
            }
        trade_id = self.journal.create(intent)
        try:
            response = await self.client_executor._call("place_market_order", **order_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.journal.update(trade_id, status="UNKNOWN", error=f"order outcome is unknown: {error}")
            raise UnhedgedPairError(
                "directional order outcome is unknown; reconcile before placing new orders"
            ) from error
        filled = _filled_shares(response, side=intent.side)
        order_id = str(_response_value(response, "order_id", "orderID", default="") or "")
        if order_id:
            self.journal.set_order_id(trade_id, order_id)
        if not _response_ok(response):
            self.journal.update(
                trade_id, status="REJECTED",
                error=str(_response_value(response, "message", "errorMsg", default="directional order rejected")),
            )
            raise RuntimeError(
                f"directional {intent.side} {intent.order_type} rejected: "
                f"{_response_value(response, 'message', 'errorMsg', default='unknown error')}"
            )
        if filled is None or filled <= 1e-8:
            self.journal.update(trade_id, status="REJECTED", error="response did not confirm a fill")
            raise RuntimeError("directional response did not confirm a fill")
        if intent.order_type == "FOK" and filled + 1e-8 < intent.shares:
            self.journal.update(trade_id, status="REJECTED", error="FOK did not fill in full")
            raise RuntimeError("directional FOK did not fill in full")
        economics = _trade_fill_economics(response)
        record = self.journal.add_fill(
            trade_id, filled, fill_id=order_id or "placement",
            price=economics["price"] if economics["price"] is not None else intent.limit_price,
            fee_usd=economics["fee_usd"],
        )
        return DirectionalResult(
            trade_id=trade_id,
            order_id=order_id,
            shares=float(record.get("matched_shares") or 0.0),
            status=str(record.get("status") or ""),
            side=intent.side,
        )


class LiveRiskController:
    """Persistent fail-closed gate for live order admission.

    Exposure is derived from the live journal, so a process restart cannot
    reset the position budget. A manual kill-switch file or environment flag
    permanently halts admission until the operator clears the persisted halt
    state and restarts with an explicit decision.
    """

    def __init__(self, journal: LiveOrderJournal, equity_usd: float,
                 state_path: str = "live-risk.json",
                 kill_switch_path: str = "live-kill-switch",
                 max_total_exposure_fraction: float = 0.25,
                 max_market_exposure_fraction: float = 0.05,
                 max_open_pairs: int = 10,
                 max_daily_loss_usd: float = 0.0,
                 extra_journals: Optional[Sequence[LiveDirectionalJournal]] = None,
                 max_open_directional: int = 5,
                 negrisk_journal=None,
                 max_open_negrisk: int = 2):
        self.journal = journal
        self.extra_journals = list(extra_journals or [])
        self.negrisk_journal = negrisk_journal
        self.max_open_directional = max(0, int(max_open_directional))
        self.max_open_negrisk = max(0, int(max_open_negrisk))
        self.equity_usd = float(equity_usd)
        self.state_path = state_path
        self.kill_switch_path = kill_switch_path
        total_fraction = float(max_total_exposure_fraction)
        market_fraction = float(max_market_exposure_fraction)
        daily_loss = float(max_daily_loss_usd)
        self.max_total_exposure_fraction = (
            max(0.0, total_fraction) if math.isfinite(total_fraction) else float("nan")
        )
        self.max_market_exposure_fraction = (
            max(0.0, market_fraction) if math.isfinite(market_fraction) else float("nan")
        )
        self.max_open_pairs = max(1, int(max_open_pairs))
        self.max_daily_loss_usd = max(0.0, daily_loss) if math.isfinite(daily_loss) else float("nan")
        self.state = {
            "halted": False,
            "halt_reason": "",
            "daily_loss_usd": 0.0,
            "utc_day": time.strftime("%Y-%m-%d", time.gmtime()),
            "last_balance_usd": None,
            "last_allowance_usd": None,
            "last_account_snapshot_at": None,
            "flatten_required": False,
            "flatten_completed": False,
            "flatten_result": {},
        }
        self.load()
        self._roll_day()

    def load(self) -> None:
        if not self.state_path:
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("risk state must be an object")
            self.state.update(loaded)
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Risk state is unreadable: {self.state_path}") from error

    def save(self) -> None:
        if not self.state_path:
            return
        directory = os.path.dirname(os.path.abspath(self.state_path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".live-risk-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.state_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self.state.get("utc_day") != today:
            self.state["utc_day"] = today
            self.state["daily_loss_usd"] = 0.0
            self.save()

    def halt(self, reason: str) -> None:
        self.state["halted"] = True
        self.state["halt_reason"] = str(reason)
        if not self.state.get("flatten_completed"):
            self.state["flatten_required"] = True
        self.save()

    def record_realized_pnl(self, pnl_usd: float) -> None:
        self._roll_day()
        if pnl_usd < 0.0:
            self.state["daily_loss_usd"] = float(self.state.get("daily_loss_usd", 0.0)) + abs(float(pnl_usd))
        self.save()

    def record_account_snapshot(self, snapshot: dict) -> None:
        """Persist the latest collateral observations for audit and alarms."""
        self.state["last_balance_usd"] = float(snapshot["balance"])
        self.state["last_allowance_usd"] = float(snapshot["allowance"])
        self.state["last_account_snapshot_at"] = time.time()
        self.save()

    def extra_exposure(self) -> float:
        total = sum(journal.open_exposure() for journal in self.extra_journals)
        if self.negrisk_journal is not None:
            total += self.negrisk_journal.open_exposure()
        return total

    def extra_market_exposure(self, market_id: str) -> float:
        total = sum(journal.market_exposure(market_id) for journal in self.extra_journals)
        if self.negrisk_journal is not None:
            total += self.negrisk_journal.market_exposure(market_id)
        return total

    def extra_open_count(self) -> int:
        return sum(len(journal.incomplete_trades()) for journal in self.extra_journals)

    def check_startup(self) -> None:
        self._roll_day()
        issues = list(self.journal.integrity_issues())
        for extra in self.extra_journals:
            issues.extend(extra.integrity_issues())
        if self.negrisk_journal is not None:
            issues.extend(self.negrisk_journal.integrity_issues())
        if issues:
            self.halt("live journal integrity failure: " + "; ".join(issues))
            raise RiskHaltError(self.state["halt_reason"])
        unhedged = [
            str(record.get("pair_id", ""))
            for record in self.journal.state["pairs"].values()
            if record.get("status") == "UNHEDGED"
        ]
        if unhedged:
            self.halt("unhedged live pair requires manual reconciliation: " + ", ".join(unhedged))
            raise RiskHaltError(self.state["halt_reason"])
        if self.negrisk_journal is not None:
            stable = getattr(self.negrisk_journal, "STABLE_OPEN", {"ASSEMBLED"})
            unfinished = [
                str(record.get("basket_id", ""))
                for record in self.negrisk_journal.incomplete_baskets()
                if record.get("status") not in stable
            ]
            if unfinished:
                self.halt(
                    "unfinished NegRisk basket requires manual reconciliation: "
                    + ", ".join(unfinished)
                )
                raise RiskHaltError(self.state["halt_reason"])
        today = self.state["utc_day"]
        journal_loss = sum(
            max(0.0, -float(record.get("realized_pnl", 0.0)))
            for record in self.journal.state["pairs"].values()
            if record.get("status") in {"SETTLED", "ROLLED_BACK"}
            and time.strftime(
                "%Y-%m-%d",
                time.gmtime(float(record.get("settled_at", record.get("updated_at", 0.0))))
            ) == today
        )
        if journal_loss > float(self.state.get("daily_loss_usd", 0.0)):
            self.state["daily_loss_usd"] = journal_loss
            self.save()
        if self.state.get("halted"):
            raise RiskHaltError(self.state.get("halt_reason") or "persisted live risk halt")
        if os.getenv("POLYMARKET_KILL_SWITCH", "").strip() == "1":
            self.halt("POLYMARKET_KILL_SWITCH=1")
            raise RiskHaltError(self.state["halt_reason"])
        if self.kill_switch_path and os.path.exists(self.kill_switch_path):
            self.halt(f"kill-switch file exists: {self.kill_switch_path}")
            raise RiskHaltError(self.state["halt_reason"])
        if not math.isfinite(self.equity_usd) or self.equity_usd <= 0.0:
            self.halt("live risk equity is not positive")
            raise RiskHaltError(self.state["halt_reason"])
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.max_total_exposure_fraction,
                self.max_market_exposure_fraction,
                self.max_daily_loss_usd,
            )
        ):
            self.halt("live risk configuration is invalid")
            raise RiskHaltError(self.state["halt_reason"])
        if (self.max_total_exposure_fraction > 1.0
                or self.max_market_exposure_fraction > 1.0):
            self.halt("live exposure fractions cannot exceed 1.0")
            raise RiskHaltError(self.state["halt_reason"])
        if self.max_daily_loss_usd > 0.0 and float(self.state.get("daily_loss_usd", 0.0)) >= self.max_daily_loss_usd:
            self.halt("daily loss limit reached")
            raise RiskHaltError(self.state["halt_reason"])

    def poll_kill_switch(self) -> bool:
        """Halt if the operator file or env flag appears after startup."""
        if self.state.get("halted"):
            return True
        if os.getenv("POLYMARKET_KILL_SWITCH", "").strip() == "1":
            self.halt("POLYMARKET_KILL_SWITCH=1")
            return True
        if self.kill_switch_path and os.path.exists(self.kill_switch_path):
            self.halt(f"kill-switch file exists: {self.kill_switch_path}")
            return True
        return False

    def check(self, opportunity: ArbitrageOpportunity) -> None:
        self.check_startup()
        records = self.journal.incomplete_pairs()
        required = float(opportunity.execution_capital_required)
        total = self.journal.open_exposure() + self.extra_exposure()
        market = self.journal.market_exposure(opportunity.market_id) + self.extra_market_exposure(opportunity.market_id)
        if len(records) >= self.max_open_pairs:
            raise RiskHaltError("maximum number of open live pairs reached")
        if total + required > self.equity_usd * self.max_total_exposure_fraction + 1e-9:
            raise RiskHaltError("live total exposure limit")
        if market + required > self.equity_usd * self.max_market_exposure_fraction + 1e-9:
            raise RiskHaltError("live market exposure limit")

    def check_directional(self, notional: float, market_id: str = "") -> None:
        """Admit a single-leg trade against the same live exposure budget."""
        self.check_startup()
        required = float(notional)
        if not math.isfinite(required) or required < 0.0:
            raise RiskHaltError("directional notional is invalid")
        if self.max_open_directional <= 0:
            raise RiskHaltError("directional execution is disabled by risk limits")
        if self.extra_open_count() >= self.max_open_directional:
            raise RiskHaltError("maximum number of open directional trades reached")
        total = self.journal.open_exposure() + self.extra_exposure()
        if total + required > self.equity_usd * self.max_total_exposure_fraction + 1e-9:
            raise RiskHaltError("live total exposure limit")
        if market_id:
            market = self.journal.market_exposure(market_id) + self.extra_market_exposure(market_id)
            if market + required > self.equity_usd * self.max_market_exposure_fraction + 1e-9:
                raise RiskHaltError("live market exposure limit")

    def check_negrisk(self, opportunity) -> None:
        """Admit an n-leg NegRisk basket against the shared live exposure budget."""
        self.check_startup()
        if self.negrisk_journal is None:
            raise RiskHaltError("NegRisk journal is not configured")
        if self.max_open_negrisk <= 0:
            raise RiskHaltError("NegRisk execution is disabled by risk limits")
        required = float(opportunity.execution_capital_required)
        if not math.isfinite(required) or required < 0.0:
            raise RiskHaltError("NegRisk capital reservation is invalid")
        open_baskets = self.negrisk_journal.incomplete_baskets()
        if len(open_baskets) >= self.max_open_negrisk:
            raise RiskHaltError("maximum number of open NegRisk baskets reached")
        total = self.journal.open_exposure() + self.extra_exposure()
        if total + required > self.equity_usd * self.max_total_exposure_fraction + 1e-9:
            raise RiskHaltError("live total exposure limit")
        market = (
            self.journal.market_exposure(opportunity.market_id)
            + self.extra_market_exposure(opportunity.market_id)
        )
        if market + required > self.equity_usd * self.max_market_exposure_fraction + 1e-9:
            raise RiskHaltError("live market exposure limit")


@dataclass
class LivePairResult:
    pair_id: str
    yes_order_id: str
    no_order_id: str
    shares: float
    status: str = "HEDGED_PENDING_CONFIRMATION"
    rolled_back: bool = False


def _response_value(response, *names, default=None):
    for name in names:
        if isinstance(response, dict) and name in response:
            return response[name]
        if hasattr(response, name):
            return getattr(response, name)
    return default


def _event_name(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower().rsplit(".", 1)[-1]


def _response_ok(response) -> bool:
    value = _response_value(response, "ok", "success", default=False)
    return bool(value)


def _response_status(response) -> Optional[int]:
    value = _response_value(response, "status_code", "statusCode", "http_status", "status", default=None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _numeric_value(response, *names) -> Optional[float]:
    value = _response_value(response, *names, default=None)
    if isinstance(value, dict):
        value = _response_value(value, "value", "amount", "total", default=None)
    if isinstance(value, str) and value.strip().lower() in {"unlimited", "infinite", "inf"}:
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filled_shares(response, side: Optional[str] = None) -> Optional[float]:
    names = ("making_amount", "makingAmount") if str(side or "").upper() == "SELL" else ()
    value = _response_value(response, *names, "taking_amount", "takingAmount", "size_matched", "sizeMatched")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trade_fill_economics(response) -> dict:
    """Extract actual fill economics from SDK trade/user-event fields."""
    price = _numeric_value(response, "price")
    fee_usd = _numeric_value(response, "fee_usd", "fee", "fee_amount")
    fee_rate_bps = _numeric_value(response, "fee_rate_bps", "feeRateBps")
    shares = _numeric_value(response, "size", "matched_amount", "matchedAmount")
    if fee_usd is None and shares is not None and price is not None and fee_rate_bps is not None:
        fee_usd = shares * price * (1.0 - price) * fee_rate_bps / 10_000.0
    timestamp = _response_value(response, "timestamp", "matched_at", "match_time", default=None)
    if hasattr(timestamp, "timestamp"):
        timestamp_ms = int(float(timestamp.timestamp()) * 1000)
    else:
        try:
            timestamp_ms = int(float(timestamp)) if timestamp is not None else None
            if timestamp_ms is not None and timestamp_ms < 10_000_000_000:
                timestamp_ms *= 1000
        except (TypeError, ValueError):
            timestamp_ms = None
    return {
        "price": price,
        "fee_usd": fee_usd,
        "fee_rate_bps": fee_rate_bps,
        "tx_hash": _response_value(response, "transaction_hash", "tx_hash", default=None),
        "timestamp_ms": timestamp_ms,
    }


def _epoch_seconds(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 1000.0 if numeric >= 10_000_000_000 else numeric


class OfficialFOKExecutor:
    """Two-leg FOK executor using the official Python SDK.

    The client is injected so the safety logic can be tested without a wallet.
    FOK is required: a partial first leg is not an arbitrage pair. If the second
    leg is rejected after the first filled, the executor attempts an immediate
    FAK unwind and raises if that unwind is not fully confirmed.
    """

    def __init__(self, client, rollback_min_price: float = 0.0001,
                 journal: Optional[LiveOrderJournal] = None,
                 max_retries: int = 3, auto_merge: bool = False,
                 auto_redeem: bool = True,
                 directional_journal: Optional[LiveDirectionalJournal] = None,
                 rest_seconds: Optional[float] = None):
        self.client = client
        self.rollback_min_price = max(0.0001, min(1.0, rollback_min_price))
        self.journal = journal
        self.directional_journal = directional_journal
        self.negrisk_journal = None
        self.negrisk_executor = None
        self.max_retries = max(0, int(max_retries))
        self.auto_merge = bool(auto_merge)
        self.auto_redeem = bool(auto_redeem)
        self.maker_rest_seconds = (
            rest_seconds if rest_seconds is not None else maker_rest_seconds()
        )
        self.user_stream_healthy = False

    @classmethod
    async def create_from_env(cls, journal: Optional[LiveOrderJournal] = None,
                              directional_journal: Optional[LiveDirectionalJournal] = None):
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        if not private_key or private_key.lower().startswith("your_"):
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is missing or is still a placeholder")
        try:
            from polymarket import AsyncSecureClient
        except ImportError as error:
            raise RuntimeError("Install the official SDK with: uv sync --extra live") from error
        kwargs = {"private_key": private_key}
        wallet = os.getenv("POLYMARKET_WALLET_ADDRESS", "").strip()
        if wallet:
            kwargs["wallet"] = wallet
        client = await AsyncSecureClient.create(**kwargs)
        return cls(
            client,
            journal=journal,
            directional_journal=directional_journal,
            auto_merge=os.getenv("AUTO_MERGE_COMPLETE_SETS", "0") == "1",
            auto_redeem=os.getenv("AUTO_REDEEM_RESOLVED_POSITIONS", "1") == "1",
        )

    @staticmethod
    def _exception_status(error) -> Optional[int]:
        return _response_status(error) or _response_status(getattr(error, "response", None))

    async def _call(self, name: str, **kwargs):
        method = getattr(self.client, name, None)
        if method is None:
            raise RuntimeError(f"Official client does not expose {name}")
        for attempt in range(self.max_retries + 1):
            try:
                result = method(**kwargs)
                result = await result if inspect.isawaitable(result) else result
                status = _response_status(result)
                if status == 425:
                    raise MatchingEngineRestartError("matching engine restart (HTTP 425)")
                if status == 503:
                    raise CancelOnlyError("cancel-only/post-only mode (HTTP 503)")
                return result
            except MatchingEngineRestartError:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(min(2.0, 0.25 * (2 ** attempt)))
            except Exception as error:
                status = self._exception_status(error)
                if status == 425 and attempt < self.max_retries:
                    await asyncio.sleep(min(2.0, 0.25 * (2 ** attempt)))
                    continue
                if status == 425:
                    raise MatchingEngineRestartError("matching engine restart (HTTP 425)") from error
                if status == 503:
                    raise CancelOnlyError("cancel-only/post-only mode (HTTP 503)") from error
                raise

    async def preflight(self, required_usd: float = 0.01) -> dict:
        """Verify collateral and allowance; never silently approve spending."""
        if required_usd < 0.0:
            raise ValueError("required_usd must not be negative")
        setup = getattr(self.client, "setup_trading_approvals", None)
        if setup and os.getenv("POLYMARKET_SETUP_APPROVALS") == "1":
            result = setup()
            if inspect.isawaitable(result):
                await result
        snapshot = await self._call("get_balance_allowance", asset_type="COLLATERAL")
        allowances = _response_value(snapshot, "allowances", default=None)
        if allowances is not None:
            balance_raw = _numeric_value(snapshot, "balance")
            if not isinstance(allowances, dict) or not allowances:
                raise RuntimeError("collateral allowance map is empty; refusing live execution")
            configured_spender = os.getenv("POLYMARKET_ALLOWANCE_SPENDER", "").strip().lower()
            if configured_spender:
                selected = next((value for key, value in allowances.items()
                                 if str(key).lower() == configured_spender), None)
                if selected is None:
                    raise RuntimeError("configured POLYMARKET_ALLOWANCE_SPENDER is not in the allowance response")
            elif len(allowances) == 1:
                selected = next(iter(allowances.values()))
            else:
                raise RuntimeError(
                    "multiple collateral spenders returned; set POLYMARKET_ALLOWANCE_SPENDER explicitly"
                )
            balance = balance_raw / 1_000_000 if balance_raw is not None else None
            allowance_raw = _numeric_value({"value": selected}, "value")
            allowance = allowance_raw / 1_000_000 if allowance_raw is not None else None
        else:
            balance = _numeric_value(snapshot, "balance", "available", "cash", "collateral")
            allowance = _numeric_value(snapshot, "allowance", "approved", "spending_limit")
        if balance is None or allowance is None:
            raise RuntimeError("balance/allowance response is incomplete; refusing live execution")
        if (not math.isfinite(balance) or balance < 0.0
                or not math.isfinite(allowance) or allowance < 0.0):
            raise RuntimeError("balance/allowance response is invalid; refusing live execution")
        if balance + 1e-9 < required_usd:
            raise RuntimeError(f"insufficient collateral: {balance:.6f} < {required_usd:.6f}")
        if allowance + 1e-9 < required_usd:
            raise RuntimeError(f"insufficient collateral allowance: {allowance:.6f} < {required_usd:.6f}")
        return {"balance": balance, "allowance": allowance}

    async def reconcile(self, stale_after_seconds: float = 30.0,
                        recover_orphans: bool = True,
                        scan_account: Optional[bool] = None) -> List[dict]:
        """Re-read unfinished pairs and optionally scan the whole account.

        Known-pair recovery is not enough: an external order, leftover
        conditional-token balance, or unexplained collateral move is inventory
        this process does not own and must not trade around.
        """
        if not self.journal:
            return []
        issues = self.journal.integrity_issues()
        if issues:
            raise UnhedgedPairError("live journal integrity failure: " + "; ".join(issues))
        if scan_account is None:
            scan_account = recover_orphans
        conditions = sorted({
            str(record.get("condition_id", ""))
            for record in self.journal.incomplete_pairs()
            if record.get("condition_id")
        })
        open_orders: Dict[str, object] = {}
        account_trades: List[object] = []
        trade_watermarks = self.journal.state.setdefault("trade_watermarks", {})
        if scan_account:
            open_orders.update(await self._list_open_orders())
        for condition_id in conditions:
            if recover_orphans and not scan_account:
                open_orders.update(await self._list_open_orders(market=condition_id))
            after = self._trade_query_after(condition_id, trade_watermarks)
            account_trades.extend(
                await self._list_account_trades(market=condition_id, after=after)
            )
        if recover_orphans or account_trades:
            self._recover_orphan_order_ids(open_orders, account_trades)
        orphaned_intents = [
            str(record.get("pair_id", ""))
            for record in self.journal.incomplete_pairs()
            if record.get("status") == "PENDING"
            and not record.get("yes_order_id")
            and not record.get("no_order_id")
        ]
        if orphaned_intents:
            for pair_id in orphaned_intents:
                self.journal.update(
                    pair_id,
                    status="UNHEDGED",
                    error="pair intent has no submitted order IDs after restart",
                )
            raise UnhedgedPairError(
                "live pair intent has no submitted order IDs after restart: "
                + ", ".join(orphaned_intents)
            )
        for trade in account_trades:
            order_id = str(_response_value(trade, "taker_order_id", "order_id", default=""))
            if not order_id:
                continue
            record = self.journal.pair_for_order(order_id)
            if record is None:
                continue
            token_id = str(_response_value(trade, "token_id", "asset_id", default=""))
            if token_id not in {record["yes_token_id"], record["no_token_id"]}:
                raise UnhedgedPairError(f"trade {order_id} has an unexpected token")
            trade_market = _response_value(trade, "condition_id", "market", default=None)
            if trade_market is not None and str(trade_market) != str(record["condition_id"]):
                raise UnhedgedPairError(f"trade {order_id} belongs to an unexpected condition")
            trade_side = str(_response_value(trade, "side", default="")).upper()
            if trade_side and trade_side != "BUY":
                raise UnhedgedPairError(f"trade {order_id} has an unexpected side")
            leg = "yes" if token_id == record["yes_token_id"] else "no"
            size = _numeric_value(trade, "size", "matched_amount")
            trade_id = _response_value(trade, "id", "trade_id", "tradeId", default=None)
            if size is None or size < 0.0:
                raise UnhedgedPairError(f"trade {trade_id or order_id} has no valid matched size")
            economics = _trade_fill_economics(trade)
            self.journal.add_fill(
                record["pair_id"], leg, size, str(trade_id) if trade_id else None,
                price=economics["price"], fee_usd=economics["fee_usd"],
                fee_rate_bps=economics["fee_rate_bps"], tx_hash=economics["tx_hash"],
                timestamp_ms=economics["timestamp_ms"],
            )
        self._advance_trade_watermarks(account_trades, trade_watermarks)
        if account_trades:
            self.journal.save()
        reconciled = []
        for record in self.journal.incomplete_pairs():
            for leg in ("yes", "no"):
                order_id = str(record.get(f"{leg}_order_id", ""))
                if not order_id:
                    continue
                order = await self._call("get_order", order_id=order_id)
                if order is None:
                    raise UnhedgedPairError(f"order {order_id} is missing during reconciliation")
                order_token = _response_value(order, "token_id", "asset_id", default=None)
                if order_token is not None and str(order_token) != str(record[f"{leg}_token_id"]):
                    raise UnhedgedPairError(f"order {order_id} has an unexpected token")
                order_market = _response_value(order, "condition_id", "market", default=None)
                if order_market is not None and str(order_market) != str(record["condition_id"]):
                    raise UnhedgedPairError(f"order {order_id} belongs to an unexpected condition")
                order_side = str(_response_value(order, "side", default="")).upper()
                if order_side and order_side != "BUY":
                    raise UnhedgedPairError(f"order {order_id} has an unexpected side")
                matched = _filled_shares(order)
                if matched is None:
                    raise RuntimeError(f"order {order_id} has no confirmed matched size")
                self.journal.set_matched(record["pair_id"], leg, matched)
                status = str(_response_value(order, "status", "state", default="")).upper()
                self.journal.set_order_status(record["pair_id"], leg, status)
                if order_id in open_orders:
                    status = "OPEN"
                if status in {"OPEN", "LIVE", "ACTIVE", "PENDING"}:
                    age = time.time() - float(record.get("created_at", time.time()))
                    timeout = (
                        self.maker_rest_seconds
                        if str(record.get("order_style", "FOK")).upper() == "GTC"
                        else stale_after_seconds
                    )
                    if age >= max(0.0, timeout):
                        await self._call("cancel_order", order_id=order_id)
                        cancelled = await self._call("get_order", order_id=order_id)
                        cancelled_matched = _filled_shares(cancelled)
                        if cancelled_matched is None:
                            raise RuntimeError(f"order {order_id} has no confirmed matched size after cancel")
                        self.journal.set_matched(record["pair_id"], leg, cancelled_matched)
                        cancelled_status = str(
                            _response_value(cancelled, "status", "state", default="CANCELLED") or "CANCELLED"
                        ).upper()
                        self.journal.set_order_status(record["pair_id"], leg, cancelled_status)
            refreshed = self.journal._record(record["pair_id"])
            self._update_pair_status(refreshed)
            if refreshed["status"] == "UNHEDGED" and str(refreshed.get("order_style", "")).upper() == "GTC":
                await self._unwind_gtc_imbalance(refreshed)
                refreshed = self.journal._record(record["pair_id"])
            if refreshed["status"] == "UNHEDGED":
                raise UnhedgedPairError(f"pair {record['pair_id']} is not fully hedged after reconciliation")
            if refreshed["status"] == "RESTING":
                reconciled.append(refreshed)
                continue
            await self._reconcile_conditional_balances(refreshed)
            reconciled.append(refreshed)
        if scan_account:
            await self._reconcile_external_account_state(open_orders)
        return reconciled

    def _recover_orphan_order_ids(self, open_orders: Dict[str, object],
                                  account_trades: Sequence[object]) -> None:
        """Bind durable pair intents to accepted orders after a crash."""
        if not self.journal:
            return
        candidates_by_leg: Dict[tuple[str, str], set[str]] = {}
        for order_id, order in open_orders.items():
            token = str(_response_value(order, "token_id", "asset_id", default=""))
            condition = str(_response_value(order, "condition_id", "market", default=""))
            side = str(_response_value(order, "side", default="")).upper()
            if token and condition and side in {"", "BUY"}:
                candidates_by_leg.setdefault((condition, token), set()).add(str(order_id))
        for trade in account_trades:
            order_id = str(_response_value(trade, "taker_order_id", "order_id", default=""))
            token = str(_response_value(trade, "token_id", "asset_id", default=""))
            condition = str(_response_value(trade, "condition_id", "market", default=""))
            side = str(_response_value(trade, "side", default="")).upper()
            if order_id and token and condition and side in {"", "BUY"}:
                candidates_by_leg.setdefault((condition, token), set()).add(order_id)

        for record in self.journal.incomplete_pairs():
            created_at = float(record.get("created_at", 0.0))
            for leg in ("yes", "no"):
                if record.get(f"{leg}_order_id"):
                    continue
                key = (str(record.get("condition_id", "")), str(record.get(f"{leg}_token_id", "")))
                candidates = set()
                for order_id in candidates_by_leg.get(key, set()):
                    order = open_orders.get(order_id)
                    source = order if order is not None else next(
                        (trade for trade in account_trades
                         if str(_response_value(trade, "taker_order_id", "order_id", default="")) == order_id),
                        None,
                    )
                    observed_at = _epoch_seconds(_response_value(
                        source, "created_at", "matched_at", "match_time", "timestamp", default=None
                    )) if source is not None else None
                    if observed_at is None or observed_at >= created_at - 10.0:
                        candidates.add(order_id)
                if len(candidates) > 1:
                    raise UnhedgedPairError(
                        f"pair {record['pair_id']} has ambiguous orphan {leg} orders"
                    )
                if candidates:
                    self.journal.set_order_id(record["pair_id"], leg, next(iter(candidates)))

    @staticmethod
    def _transaction_hash(value) -> str:
        raw = _response_value(value, "transaction_hash", "transactionHash", "hash", default="")
        return str(raw or "")

    async def merge_pair(self, record: dict) -> dict:
        """Merge a fully matched binary pair and wait for chain confirmation."""
        if not self.journal:
            raise RuntimeError("complete-set merge requires a live journal")
        pair_id = str(record["pair_id"])
        current = self.journal._record(pair_id)
        if current.get("status") == "SETTLED":
            return current
        if current.get("status") != "HEDGED":
            raise RuntimeError(f"pair {pair_id} is not fully hedged")
        if current.get("settlement_type") == "MERGE_SUBMITTED":
            raise RuntimeError(f"pair {pair_id} has a submitted merge requiring reconciliation")
        merge = getattr(self.client, "merge_positions", None)
        if merge is None:
            raise RuntimeError("Official client does not expose merge_positions")
        requested = Decimal(str(current.get("requested_shares", 0.0)))
        amount = int((requested * Decimal(1_000_000)).to_integral_value(rounding=ROUND_DOWN))
        if amount <= 0:
            raise RuntimeError(f"pair {pair_id} has no mergeable base-unit amount")
        attempts = int(current.get("merge_attempts", 0)) + 1
        self.journal.update(pair_id, merge_attempts=attempts, settlement_type="MERGE_PENDING")
        handle = await self._call(
            "merge_positions",
            condition_id=str(current["condition_id"]),
            amount=amount,
            metadata=f"Merge complete set for pair {pair_id}",
        )
        transaction_id = str(_response_value(handle, "transaction_id", "transactionID", default="") or "")
        submitted_hash = self._transaction_hash(handle)
        self.journal.update(
            pair_id,
            settlement_type="MERGE_SUBMITTED",
            settlement_tx_id=transaction_id,
            settlement_tx_hash=submitted_hash,
        )
        wait = getattr(handle, "wait", None)
        if wait is None:
            raise RuntimeError("merge transaction handle has no wait method")
        outcome = wait()
        outcome = await outcome if inspect.isawaitable(outcome) else outcome
        transaction_hash = self._transaction_hash(outcome) or submitted_hash
        if not transaction_hash:
            raise RuntimeError("merge transaction completed without a transaction hash")
        return self.journal.mark_merged(pair_id, transaction_hash)

    async def redeem_pair(self, record: dict) -> dict:
        """Redeem the winning token after resolution and wait for confirmation."""
        if not self.journal:
            raise RuntimeError("market redemption requires a live journal")
        pair_id = str(record["pair_id"])
        current = self.journal._record(pair_id)
        if current.get("status") == "SETTLED":
            return current
        if current.get("status") != "RESOLVED_PENDING_REDEMPTION":
            raise RuntimeError(f"pair {pair_id} is not pending redemption")
        if current.get("settlement_type") == "REDEEM_SUBMITTED":
            raise RuntimeError(f"pair {pair_id} has a submitted redemption requiring reconciliation")
        if getattr(self.client, "redeem_positions", None) is None:
            raise RuntimeError("Official client does not expose redeem_positions")
        attempts = int(current.get("redemption_attempts", 0)) + 1
        self.journal.update(
            pair_id, redemption_attempts=attempts, settlement_type="REDEEM_PENDING"
        )
        handle = await self._call(
            "redeem_positions", condition_id=str(current["condition_id"])
        )
        transaction_id = str(_response_value(handle, "transaction_id", "transactionID", default="") or "")
        submitted_hash = self._transaction_hash(handle)
        self.journal.update(
            pair_id,
            settlement_type="REDEEM_SUBMITTED",
            redemption_tx_id=transaction_id,
            settlement_tx_id=transaction_id,
            settlement_tx_hash=submitted_hash,
        )
        wait = getattr(handle, "wait", None)
        if wait is None:
            raise RuntimeError("redemption transaction handle has no wait method")
        outcome = wait()
        outcome = await outcome if inspect.isawaitable(outcome) else outcome
        transaction_hash = self._transaction_hash(outcome) or submitted_hash
        if not transaction_hash:
            raise RuntimeError("redemption transaction completed without a transaction hash")
        return self.journal.mark_redeemed(pair_id, transaction_hash)

    async def _submitted_transaction_hash(self, record: dict) -> Optional[str]:
        """Poll one already-submitted settlement transaction without resubmitting it."""
        transaction_id = str(record.get("settlement_tx_id") or record.get("redemption_tx_id") or "")
        transaction_hash = str(record.get("settlement_tx_hash") or "")
        if transaction_id:
            context = getattr(self.client, "_ctx", None)
            relayer = getattr(context, "relayer", None)
            get_relayer_transaction = getattr(self.client, "get_relayer_transaction", None)
            if get_relayer_transaction is not None:
                transaction = get_relayer_transaction(transaction_id=transaction_id)
                transaction = await transaction if inspect.isawaitable(transaction) else transaction
            elif relayer is not None:
                try:
                    from polymarket._internal.actions.relayer.poll import fetch_gasless_transaction
                except ImportError as error:
                    raise RuntimeError("official SDK relayer polling is unavailable") from error
                transaction = await fetch_gasless_transaction(
                    relayer, transaction_id=transaction_id
                )
            else:
                transaction = None
            if transaction is not None:
                raw_state = _response_value(transaction, "state", default="")
                raw_state = getattr(raw_state, "value", raw_state)
                state = str(raw_state or "").strip().lower().rsplit(".", 1)[-1]
                if state.startswith("state_"):
                    state = state[6:]
                if state in {"failed", "invalid"}:
                    raise RuntimeError(
                        f"settlement transaction {transaction_id} reached terminal state {state}"
                    )
                if state != "confirmed":
                    return None
                return self._transaction_hash(transaction) or transaction_hash or None
        if transaction_hash:
            receipt = None
            get_receipt = getattr(self.client, "get_transaction_receipt", None)
            if get_receipt is not None:
                receipt = get_receipt(transaction_hash=transaction_hash)
                receipt = await receipt if inspect.isawaitable(receipt) else receipt
            else:
                context = getattr(self.client, "_ctx", None)
                rpc = getattr(context, "rpc", None)
                if rpc is None:
                    return None
                receipt = await rpc.eth_get_transaction_receipt(transaction_hash)
            if receipt is None:
                return None
            status = _response_value(receipt, "status", default=None)
            if status in ("0x1", 1, "1"):
                return transaction_hash
            if status in ("0x0", 0, "0"):
                raise RuntimeError(f"settlement transaction {transaction_hash} reverted")
            raise RuntimeError(
                f"settlement transaction {transaction_hash} has unknown receipt status {status!r}"
            )
        return None

    async def _resume_submitted_settlements(self) -> List[dict]:
        """Finalize confirmed submitted settlements without creating new transactions."""
        if not self.journal:
            return []
        settled = []
        for record in list(self.journal.incomplete_pairs()):
            if record.get("settlement_type") not in {"MERGE_SUBMITTED", "REDEEM_SUBMITTED"}:
                continue
            try:
                transaction_hash = await self._submitted_transaction_hash(record)
                if not transaction_hash:
                    continue
                if record.get("settlement_type") == "MERGE_SUBMITTED":
                    settled.append(self.journal.mark_merged(record["pair_id"], transaction_hash))
                else:
                    settled.append(self.journal.mark_redeemed(record["pair_id"], transaction_hash))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.journal.update(record["pair_id"], error=str(error))
        return settled

    async def settle_hedged_pairs(self) -> List[dict]:
        """Best-effort merge/redeem; temporary chain lag is retried safely."""
        settled: List[dict] = []
        if self.journal:
            settled = await self._resume_submitted_settlements()
            for record in list(self.journal.incomplete_pairs()):
                action = None
                if record.get("status") == "HEDGED" and self.auto_merge:
                    action = self.merge_pair
                elif record.get("status") == "RESOLVED_PENDING_REDEMPTION" and self.auto_redeem:
                    action = self.redeem_pair
                if action is None:
                    continue
                try:
                    settled.append(await action(record))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.journal.update(record["pair_id"], error=str(error))
                    # A submitted chain transaction remains submitted. It must not
                    # be recreated after a timeout or process restart.
        negrisk_executor = getattr(self, "negrisk_executor", None)
        if negrisk_executor is not None:
            settled.extend(await negrisk_executor.settle_baskets())
        return settled

    def _trade_query_after(self, condition_id: str, watermarks: dict) -> Optional[str]:
        """Return an overlapping Unix-seconds watermark for incremental recovery."""
        observed = _epoch_seconds(watermarks.get(str(condition_id)))
        if observed is None:
            created = [
                _epoch_seconds(record.get("created_at"))
                for record in self.journal.incomplete_pairs()
                if str(record.get("condition_id", "")) == str(condition_id)
            ]
            created = [value for value in created if value is not None]
            observed = min(created) if created else None
        if observed is None:
            return None
        return str(max(0, int(observed - 60.0)))

    @staticmethod
    def _advance_trade_watermarks(trades: Sequence[object], watermarks: dict) -> None:
        latest_by_market: Dict[str, float] = {}
        for trade in trades:
            market = _response_value(trade, "condition_id", "market", default=None)
            timestamp = _response_value(
                trade, "matched_at", "match_time", "timestamp", "updated_at", default=None
            )
            observed = _epoch_seconds(timestamp)
            if market is None or observed is None:
                continue
            key = str(market)
            latest_by_market[key] = max(observed, latest_by_market.get(key, 0.0))
        for market, observed in latest_by_market.items():
            previous = _epoch_seconds(watermarks.get(market))
            if previous is None or observed > previous:
                watermarks[market] = int(observed)

    async def _list_account_trades(self, market: Optional[str] = None,
                                   after: Optional[str] = None) -> List[object]:
        """Drain the official account-trade paginator for restart recovery."""
        method = getattr(self.client, "list_account_trades", None)
        if method is None:
            raise RuntimeError("Official client does not expose list_account_trades")
        params = {}
        if market:
            params["market"] = str(market)
        if after:
            params["after"] = str(after)
        result = method(**params)
        result = await result if inspect.isawaitable(result) else result
        trades: List[object] = []
        if hasattr(result, "iter_items"):
            async for trade in result.iter_items():
                trades.append(trade)
        elif hasattr(result, "__aiter__"):
            async for page in result:
                trades.extend(_response_value(page, "items", default=()) or ())
        elif isinstance(result, (list, tuple)):
            trades.extend(result)
        elif isinstance(result, dict):
            items = result.get("data", result.get("trades", []))
            if isinstance(items, (list, tuple)):
                trades.extend(items)
        return trades

    async def _list_open_orders(self, market: Optional[str] = None) -> Dict[str, object]:
        method = getattr(self.client, "list_open_orders", None)
        if method is None:
            raise RuntimeError("Official client does not expose list_open_orders")
        result = method(**({"market": str(market)} if market else {}))
        result = await result if inspect.isawaitable(result) else result
        orders: Dict[str, object] = {}
        if hasattr(result, "__aiter__"):
            async for page in result:
                items = _response_value(page, "items", default=()) or ()
                for order in items:
                    order_id = _response_value(order, "id", "order_id", "orderID", default="")
                    if order_id:
                        orders[str(order_id)] = order
        elif isinstance(result, (list, tuple)):
            for order in result:
                order_id = _response_value(order, "id", "order_id", "orderID", default="")
                if order_id:
                    orders[str(order_id)] = order
        elif isinstance(result, dict):
            items = result.get("data", result.get("orders", []))
            for order in items if isinstance(items, (list, tuple)) else []:
                order_id = _response_value(order, "id", "order_id", "orderID", default="")
                if order_id:
                    orders[str(order_id)] = order
        return orders

    async def _conditional_balance(self, token_id: str) -> float:
        """Read a conditional-token balance in share units from the official API."""
        snapshot = await self._call(
            "get_balance_allowance", asset_type="CONDITIONAL", token_id=str(token_id)
        )
        raw_balance = _numeric_value(snapshot, "balance")
        if raw_balance is None or not math.isfinite(raw_balance) or raw_balance < 0.0:
            raise RuntimeError(f"conditional balance is missing for token {token_id}")
        allowances = _response_value(snapshot, "allowances", default=None)
        # The official SDK returns conditional balances in 1e6 base units. Test
        # doubles and non-official adapters may return already-normalized shares.
        balance = raw_balance / 1_000_000.0 if isinstance(allowances, dict) else raw_balance
        if not math.isfinite(balance) or balance < 0.0:
            raise RuntimeError(f"conditional balance is invalid for token {token_id}")
        return balance

    async def _reconcile_conditional_balances(self, record: dict) -> None:
        """Fail closed when confirmed fills are not present in the wallet."""
        status = record.get("status")
        if status == "HEDGED" and record.get("settlement_type") != "MERGE_SUBMITTED":
            required = float(record["requested_shares"])
            for leg in ("yes", "no"):
                balance = await self._conditional_balance(record[f"{leg}_token_id"])
                if balance + 1e-8 < required:
                    raise UnhedgedPairError(
                        f"pair {record['pair_id']} {leg} conditional balance is "
                        f"{balance:.8f}, below required {required:.8f}"
                    )
        elif status == "RESOLVED_PENDING_REDEMPTION" and record.get("settlement_type") != "REDEEM_SUBMITTED":
            winning = str(record.get("winning_outcome", ""))
            token_id = None
            if winning in {"yes", "Yes", "YES", str(record.get("yes_token_id"))}:
                token_id = record["yes_token_id"]
            elif winning in {"no", "No", "NO", str(record.get("no_token_id"))}:
                token_id = record["no_token_id"]
            if token_id is None:
                raise UnhedgedPairError(f"pair {record['pair_id']} has no redeemable winning token")
            balance = await self._conditional_balance(token_id)
            required = float(record["requested_shares"])
            if balance + 1e-8 < required:
                raise UnhedgedPairError(
                    f"pair {record['pair_id']} winning conditional balance is "
                    f"{balance:.8f}, below required {required:.8f}"
                )

    async def _list_positions(self) -> List[object]:
        """Best-effort account position list; missing SDK methods skip the scan."""
        for name in ("list_positions", "get_positions", "get_current_positions"):
            method = getattr(self.client, name, None)
            if method is None:
                continue
            result = method()
            result = await result if inspect.isawaitable(result) else result
            if isinstance(result, (list, tuple)):
                return list(result)
            if isinstance(result, dict):
                items = result.get("data", result.get("positions", result.get("items", [])))
                if isinstance(items, (list, tuple)):
                    return list(items)
        return []

    def _known_live_order_ids(self) -> set[str]:
        known = set()
        if self.journal:
            for record in self.journal.incomplete_pairs():
                for field in ("yes_order_id", "no_order_id"):
                    order_id = str(record.get(field) or "")
                    if order_id:
                        known.add(order_id)
        if self.directional_journal:
            known |= self.directional_journal.known_order_ids()
        negrisk_journal = getattr(self, "negrisk_journal", None)
        if negrisk_journal is not None:
            known |= negrisk_journal.known_order_ids()
        return known

    def _known_live_token_ids(self) -> set[str]:
        known = set()
        if self.journal:
            for record in self.journal.incomplete_pairs():
                for field in ("yes_token_id", "no_token_id"):
                    token_id = str(record.get(field) or "")
                    if token_id:
                        known.add(token_id)
        if self.directional_journal:
            known |= self.directional_journal.known_inventory_token_ids()
        negrisk_journal = getattr(self, "negrisk_journal", None)
        if negrisk_journal is not None:
            known |= negrisk_journal.known_inventory_token_ids()
        return known

    async def _reconcile_external_account_state(self, open_orders: Dict[str, object]) -> None:
        """Halt when the account holds orders or tokens this journal does not own."""
        known_orders = self._known_live_order_ids()
        external_orders = sorted(order_id for order_id in open_orders if order_id not in known_orders)
        if external_orders:
            raise UnhedgedPairError(
                "account has open orders outside the live journal: " + ", ".join(external_orders)
            )
        known_tokens = self._known_live_token_ids()
        external_positions = []
        for position in await self._list_positions():
            token_id = str(_response_value(position, "token_id", "asset_id", "assetId", default="") or "")
            size = _numeric_value(position, "size", "balance", "shares", "position")
            if not token_id or size is None or size <= 1e-8:
                continue
            if token_id not in known_tokens:
                external_positions.append(f"{token_id}:{size}")
        if external_positions:
            raise UnhedgedPairError(
                "account has conditional-token inventory outside the live journal: "
                + ", ".join(external_positions)
            )

    async def cancel_all_open_orders(self) -> List[str]:
        """Cancel every resting order; missing IDs are an error, not a skip."""
        cancelled: List[str] = []
        cancel_all = getattr(self.client, "cancel_all_orders", None) or getattr(self.client, "cancel_all", None)
        if cancel_all is not None:
            result = cancel_all()
            result = await result if inspect.isawaitable(result) else result
            if isinstance(result, (list, tuple)):
                cancelled.extend(str(item) for item in result)
            elif isinstance(result, dict):
                items = result.get("cancelled", result.get("order_ids", result.get("orders", [])))
                if isinstance(items, (list, tuple)):
                    cancelled.extend(str(item) for item in items)
        orders = await self._list_open_orders()
        remaining = [order_id for order_id in orders if order_id not in set(cancelled)]
        for order_id in remaining:
            await self._call("cancel_order", order_id=order_id)
            cancelled.append(order_id)
        leftover = await self._list_open_orders()
        if leftover:
            raise UnhedgedPairError(
                "open orders remain after cancel-all: " + ", ".join(sorted(leftover))
            )
        return cancelled

    async def _inventory_snapshot(self) -> dict:
        snapshot = {
            "open_orders": [],
            "pairs": [],
            "conditional_balances": [],
        }
        try:
            snapshot["open_orders"] = sorted(await self._list_open_orders())
        except Exception as error:
            snapshot["open_orders_error"] = str(error)
        if self.journal:
            for record in self.journal.incomplete_pairs():
                snapshot["pairs"].append({
                    "pair_id": record.get("pair_id"),
                    "status": record.get("status"),
                    "yes_matched_shares": record.get("yes_matched_shares"),
                    "no_matched_shares": record.get("no_matched_shares"),
                    "error": record.get("error", ""),
                })
                for leg in ("yes", "no"):
                    token_id = str(record.get(f"{leg}_token_id") or "")
                    if not token_id:
                        continue
                    try:
                        balance = await self._conditional_balance(token_id)
                        snapshot["conditional_balances"].append({
                            "token_id": token_id, "shares": balance, "leg": leg,
                            "pair_id": record.get("pair_id"),
                        })
                    except Exception as error:
                        snapshot["conditional_balances"].append({
                            "token_id": token_id, "error": str(error), "leg": leg,
                            "pair_id": record.get("pair_id"),
                        })
        if self.directional_journal:
            snapshot["directional"] = self.directional_journal.summary()
            snapshot["directional_inventory"] = self.directional_journal.inventory_by_token()
        return snapshot

    async def _flatten_directional_inventory(self) -> dict:
        """FAK-sell unhedged directional inventory. Hedged Yes+No pairs stay put."""
        flattened: List[dict] = []
        leftover: List[dict] = []
        journal = self.directional_journal
        if journal is None:
            return {"flattened": flattened, "leftover": leftover}
        for token_id, shares in list(journal.inventory_by_token().items()):
            if shares <= 1e-8:
                continue
            if shares < 0.0:
                leftover.append({
                    "token_id": token_id, "shares": shares,
                    "error": "negative directional inventory requires manual review",
                })
                continue
            try:
                response = await self._call(
                    "place_market_order",
                    token_id=token_id,
                    side="SELL",
                    shares=f"{shares:.6f}",
                    min_price=f"{self.rollback_min_price:.6f}",
                    order_type="FAK",
                )
            except Exception as error:
                leftover.append({"token_id": token_id, "shares": shares, "error": str(error)})
                continue
            filled = _filled_shares(response, side="SELL") or 0.0
            if not math.isfinite(filled) or filled < 0.0:
                filled = 0.0
            filled = min(shares, filled)
            if not _response_ok(response) or filled <= 1e-8:
                leftover.append({
                    "token_id": token_id, "shares": shares,
                    "error": str(_response_value(response, "message", "errorMsg", default="directional FAK rejected")),
                })
                continue
            remaining = max(0.0, shares - filled)
            order_id = str(_response_value(response, "order_id", "orderID", default="") or "")
            journal.record_halt_flatten(token_id, filled, remaining, order_id=order_id)
            flattened.append({"token_id": token_id, "sold": filled, "remaining": remaining})
            if remaining > 1e-8:
                leftover.append({
                    "token_id": token_id, "shares": remaining,
                    "error": "partial FAK; leftover directional inventory needs manual handling",
                })
        return {"flattened": flattened, "leftover": leftover}

    async def flatten_on_halt(self, reason: str) -> dict:
        """Cancel resting orders, FAK-sell directional inventory, snapshot pair leftovers."""
        cancelled = await self.cancel_all_open_orders()
        await self._unwind_resting_gtc_pairs()
        directional_flatten = await self._flatten_directional_inventory()
        inventory = await self._inventory_snapshot()
        leftover = list(directional_flatten.get("leftover") or [])
        result = {
            "reason": str(reason),
            "cancelled_order_ids": cancelled,
            "inventory": inventory,
            "directional_flatten": directional_flatten,
            "note": (
                "matched pair inventory is not auto-sold; residual pairs need manual handling. "
                "unhedged directional inventory was FAK-sold; leftovers are in halt_inventory"
            ),
        }
        if leftover:
            result["note"] += "; directional leftover remains"
        if self.journal:
            self.journal.state["halt_inventory"] = result
            self.journal.save()
        if self.directional_journal:
            self.directional_journal.state["halt_inventory"] = {
                "reason": str(reason),
                "directional_flatten": directional_flatten,
            }
            self.directional_journal.save()
        return result

    async def apply_halt_actions(self, risk: "LiveRiskController") -> dict:
        """Run cancel-all once after a persisted halt, then leave inventory for review."""
        if not risk.state.get("halted"):
            return {}
        if risk.state.get("flatten_completed") and not risk.state.get("flatten_required"):
            return dict(risk.state.get("flatten_result") or {})
        result = await self.flatten_on_halt(risk.state.get("halt_reason") or "live risk halt")
        risk.state["flatten_required"] = False
        risk.state["flatten_completed"] = True
        risk.state["flatten_result"] = {
            "cancelled_order_ids": result.get("cancelled_order_ids", []),
            "open_orders": result.get("inventory", {}).get("open_orders", []),
            "pairs": result.get("inventory", {}).get("pairs", []),
            "directional_flatten": result.get("directional_flatten", {}),
        }
        risk.save()
        return result

    def _update_pair_status(self, record: dict) -> dict:
        requested = float(record["requested_shares"])
        yes = float(record["yes_matched_shares"])
        no = float(record["no_matched_shares"])
        complete_yes = yes + 1e-8 >= requested
        complete_no = no + 1e-8 >= requested
        terminal = {"CANCELED", "CANCELLED", "UNMATCHED", "FAILED", "REJECTED", "KILLED", "EXPIRED", "FILLED", "MATCHED"}
        yes_terminal = record.get("yes_order_status") in terminal
        no_terminal = record.get("no_order_status") in terminal
        is_gtc = str(record.get("order_style", "FOK")).upper() == "GTC"
        if complete_yes and complete_no:
            if record.get("status") not in {
                "SETTLED", "ROLLED_BACK", "RESOLVED_PENDING_REDEMPTION"
            }:
                record["status"] = "HEDGED"
        elif yes <= 1e-8 and no <= 1e-8 and yes_terminal and no_terminal:
            record["status"] = "REJECTED"
        elif is_gtc and not (yes_terminal and no_terminal):
            # One or both GTC legs are still working. A single fill is not
            # an unhedged halt while the other order remains live.
            if record.get("status") not in {
                "SETTLED", "ROLLED_BACK", "RESOLVED_PENDING_REDEMPTION", "HEDGED",
            }:
                record["status"] = "RESTING"
        elif (yes > 1e-8 or no > 1e-8) and (yes_terminal or no_terminal):
            record["status"] = "UNHEDGED"
        elif record.get("status") not in {
            "PENDING", "REJECTED", "HEDGED", "RESOLVED_PENDING_REDEMPTION",
            "SETTLED", "ROLLED_BACK", "RESTING",
        }:
            record["status"] = "PENDING"
        record["updated_at"] = time.time()
        self.journal.save()
        return record

    def _handle_directional_user_event(self, record: dict, event, event_type: str) -> dict:
        """Apply a user-stream fill to a directional trade. BUY and SELL are both valid."""
        trade_id = str(record["trade_id"])
        event_side = str(_response_value(event, "side", default="") or "").upper()
        expected = str(record.get("side") or "").upper()
        if event_side and expected and event_side != expected:
            raise UnhedgedPairError(
                f"directional event {record.get('order_id')} has unexpected side {event_side}"
            )
        event_token = str(_response_value(event, "token_id", "asset_id", "assetId", default="") or "")
        if event_token and event_token != str(record.get("token_id") or ""):
            raise UnhedgedPairError(
                f"directional event {record.get('order_id')} has unexpected token {event_token}"
            )
        matched = _filled_shares(event, side=expected or event_side)
        if event_type == "trade" or _response_value(event, "trade_id", "tradeId", default=None) is not None:
            fill_id = _response_value(event, "trade_id", "tradeId", "id", default=None)
            if matched is None:
                matched = _response_value(event, "size", "amount", default=None)
            if matched is not None:
                economics = _trade_fill_economics(event)
                return self.directional_journal.add_fill(
                    trade_id, float(matched), str(fill_id) if fill_id else None,
                    price=economics["price"], fee_usd=economics["fee_usd"],
                )
        elif matched is not None:
            return self.directional_journal.add_fill(trade_id, float(matched))
        if event_type == "order":
            order_status = _response_value(event, "status", "state", default=None)
            if order_status is not None:
                return self.directional_journal.update(
                    trade_id, order_status=str(order_status).upper()
                )
        return self.directional_journal._record(trade_id)

    def _handle_negrisk_user_event(self, record: dict, event, event_type: str,
                                   order_id: str, token_id: str) -> dict:
        """Apply a user-stream fill to one NegRisk basket leg. Never guess by token."""
        journal = getattr(self, "negrisk_journal", None)
        if journal is None:
            raise UnhedgedPairError("NegRisk user event arrived without a NegRisk journal")
        event_side = str(_response_value(event, "side", default="") or "").upper()
        if event_side and event_side not in {"BUY", "SELL"}:
            raise UnhedgedPairError(
                f"NegRisk event {order_id} has unexpected side {event_side}"
            )
        legs = record.get("legs") or []
        leg = next((item for item in legs if str(item.get("order_id") or "") == str(order_id)), None)
        if leg is None:
            return record
        expected_token = str(leg.get("token_id") or "")
        if token_id and expected_token and token_id != expected_token:
            raise UnhedgedPairError(
                f"NegRisk event {order_id} has unexpected token {token_id}"
            )
        matched = _filled_shares(event, side=event_side or "BUY")
        if matched is None:
            matched = _response_value(event, "size", "amount", default=None)
        if matched is not None:
            journal.set_leg_order(
                record["basket_id"], expected_token, order_id, float(matched),
            )
        refreshed = journal._record(record["basket_id"])
        if refreshed.get("status") == "UNHEDGED":
            raise UnhedgedPairError(
                f"user stream reports an unfinished NegRisk basket: {refreshed.get('basket_id')}"
            )
        return refreshed

    def handle_user_event(self, event) -> Optional[dict]:
        """Apply one order/trade event and return the affected pair or directional trade."""
        if isinstance(event, dict):
            event = event.get("data", event)
        event_type = str(_response_value(event, "event_type", "type", default="")).lower()
        payload = _response_value(event, "payload", default=None)
        if payload is not None:
            event = payload
        order_id = _response_value(event, "order_id", "orderID", "orderId", "taker_order_id", "id", default="")
        token_id = str(_response_value(event, "token_id", "asset_id", "assetId", default=""))
        negrisk_journal = getattr(self, "negrisk_journal", None)
        if negrisk_journal is not None and order_id:
            basket = negrisk_journal.basket_for_order(order_id)
            if basket is not None:
                return self._handle_negrisk_user_event(basket, event, event_type, str(order_id), token_id)
        if self.directional_journal and order_id:
            directional = self.directional_journal.record_for_order(order_id)
            if directional is not None:
                return self._handle_directional_user_event(directional, event, event_type)
        if not self.journal:
            return None
        record = self.journal.pair_for_order(order_id) if order_id else None
        # A token can be present in multiple unrelated account orders. Never
        # attribute a user event by token alone; REST reconciliation can recover
        # an event whose order identifier was malformed or omitted.
        if record is None:
            return None
        event_market = _response_value(event, "condition_id", "market", default=None)
        if event_market is not None and str(event_market) != str(record["condition_id"]):
            raise UnhedgedPairError(
                f"user event {order_id} belongs to an unexpected condition"
            )
        event_side = str(_response_value(event, "side", default="")).upper()
        if event_side and event_side != "BUY":
            raise UnhedgedPairError(
                f"user event {order_id} has an unexpected side {event_side}"
            )
        leg = "yes" if token_id == record["yes_token_id"] or order_id == record.get("yes_order_id") else "no"
        matched = _filled_shares(event)
        if event_type == "trade" or _response_value(event, "trade_id", "tradeId", default=None) is not None:
            trade_id = _response_value(event, "trade_id", "tradeId", "id", default=None)
            if matched is None:
                matched = _response_value(event, "size", "amount", default=None)
            if matched is not None:
                economics = _trade_fill_economics(event)
                self.journal.add_fill(
                    record["pair_id"], leg, float(matched), str(trade_id) if trade_id else None,
                    price=economics["price"], fee_usd=economics["fee_usd"],
                    fee_rate_bps=economics["fee_rate_bps"], tx_hash=economics["tx_hash"],
                    timestamp_ms=economics["timestamp_ms"],
                )
        elif matched is not None:
            self.journal.add_fill(record["pair_id"], leg, float(matched))
        if event_type == "order":
            order_status = _response_value(event, "status", "state", default=None)
            if order_status is not None:
                self.journal.set_order_status(record["pair_id"], leg, str(order_status))
        status = self._update_pair_status(self.journal._record(record["pair_id"]))
        if status["status"] == "UNHEDGED":
            raise UnhedgedPairError(f"user stream reports an unhedged pair: {status['pair_id']}")
        return status

    async def _rollback_leg(self, token_id: str, shares: float, label: str,
                            pair_id: str = "", leg: str = "",
                            entry_price: Optional[float] = None) -> None:
        rollback = await self._call(
            "place_market_order",
            token_id=token_id,
            side="SELL",
            shares=f"{shares:.6f}",
            min_price=f"{self.rollback_min_price:.6f}",
            order_type="FAK",
        )
        if self.journal and pair_id and leg:
            self.journal.record_rollback(pair_id, leg, rollback, entry_price=entry_price)
        rollback_filled = _filled_shares(rollback, side="SELL")
        if (not _response_ok(rollback) or rollback_filled is None
                or rollback_filled + 1e-8 < shares):
            raise RuntimeError(f"{label} rollback was not fully confirmed")

    async def _rollback_yes(self, opportunity: ArbitrageOpportunity, pair_id: str,
                            yes_filled: float, cause: BaseException) -> None:
        try:
            await self._rollback_leg(
                opportunity.yes_token_id, yes_filled, "YES", pair_id, "yes",
                opportunity.yes_worst_price,
            )
        except Exception as rollback_error:
            if self.journal:
                self.journal.update(pair_id, rollback_status="FAILED", status="UNHEDGED",
                                    error=f"{cause}; {rollback_error}")
            raise UnhedgedPairError(
                f"YES leg requires rollback after the other leg failed; rollback was not fully confirmed"
            ) from cause
        if self.journal:
            self.journal.update(pair_id, rollback_status="CONFIRMED", status="ROLLED_BACK", error=str(cause))
        raise RuntimeError(f"YES leg was rolled back: {cause}") from cause

    async def _place_gtc_leg(self, token_id: str, price: float, shares: float) -> object:
        return await self._call(
            "place_limit_order",
            token_id=token_id,
            price=f"{price:.6f}",
            size=f"{shares:.6f}",
            side="BUY",
            post_only=True,
        )

    async def _cancel_gtc_order(self, order_id: str) -> None:
        if not order_id:
            return
        try:
            await self._call("cancel_order", order_id=order_id)
        except Exception:
            # A missing cancel is not success; later reconcile or halt must see it.
            raise

    async def _unwind_gtc_imbalance(self, record: dict) -> dict:
        """FAK-sell a one-sided GTC fill after cancel. Fail-closed if unwind is incomplete."""
        if not self.journal:
            raise UnhedgedPairError("GTC unwind requires a live journal")
        pair_id = str(record["pair_id"])
        current = self.journal._record(pair_id)
        requested = float(current.get("requested_shares", 0.0))
        yes = float(current.get("yes_matched_shares", 0.0))
        no = float(current.get("no_matched_shares", 0.0))
        if yes + 1e-8 >= requested and no + 1e-8 >= requested:
            return self.journal.set_status(pair_id, "HEDGED")
        errors = []
        if yes > 1e-8:
            try:
                await self._rollback_leg(
                    str(current["yes_token_id"]), yes, "YES", pair_id, "yes",
                )
            except Exception as error:
                errors.append(f"YES unwind: {error}")
        if no > 1e-8:
            try:
                await self._rollback_leg(
                    str(current["no_token_id"]), no, "NO", pair_id, "no",
                )
            except Exception as error:
                errors.append(f"NO unwind: {error}")
        if errors:
            self.journal.update(pair_id, rollback_status="FAILED", status="UNHEDGED",
                                error="; ".join(errors))
            raise UnhedgedPairError(
                f"GTC pair {pair_id} could not be fully unwound: " + "; ".join(errors)
            )
        if yes <= 1e-8 and no <= 1e-8:
            return self.journal.update(pair_id, rollback_status="NOT_REQUIRED", status="REJECTED")
        return self.journal.update(pair_id, rollback_status="CONFIRMED", status="ROLLED_BACK")

    async def _unwind_resting_gtc_pairs(self) -> None:
        if not self.journal:
            return
        for record in list(self.journal.incomplete_pairs()):
            if str(record.get("order_style", "")).upper() != "GTC":
                continue
            if record.get("status") not in {"RESTING", "PENDING", "UNHEDGED"}:
                continue
            yes = float(record.get("yes_matched_shares", 0.0))
            no = float(record.get("no_matched_shares", 0.0))
            requested = float(record.get("requested_shares", 0.0))
            if yes + 1e-8 >= requested and no + 1e-8 >= requested:
                self.journal.set_status(record["pair_id"], "HEDGED")
                continue
            if yes <= 1e-8 and no <= 1e-8:
                self.journal.set_status(record["pair_id"], "REJECTED", "halt cancelled resting GTC")
                continue
            await self._unwind_gtc_imbalance(record)

    async def execute_gtc(self, opportunity: ArbitrageOpportunity) -> LivePairResult:
        """Rest both binary legs as post-only GTC. Never fall back to FOK."""
        if opportunity.shares <= 0.0:
            raise ValueError("cannot execute an empty pair")
        await self.preflight(required_usd=opportunity.execution_capital_required)
        pair_id = self.journal.create_pair(opportunity) if self.journal else ""
        if self.journal:
            self.journal.update(pair_id, order_style="GTC", status="RESTING")
        tick = opportunity.tick_size if opportunity.tick_size > 0.0 else 0.01
        yes_price = maker_limit_price(opportunity.yes_worst_price, tick)
        no_price = maker_limit_price(opportunity.no_worst_price, tick)
        yes_order_id = ""
        no_order_id = ""
        try:
            yes = await self._place_gtc_leg(opportunity.yes_token_id, yes_price, opportunity.shares)
        except Exception as error:
            if self.journal:
                self.journal.update(
                    pair_id, status="UNHEDGED",
                    error=f"YES GTC outcome is unknown: {error}",
                )
            raise UnhedgedPairError(
                "YES GTC outcome is unknown; reconcile before placing new orders"
            ) from error
        yes_order_id = str(_response_value(yes, "order_id", "orderID", default="") or "")
        if not _response_ok(yes) and not yes_order_id:
            if self.journal:
                self.journal.set_status(
                    pair_id, "REJECTED",
                    str(_response_value(yes, "message", "errorMsg", default="YES post-only rejected")),
                )
            raise RuntimeError(
                f"YES GTC post-only rejected: "
                f"{_response_value(yes, 'message', 'errorMsg', default='unknown error')}"
            )
        if not yes_order_id:
            if self.journal:
                self.journal.update(pair_id, status="UNHEDGED", error="YES GTC response had no order ID")
            raise UnhedgedPairError("YES GTC response did not include an order ID")
        yes_filled = _filled_shares(yes, side="BUY") or 0.0
        if self.journal:
            self.journal.set_order_id(pair_id, "yes", yes_order_id)
            self.journal.set_order_status(pair_id, "yes", "OPEN")
            if yes_filled > 1e-8:
                self.journal.set_placement_fill(pair_id, "yes", yes_filled)
        try:
            no = await self._place_gtc_leg(opportunity.no_token_id, no_price, opportunity.shares)
        except Exception as error:
            await self._cancel_gtc_order(yes_order_id)
            if yes_filled > 1e-8:
                if self.journal:
                    self.journal.update(pair_id, status="UNHEDGED",
                                        error=f"NO GTC outcome is unknown: {error}")
                raise UnhedgedPairError(
                    "NO GTC outcome is unknown after YES may have filled"
                ) from error
            if self.journal:
                self.journal.update(pair_id, status="UNHEDGED",
                                    error=f"NO GTC outcome is unknown: {error}")
            raise UnhedgedPairError(
                "NO GTC outcome is unknown; reconcile before placing new orders"
            ) from error
        no_order_id = str(_response_value(no, "order_id", "orderID", default="") or "")
        if not _response_ok(no) and not no_order_id:
            await self._cancel_gtc_order(yes_order_id)
            if yes_filled > 1e-8:
                if self.journal:
                    self.journal.update(
                        pair_id, status="UNHEDGED",
                        error=str(_response_value(no, "message", "errorMsg", default="NO post-only rejected")),
                    )
                await self._unwind_gtc_imbalance(self.journal._record(pair_id))
            elif self.journal:
                self.journal.set_status(
                    pair_id, "REJECTED",
                    str(_response_value(no, "message", "errorMsg", default="NO post-only rejected")),
                )
            raise RuntimeError(
                f"NO GTC post-only rejected: "
                f"{_response_value(no, 'message', 'errorMsg', default='unknown error')}"
            )
        if not no_order_id:
            await self._cancel_gtc_order(yes_order_id)
            if self.journal:
                self.journal.update(pair_id, status="UNHEDGED", error="NO GTC response had no order ID")
            raise UnhedgedPairError("NO GTC response did not include an order ID")
        no_filled = _filled_shares(no, side="BUY") or 0.0
        if self.journal:
            self.journal.set_order_id(pair_id, "no", no_order_id)
            self.journal.set_order_status(pair_id, "no", "OPEN")
            if no_filled > 1e-8:
                self.journal.set_placement_fill(pair_id, "no", no_filled)
            self.journal.set_status(pair_id, "RESTING")
        return LivePairResult(
            pair_id=pair_id,
            yes_order_id=yes_order_id,
            no_order_id=no_order_id,
            shares=opportunity.shares,
            status="RESTING",
        )

    async def execute(self, opportunity: ArbitrageOpportunity) -> LivePairResult:
        if not hasattr(opportunity, "yes_token_id") or not hasattr(opportunity, "no_token_id"):
            raise ValueError("OfficialFOKExecutor only accepts binary Yes/No opportunities")
        if opportunity.shares <= 0.0:
            raise ValueError("cannot execute an empty pair")
        if str(getattr(opportunity, "order_style", "FOK")).upper() == "GTC" or not getattr(opportunity, "is_taker", True):
            return await self.execute_gtc(opportunity)
        await self.preflight(required_usd=opportunity.execution_capital_required)
        pair_id = self.journal.create_pair(opportunity) if self.journal else ""
        try:
            yes = await self._call(
                "place_market_order",
                token_id=opportunity.yes_token_id,
                side="BUY",
                amount=f"{opportunity.yes_execution_amount:.6f}",
                max_spend=f"{opportunity.yes_execution_amount + opportunity.yes_execution_fee_cap:.6f}",
                max_price=f"{opportunity.yes_worst_price:.6f}",
                order_type="FOK",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.journal:
                self.journal.update(
                    pair_id, status="UNHEDGED",
                    error=f"YES order outcome is unknown: {error}",
                )
            raise UnhedgedPairError(
                "YES order outcome is unknown; reconcile before placing new orders"
            ) from error
        if not _response_ok(yes):
            if self.journal:
                self.journal.set_status(pair_id, "REJECTED",
                                        str(_response_value(yes, "message", "errorMsg", default="YES FOK rejected")))
            raise RuntimeError(f"YES FOK rejected: {_response_value(yes, 'message', 'errorMsg', default='unknown error')}")
        yes_filled = _filled_shares(yes, side="BUY")
        if yes_filled is None or yes_filled + 1e-8 < opportunity.shares:
            if self.journal:
                if yes_filled is not None and yes_filled > 1e-8:
                    yes_order_id = str(_response_value(yes, "order_id", "orderID", default=""))
                    self.journal.set_order_id(pair_id, "yes", yes_order_id)
                    self.journal.set_placement_fill(pair_id, "yes", yes_filled)
            if yes_filled is not None and yes_filled > 1e-8:
                await self._rollback_yes(
                    opportunity, pair_id, yes_filled,
                    RuntimeError("YES response did not confirm a full FOK fill"),
                )
            if self.journal:
                self.journal.set_status(pair_id, "REJECTED", "YES response did not confirm a full FOK fill")
            raise RuntimeError("YES response did not confirm a full FOK fill")
        if yes_filled > opportunity.shares + 1e-8:
            await self._rollback_yes(
                opportunity, pair_id, yes_filled,
                RuntimeError("YES FOK filled more shares than the requested pair size"),
            )
        yes_order_id = str(_response_value(yes, "order_id", "orderID", default=""))
        if not yes_order_id:
            if yes_filled > 1e-8:
                await self._rollback_yes(
                    opportunity, pair_id, yes_filled,
                    RuntimeError("YES response did not include an order ID"),
                )
            if self.journal:
                self.journal.set_status(pair_id, "REJECTED", "YES response did not include an order ID")
            raise RuntimeError("YES response did not include an order ID")
        if self.journal:
            self.journal.set_order_id(pair_id, "yes", yes_order_id)
            self.journal.add_trade_ids(pair_id, "yes", _response_value(yes, "trade_ids", "tradeIDs", default=()))
            self.journal.add_transaction_hashes(
                pair_id, "yes", _response_value(yes, "transactions_hashes", "transactionsHashes", default=())
            )
            self.journal.set_placement_fill(pair_id, "yes", yes_filled)

        no_filled = 0.0
        no_response_received = False
        try:
            no = await self._call(
                "place_market_order",
                token_id=opportunity.no_token_id,
                side="BUY",
                amount=f"{opportunity.no_execution_amount:.6f}",
                max_spend=f"{opportunity.no_execution_amount + opportunity.no_execution_fee_cap:.6f}",
                max_price=f"{opportunity.no_worst_price:.6f}",
                order_type="FOK",
            )
            no_response_received = True
            reported_no_filled = _filled_shares(no, side="BUY")
            no_filled = reported_no_filled if reported_no_filled is not None else 0.0
            if (not _response_ok(no) or reported_no_filled is None
                    or reported_no_filled + 1e-8 < opportunity.shares):
                raise RuntimeError(
                    f"NO FOK rejected or incomplete: "
                    f"{_response_value(no, 'message', 'errorMsg', default='unknown error')}"
                )
            if no_filled > opportunity.shares + 1e-8:
                raise RuntimeError("NO FOK filled more shares than the requested pair size")
            no_order_id = str(_response_value(no, "order_id", "orderID", default=""))
            if not no_order_id:
                raise RuntimeError("NO response did not include an order ID")
            if self.journal:
                self.journal.set_order_id(pair_id, "no", no_order_id)
                self.journal.add_trade_ids(pair_id, "no", _response_value(no, "trade_ids", "tradeIDs", default=()))
                self.journal.add_transaction_hashes(
                    pair_id, "no", _response_value(no, "transactions_hashes", "transactionsHashes", default=())
                )
                self.journal.set_placement_fill(pair_id, "no", no_filled)
                self.journal.set_status(pair_id, "PENDING")
        except Exception as error:
            no_outcome_unknown = (
                not no_response_received
                or (_response_ok(no) and _filled_shares(no, side="BUY") is None)
            ) if no_response_received else True
            if no_outcome_unknown:
                try:
                    await self._rollback_leg(
                        opportunity.yes_token_id, yes_filled, "YES", pair_id, "yes",
                        opportunity.yes_worst_price,
                    )
                except Exception as rollback_error:
                    if self.journal:
                        self.journal.update(
                            pair_id, rollback_status="FAILED", status="UNHEDGED",
                            error=f"{error}; YES rollback: {rollback_error}",
                        )
                    raise UnhedgedPairError(
                        "NO order outcome is unknown and YES rollback was not confirmed"
                    ) from error
                if self.journal:
                    self.journal.update(
                        pair_id, rollback_status="CONFIRMED", status="UNHEDGED",
                        error=f"NO order outcome is unknown: {error}",
                    )
                raise UnhedgedPairError(
                    "NO order outcome is unknown; reconcile before placing new orders"
                ) from error
            if no_filled > 1e-8:
                try:
                    await self._rollback_leg(
                        opportunity.no_token_id, no_filled, "NO", pair_id, "no",
                        opportunity.no_worst_price,
                    )
                except Exception as rollback_error:
                    if self.journal:
                        self.journal.update(
                            pair_id, rollback_status="FAILED", status="UNHEDGED",
                            error=f"{error}; NO rollback: {rollback_error}",
                        )
                    raise UnhedgedPairError(
                        "NO leg requires rollback after the second-leg failure; rollback was not fully confirmed"
                    ) from error
            await self._rollback_yes(opportunity, pair_id, yes_filled, error)

        return LivePairResult(
            pair_id=pair_id,
            yes_order_id=yes_order_id,
            no_order_id=no_order_id,
            shares=opportunity.shares,
        )

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result


async def consume_user_stream(executor: OfficialFOKExecutor) -> None:
    """Forward official private order/trade events into the live journal."""
    try:
        from polymarket.streams import UserSpec
    except ImportError as error:
        raise RuntimeError("The official SDK does not provide UserSpec") from error
    subscribe = getattr(executor.client, "subscribe", None)
    if subscribe is None:
        raise RuntimeError("Official client does not expose subscribe")
    stream = subscribe(UserSpec())
    if inspect.isawaitable(stream):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        raise RuntimeError("Official user subscription is not an async iterator")
    executor.user_stream_healthy = True
    try:
        async for payload in stream:
            values = payload if isinstance(payload, list) else [payload]
            for event in values:
                executor.handle_user_event(event)
    finally:
        executor.user_stream_healthy = False
        close = getattr(stream, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result


def handle_market_event(event: dict, token_to_market: Dict[str, BinaryMarket],
                        books: Dict[str, OrderBook]) -> List[str]:
    """Apply one official market-channel event and return affected market IDs."""
    affected = set()
    event_type = _event_name(_response_value(event, "event_type", "type", default=""))
    payload = _response_value(event, "payload", default=None)
    if payload is not None:
        event = payload
    if event_type == "book":
        token_id = str(_response_value(event, "asset_id", "token_id", default=""))
        market = token_to_market.get(token_id)
        if market:
            if books.setdefault(token_id, OrderBook()).replace_snapshot(event):
                affected.add(market.market_id)
    elif event_type == "price_change":
        changes = _response_value(event, "price_changes", default=()) or ()
        for change in changes:
            token_id = str(_response_value(change, "asset_id", "token_id", default=""))
            market = token_to_market.get(token_id)
            if not market:
                continue
            try:
                price = float(_response_value(change, "price", default=None))
                size = float(_response_value(change, "size", default=None))
            except (TypeError, ValueError):
                continue
            applied = books.setdefault(token_id, OrderBook()).apply_change(
                _response_value(change, "side", default=""), price, size, event,
                level_hash=_response_value(change, "hash", default=None)
            )
            if applied:
                affected.add(market.market_id)
    return sorted(affected)
