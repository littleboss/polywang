#!/usr/bin/env python3
"""Independent NegRisk complete-set path.

This is not Yes/No combo arb. A mutually exclusive n-outcome field pays 1.00
to exactly one winner, so buying one share of every YES (or every NO, when
those tokens exist) is a different complete set. Sequential FOK across n legs
is not atomic; incomplete baskets halt rather than being forced through the
binary pair executor or the directional single-leg book.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .arbitrage_core import (
    BinaryMarket,
    OfficialFOKExecutor,
    OrderBook,
    UnhedgedPairError,
    _as_bool,
    _category,
    _filled_shares,
    _json_list,
    _response_ok,
    _response_value,
)
from .polymarket_edge import PolymarketFeeModel


def _gamma_end_ts(payload: dict) -> Optional[float]:
    raw_end = payload.get("endDate", payload.get("end_date"))
    if not raw_end:
        return None
    try:
        from datetime import datetime
        text = str(raw_end).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _gamma_fee_fields(payload: dict) -> Tuple[Optional[float], float, float, float]:
    """Return taker fee rate, exponent, min order size, tick size."""
    fee_schedule = payload.get("feeSchedule", payload.get("fee_schedule", {}))
    if isinstance(fee_schedule, str):
        try:
            import json
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
        return None, 1.0, float("nan"), float("nan")
    return fee_rate, fee_exponent, min_order_size, tick_size


def _is_yes_no_pair(outcomes: Sequence[object]) -> bool:
    names = {str(outcome).strip().lower() for outcome in outcomes}
    return names == {"yes", "no"}


@dataclass(frozen=True)
class NegRiskOutcome:
    name: str
    yes_token_id: str
    no_token_id: str = ""
    implied_yes: Optional[float] = None
    child_market_id: str = ""
    child_condition_id: str = ""


@dataclass(frozen=True)
class NegRiskMarket:
    """A complete mutually exclusive field, never a truncated volume-pool slice."""

    market_id: str
    condition_id: str
    title: str
    outcomes: Tuple[NegRiskOutcome, ...]
    category: str = "other"
    active: bool = True
    min_order_size: float = 0.0
    tick_size: float = 0.01
    taker_fee_rate: Optional[float] = None
    fee_exponent: float = 1.0
    fees_enabled: bool = True
    source: str = "nway"  # "nway" or "event"

    @property
    def yes_token_ids(self) -> Tuple[str, ...]:
        return tuple(outcome.yes_token_id for outcome in self.outcomes)

    @property
    def no_token_ids(self) -> Tuple[str, ...]:
        return tuple(outcome.no_token_id for outcome in self.outcomes if outcome.no_token_id)

    @property
    def token_ids(self) -> Tuple[str, ...]:
        tokens = list(self.yes_token_ids)
        tokens.extend(self.no_token_ids)
        return tuple(tokens)

    @property
    def has_no_tokens(self) -> bool:
        return len(self.no_token_ids) == len(self.outcomes) and all(self.no_token_ids)

    @property
    def implied_yes(self) -> Dict[str, float]:
        return {
            outcome.name: outcome.implied_yes
            for outcome in self.outcomes
            if outcome.implied_yes is not None
        }

    @classmethod
    def from_gamma(cls, payload: dict) -> Optional["NegRiskMarket"]:
        if not isinstance(payload, dict):
            return None
        nested = payload.get("markets")
        if isinstance(nested, list) and len(nested) >= 2:
            return cls.from_event(payload)
        tokens = [str(token).strip() for token in _json_list(
            payload.get("clobTokenIds", payload.get("clob_token_ids"))
        )]
        outcomes = [str(name).strip() for name in _json_list(payload.get("outcomes"))]
        if len(tokens) != len(outcomes) or len(tokens) < 2:
            return None
        if _is_yes_no_pair(outcomes):
            # Ordinary Yes/No binaries stay on the combo-arb path, even when
            # Gamma marks them negRisk. Those rows are one child of an event,
            # not a complete field.
            return None
        if any(not token or not name for token, name in zip(tokens, outcomes)):
            return None
        if len(set(tokens)) != len(tokens):
            return None
        prices = _json_list(payload.get("outcomePrices", payload.get("outcome_prices")))
        parsed_outcomes = []
        for index, (name, token) in enumerate(zip(outcomes, tokens)):
            implied = None
            if index < len(prices):
                try:
                    numeric = float(prices[index])
                except (TypeError, ValueError):
                    numeric = float("nan")
                if math.isfinite(numeric) and numeric >= 0.0:
                    implied = numeric
            parsed_outcomes.append(NegRiskOutcome(name=name, yes_token_id=token, implied_yes=implied))
        return cls._from_parts(payload, tuple(parsed_outcomes), source="nway")

    @classmethod
    def from_event(cls, payload: dict) -> Optional["NegRiskMarket"]:
        """Build a field from a Gamma event that already lists every child market."""
        nested = payload.get("markets")
        if not isinstance(nested, list) or len(nested) < 2:
            return None
        children: List[NegRiskOutcome] = []
        seen_tokens = set()
        for child in nested:
            if not isinstance(child, dict):
                return None
            binary = BinaryMarket.from_gamma(child)
            if binary is None or not binary.active:
                return None
            if binary.yes_token_id in seen_tokens or binary.no_token_id in seen_tokens:
                return None
            seen_tokens.add(binary.yes_token_id)
            seen_tokens.add(binary.no_token_id)
            label = str(child.get("groupItemTitle", child.get("group_item_title", binary.title))).strip()
            if not label:
                return None
            children.append(NegRiskOutcome(
                name=label,
                yes_token_id=binary.yes_token_id,
                no_token_id=binary.no_token_id,
                implied_yes=binary.implied_yes,
                child_market_id=binary.market_id,
                child_condition_id=binary.condition_id,
            ))
        if len(children) < 2:
            return None
        return cls._from_parts(payload, tuple(children), source="event")

    @classmethod
    def _from_parts(cls, payload: dict, outcomes: Tuple[NegRiskOutcome, ...],
                    source: str) -> Optional["NegRiskMarket"]:
        market_id = str(payload.get("id", payload.get("negRiskMarketID",
                         payload.get("conditionId", "")))).strip()
        condition_id = str(payload.get("negRiskMarketID", payload.get("conditionId",
                            payload.get("condition_id", payload.get("id", ""))))).strip()
        if not market_id or not condition_id:
            return None
        fee_rate, fee_exponent, min_order_size, tick_size = _gamma_fee_fields(payload)
        if not math.isfinite(min_order_size) or not math.isfinite(tick_size) or tick_size <= 0.0:
            min_order_size, tick_size = 0.0, 0.01
            if source == "nway":
                return None
        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tags = _json_list(tags)
        tag = tags[0] if tags else payload.get("category", "other")
        return cls(
            market_id=market_id,
            condition_id=condition_id,
            title=str(payload.get("title", payload.get("question", payload.get("slug", "")))),
            outcomes=outcomes,
            category=_category(payload.get("category", tag)),
            active=_as_bool(payload.get("active", True), True) and not _as_bool(payload.get("closed", False)),
            min_order_size=min_order_size if math.isfinite(min_order_size) else 0.0,
            tick_size=tick_size if math.isfinite(tick_size) and tick_size > 0.0 else 0.01,
            taker_fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            fees_enabled=_as_bool(payload.get("feesEnabled", payload.get("fees_enabled", True)), True),
            source=source,
        )


def event_lookup_key(row: dict) -> Optional[Tuple[str, str]]:
    """Return ('id', ...) or ('slug', ...) for a Gamma event fetch, else None."""
    if not isinstance(row, dict):
        return None
    events = row.get("events")
    if isinstance(events, str):
        events = _json_list(events)
    if isinstance(events, list):
        for item in events:
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("id") or "").strip()
            slug = str(item.get("slug") or item.get("ticker") or "").strip()
            if event_id:
                return ("id", event_id)
            if slug:
                return ("slug", slug)
    for name in ("eventId", "event_id"):
        value = str(row.get(name) or "").strip()
        if value:
            return ("id", value)
    slug = str(row.get("eventSlug") or row.get("event_slug") or "").strip()
    if slug:
        return ("slug", slug)
    return None


def collect_event_lookups(rows: Iterable[dict]) -> List[Tuple[str, str]]:
    """Keys for complete event fetches. Truncated pool rows are never grouped locally."""
    seen = set()
    keys: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if NegRiskMarket.from_gamma(row) is not None:
            continue
        binary = BinaryMarket.from_gamma(row)
        if binary is None or not binary.active or not binary.neg_risk:
            continue
        key = event_lookup_key(row)
        if key is None or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def fetch_complete_negrisk_events(
    lookups: Sequence[Tuple[str, str]],
    get_event: Callable[[str, str], Optional[dict]],
) -> List[dict]:
    """Fetch complete Gamma events. Never invent a field from truncated binaries."""
    events: List[dict] = []
    seen = set()
    for kind, value in lookups:
        key = (kind, value)
        if key in seen:
            continue
        seen.add(key)
        event = get_event(kind, value)
        if isinstance(event, dict):
            events.append(event)
    return events


def parse_negrisk_markets(rows: Iterable[dict],
                          extra_events: Optional[Iterable[dict]] = None) -> List[NegRiskMarket]:
    """Parse complete fields only. Scattered volume-pool binaries are ignored."""
    markets: List[NegRiskMarket] = []
    seen = set()
    for row in list(rows) + list(extra_events or []):
        parsed = NegRiskMarket.from_gamma(row) if isinstance(row, dict) else None
        if parsed is None or not parsed.active:
            continue
        if parsed.market_id in seen:
            continue
        seen.add(parsed.market_id)
        markets.append(parsed)
    return markets


@dataclass(frozen=True)
class NegRiskLeg:
    name: str
    token_id: str
    cost: float
    fee: float
    average_price: float
    worst_price: float
    execution_amount: float
    execution_fee_cap: float
    fills: Tuple[Tuple[float, float], ...] = ()


@dataclass
class NegRiskBookOpportunity:
    market_id: str
    condition_id: str
    title: str
    direction: str
    shares: float
    legs: Tuple[NegRiskLeg, ...]
    gross_profit: float
    net_profit: float
    return_on_capital: float
    execution_capital_required: float
    book_timestamp_ms: int
    fingerprint: str
    merge_gas_usd: float = 0.0
    is_risk_free: bool = False
    source: str = "nway"
    child_condition_ids: Tuple[str, ...] = ()
    residual_risk: str = (
        "sequential n-leg FOK is not atomic; a later-leg miss is unwound with "
        "FAK and may slip, partially fill, or remain open. Resolution redeem "
        "runs after the market resolves; pre-resolution convert stays off "
        "unless the official client exposes convert_positions."
    )

    @property
    def capital_required(self) -> float:
        return sum(leg.cost + leg.fee for leg in self.legs)

    @property
    def payout_per_share(self) -> float:
        if self.direction == "BUY_ALL_NO":
            return float(max(0, len(self.legs) - 1))
        return 1.0

    @property
    def token_ids(self) -> Tuple[str, ...]:
        return tuple(leg.token_id for leg in self.legs)


class NegRiskBookScanner:
    """Walk every outcome book and size a complete-set BUY.

    BUY_ALL_YES needs the YES token of every outcome. BUY_ALL_NO is offered
    only when every outcome also has a NO token; inventing the complement
    would be a directional guess, not a complete set.
    """

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

    def scan(self, market: NegRiskMarket, books: Dict[str, OrderBook],
             fee_model: Optional[PolymarketFeeModel] = None) -> Optional[NegRiskBookOpportunity]:
        if not market.active or len(market.outcomes) < 2:
            return None
        yes = self._scan_direction(market, books, "BUY_ALL_YES", fee_model)
        no = None
        if market.has_no_tokens:
            no = self._scan_direction(market, books, "BUY_ALL_NO", fee_model)
        candidates = [item for item in (yes, no) if item is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.net_profit)

    def _scan_direction(self, market: NegRiskMarket, books: Dict[str, OrderBook],
                        direction: str, fee_model: Optional[PolymarketFeeModel]
                        ) -> Optional[NegRiskBookOpportunity]:
        specs = []
        for outcome in market.outcomes:
            token_id = outcome.yes_token_id if direction == "BUY_ALL_YES" else outcome.no_token_id
            if not token_id:
                return None
            specs.append((outcome.name, token_id))
        synced_books: Dict[str, OrderBook] = {}
        levels_map: Dict[str, List[Tuple[float, float]]] = {}
        timestamps = []
        hashes = []
        for _, token_id in specs:
            book = books.get(token_id)
            if book is None or not book.synced or not book.timestamp_ms:
                return None
            levels = book.asks_sorted()
            if self.max_levels is not None:
                levels = levels[:self.max_levels]
            if not levels:
                return None
            synced_books[token_id] = book
            levels_map[token_id] = levels
            timestamps.append(book.timestamp_ms)
            hashes.append(book.hash or "")
        fee_model = fee_model or PolymarketFeeModel(
            market.category,
            taker_fee_rate=market.taker_fee_rate,
            fee_exponent=market.fee_exponent,
            fees_enabled=market.fees_enabled,
        )
        max_shares = min(sum(size for _, size in levels_map[token_id]) for _, token_id in specs)
        if max_shares <= 0.0:
            return None

        def execution_capital_for(shares: float) -> float:
            total = 0.0
            for _, token_id in specs:
                _, _, fills = _cost_for(synced_books[token_id], shares, levels_map[token_id])
                if not fills:
                    return float("inf")
                worst = fills[-1][0]
                fee_cap = shares * max(
                    (fee_model.fee_per_share(price, is_taker=True) for price, _ in levels_map[token_id]),
                    default=0.0,
                )
                total += shares * worst + fee_cap
            return total

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
            return None

        candidates = {max_shares}
        for _, token_id in specs:
            cumulative = 0.0
            for _, size in levels_map[token_id]:
                cumulative += size
                candidates.add(min(max_shares, cumulative))
                candidates.add(min(max_shares, size))

        best = None
        payout_per_share = 1.0 if direction == "BUY_ALL_YES" else float(len(specs) - 1)
        for shares in sorted(candidates):
            if shares <= 0.0:
                continue
            legs = []
            total_cost = 0.0
            total_fee = 0.0
            execution_capital = 0.0
            ok = True
            for name, token_id in specs:
                cost, average, fills = _cost_for(synced_books[token_id], shares, levels_map[token_id])
                if not fills:
                    ok = False
                    break
                fee = sum(fee_model.fee_usd(quantity, price, is_taker=True) for price, quantity in fills)
                worst = fills[-1][0]
                fee_cap = shares * max(
                    (fee_model.fee_per_share(price, is_taker=True) for price, _ in levels_map[token_id]),
                    default=0.0,
                )
                legs.append(NegRiskLeg(
                    name=name, token_id=token_id, cost=cost, fee=fee,
                    average_price=average, worst_price=worst,
                    execution_amount=shares * worst, execution_fee_cap=fee_cap,
                    fills=tuple(fills),
                ))
                total_cost += cost
                total_fee += fee
                execution_capital += shares * worst + fee_cap
            if not ok:
                continue
            gross = shares * payout_per_share - total_cost
            net = gross - total_fee - self.safety_buffer_usd - self.merge_gas_usd
            capital = total_cost + total_fee
            result = NegRiskBookOpportunity(
                market_id=market.market_id,
                condition_id=market.condition_id,
                title=market.title,
                direction=direction,
                shares=shares,
                legs=tuple(legs),
                gross_profit=gross,
                net_profit=net,
                return_on_capital=(net / capital if capital > 0 else 0.0),
                execution_capital_required=execution_capital,
                book_timestamp_ms=max(timestamps),
                fingerprint=f"{market.market_id}:{direction}:{':'.join(hashes)}:{shares:.12f}",
                merge_gas_usd=self.merge_gas_usd,
                is_risk_free=False,
                source=market.source,
                child_condition_ids=tuple(
                    outcome.child_condition_id or market.condition_id for outcome in market.outcomes
                ),
            )
            if result.net_profit >= self.min_net_profit_usd and result.return_on_capital >= self.min_return:
                if best is None or result.net_profit > best.net_profit:
                    best = result
        return best


def _cost_for(book: OrderBook, shares: float,
              levels: Optional[Sequence[Tuple[float, float]]] = None
              ) -> Tuple[float, float, List[Tuple[float, float]]]:
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


class LiveNegRiskJournal:
    """Atomic n-leg basket journal. Binary pairs stay on LiveOrderJournal."""

    TERMINAL_STATUSES = {"UNWOUND", "REJECTED", "SETTLED"}
    OPEN_STATUSES = {"PENDING", "PARTIAL", "ASSEMBLED", "UNHEDGED",
                     "RESOLVED_PENDING_REDEMPTION", "CONVERT_SUBMITTED"}
    STABLE_OPEN = {"ASSEMBLED", "RESOLVED_PENDING_REDEMPTION", "CONVERT_SUBMITTED"}

    def __init__(self, path: str):
        self.path = path
        self.state = {"baskets": {}, "events": []}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("baskets"), dict):
                raise ValueError("missing baskets")
            self.state.update(loaded)
            self.state.setdefault("events", [])
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"NegRisk journal is unreadable: {self.path}") from error

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".live-negrisk-", dir=directory, text=True)
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

    def create_basket(self, opportunity: NegRiskBookOpportunity) -> str:
        basket_id = f"nr:{opportunity.market_id}:{time.time_ns()}"
        now = time.time()
        legs = []
        for leg in opportunity.legs:
            legs.append({
                "name": leg.name,
                "token_id": leg.token_id,
                "order_id": "",
                "order_status": "",
                "requested_shares": float(opportunity.shares),
                "matched_shares": 0.0,
                "worst_price": float(leg.worst_price),
                "execution_amount": float(leg.execution_amount),
                "execution_fee_cap": float(leg.execution_fee_cap),
                "fill_details": {},
            })
        self.state["baskets"][basket_id] = {
            "basket_id": basket_id,
            "market_id": opportunity.market_id,
            "condition_id": opportunity.condition_id,
            "title": opportunity.title,
            "direction": opportunity.direction,
            "requested_shares": float(opportunity.shares),
            "capital_reserved": float(opportunity.execution_capital_required),
            "expected_net_profit": float(opportunity.net_profit),
            "payout_per_share": float(opportunity.payout_per_share),
            "source": opportunity.source,
            "child_condition_ids": list(opportunity.child_condition_ids),
            "legs": legs,
            "status": "PENDING",
            "rollback_status": "NOT_REQUIRED",
            "created_at": now,
            "updated_at": now,
            "error": "",
        }
        self.save()
        return basket_id

    def _record(self, basket_id: str) -> dict:
        try:
            return self.state["baskets"][basket_id]
        except KeyError as error:
            raise KeyError(f"unknown NegRisk basket: {basket_id}") from error

    def update(self, basket_id: str, **changes) -> dict:
        record = self._record(basket_id)
        record.update(changes)
        record["updated_at"] = time.time()
        self.save()
        return record

    def _leg(self, basket_id: str, token_id: str) -> dict:
        record = self._record(basket_id)
        for leg in record["legs"]:
            if str(leg.get("token_id")) == str(token_id):
                return leg
        raise KeyError(f"unknown NegRisk leg {token_id} on {basket_id}")

    def set_leg_order(self, basket_id: str, token_id: str, order_id: str,
                      matched_shares: float, status: str = "") -> dict:
        leg = self._leg(basket_id, token_id)
        leg["order_id"] = str(order_id)
        leg["matched_shares"] = float(matched_shares)
        if status:
            leg["order_status"] = str(status).upper()
        return self.update(basket_id)

    def set_status(self, basket_id: str, status: str, error: str = "") -> dict:
        return self.update(basket_id, status=status, error=error)

    def incomplete_baskets(self) -> List[dict]:
        return [record for record in self.state["baskets"].values()
                if record.get("status") not in self.TERMINAL_STATUSES]

    def incomplete_trades(self) -> List[dict]:
        """Risk duck-type: unfinished baskets, excluding fully assembled sets."""
        return [record for record in self.incomplete_baskets()
                if record.get("status") not in self.STABLE_OPEN]

    def open_exposure(self) -> float:
        total = 0.0
        for record in self.incomplete_baskets():
            reserved = float(record.get("capital_reserved", 0.0))
            if not math.isfinite(reserved) or reserved < 0.0:
                raise RuntimeError(
                    f"NegRisk basket {record.get('basket_id', '')} has invalid capital reservation"
                )
            total += reserved
        return total

    def market_exposure(self, market_id: str) -> float:
        total = 0.0
        for record in self.incomplete_baskets():
            if str(record.get("market_id")) != str(market_id):
                continue
            reserved = float(record.get("capital_reserved", 0.0))
            if not math.isfinite(reserved) or reserved < 0.0:
                raise RuntimeError(
                    f"NegRisk basket {record.get('basket_id', '')} has invalid capital reservation"
                )
            total += reserved
        return total

    def known_order_ids(self) -> set:
        known = set()
        for record in self.state["baskets"].values():
            for leg in record.get("legs") or []:
                order_id = str(leg.get("order_id") or "")
                if order_id:
                    known.add(order_id)
        return known

    def known_inventory_token_ids(self) -> set:
        known = set()
        for record in self.incomplete_baskets():
            for leg in record.get("legs") or []:
                if float(leg.get("matched_shares") or 0.0) > 1e-8:
                    token_id = str(leg.get("token_id") or "")
                    if token_id:
                        known.add(token_id)
        return known

    def basket_for_order(self, order_id: str) -> Optional[dict]:
        order_id = str(order_id)
        if not order_id:
            return None
        for record in self.state["baskets"].values():
            for leg in record.get("legs") or []:
                if str(leg.get("order_id") or "") == order_id:
                    return record
        return None

    def integrity_issues(self) -> List[str]:
        issues = []
        seen_orders = {}
        for record in self.state["baskets"].values():
            basket_id = str(record.get("basket_id", ""))
            if not record.get("condition_id"):
                issues.append(f"{basket_id} has no condition id")
            if not record.get("direction"):
                issues.append(f"{basket_id} has no direction")
            legs = record.get("legs")
            if not isinstance(legs, list) or len(legs) < 2:
                issues.append(f"{basket_id} does not list a complete set")
                continue
            try:
                requested = float(record.get("requested_shares", 0.0))
            except (TypeError, ValueError):
                requested = float("nan")
            if not math.isfinite(requested) or requested <= 0.0:
                issues.append(f"{basket_id} has invalid requested shares")
            try:
                reserved = float(record.get("capital_reserved", 0.0))
            except (TypeError, ValueError):
                reserved = float("nan")
            if not math.isfinite(reserved) or reserved < 0.0:
                issues.append(f"{basket_id} has invalid capital reservation")
            tokens = []
            for leg in legs:
                if not isinstance(leg, dict) or not leg.get("token_id"):
                    issues.append(f"{basket_id} has a leg without a token id")
                    continue
                tokens.append(str(leg["token_id"]))
                try:
                    matched = float(leg.get("matched_shares", 0.0))
                except (TypeError, ValueError):
                    matched = float("nan")
                if not math.isfinite(matched) or matched < 0.0:
                    issues.append(f"{basket_id} has invalid matched shares")
                order_id = str(leg.get("order_id") or "")
                if order_id:
                    previous = seen_orders.get(order_id)
                    if previous and previous != basket_id:
                        issues.append(f"order {order_id} belongs to multiple NegRisk baskets")
                    seen_orders[order_id] = basket_id
            if len(set(tokens)) != len(tokens):
                issues.append(f"{basket_id} repeats a token id")
        return issues

    def summary(self) -> dict:
        by_status: Dict[str, int] = {}
        unfinished = []
        for record in self.state["baskets"].values():
            status = str(record.get("status", "UNKNOWN"))
            by_status[status] = by_status.get(status, 0) + 1
            if status not in self.TERMINAL_STATUSES and status not in self.STABLE_OPEN:
                unfinished.append(str(record.get("basket_id", "")))
        return {
            "baskets": len(self.state["baskets"]),
            "by_status": dict(sorted(by_status.items())),
            "open_exposure": self.open_exposure(),
            "unfinished_baskets": unfinished,
        }

    def mark_resolved(self, market_id: str, winning: str) -> int:
        marked = 0
        for record in self.incomplete_baskets():
            if str(record.get("market_id")) != str(market_id) and str(record.get("condition_id")) != str(market_id):
                continue
            if record.get("status") not in {"ASSEMBLED", "CONVERT_SUBMITTED"}:
                continue
            self.update(
                record["basket_id"],
                status="RESOLVED_PENDING_REDEMPTION",
                winning_outcome=str(winning),
                settlement_type="MARKET_RESOLUTION",
            )
            marked += 1
        return marked

    def mark_settled(self, basket_id: str, transaction_hash: str, *,
                     settlement_type: str, realized_pnl: Optional[float] = None) -> dict:
        record = self._record(basket_id)
        reserved = float(record.get("capital_reserved", 0.0))
        payout = float(record.get("requested_shares", 0.0)) * float(record.get("payout_per_share", 1.0))
        pnl = payout - reserved if realized_pnl is None else float(realized_pnl)
        return self.update(
            basket_id,
            status="SETTLED",
            settlement_type=settlement_type,
            settlement_tx_hash=str(transaction_hash),
            realized_pnl=pnl,
            capital_reserved=0.0,
        )


class PaperNegRiskExecutor:
    def __init__(self, journal: LiveNegRiskJournal, ledger=None):
        self.journal = journal
        self.ledger = ledger

    def execute(self, opportunity: NegRiskBookOpportunity):
        if opportunity.shares <= 0.0:
            raise ValueError("cannot execute an empty NegRisk basket")
        if self.ledger is not None:
            required = opportunity.capital_required
            cash = float(self.ledger.state["cash"])
            if required > cash + 1e-9:
                raise ValueError("insufficient paper cash")
            self.ledger.state["cash"] = cash - required
            self.ledger.state["trades"].append({
                "type": "OPEN_NEGRISK",
                "market_id": opportunity.market_id,
                "direction": opportunity.direction,
                "shares": opportunity.shares,
                "cost": opportunity.capital_required,
            })
            position_id = f"nr:{opportunity.market_id}:{int(time.time() * 1_000_000)}"
            self.ledger.state["positions"][position_id] = {
                "position_id": position_id,
                "market_id": opportunity.market_id,
                "title": opportunity.title,
                "shares": opportunity.shares,
                "cost": opportunity.capital_required,
                "fees": sum(leg.fee for leg in opportunity.legs),
                "opened_at": time.time(),
                "settled": False,
                "payout": 0.0,
                "payout_per_share": opportunity.payout_per_share,
                "kind": "negrisk",
                "direction": opportunity.direction,
            }
            self.ledger.save()
        basket_id = self.journal.create_basket(opportunity)
        for index, leg in enumerate(opportunity.legs):
            self.journal.set_leg_order(
                basket_id, leg.token_id, f"paper-nr-{index}", opportunity.shares, status="FILLED",
            )
        self.journal.set_status(basket_id, "ASSEMBLED")
        return SimpleNamespace(
            basket_id=basket_id,
            shares=opportunity.shares,
            direction=opportunity.direction,
            status="ASSEMBLED",
            order_ids=tuple(f"paper-nr-{index}" for index in range(len(opportunity.legs))),
        )


@dataclass
class LiveNegRiskResult:
    basket_id: str
    shares: float
    direction: str
    status: str
    order_ids: Tuple[str, ...] = ()


class OfficialNegRiskExecutor:
    """Sequential FOK across a complete NegRisk field, then FAK unwind on miss.

    Resolution redeem uses official ``redeem_positions`` per child condition.
    Pre-resolution convert calls ``convert_positions`` only when the SDK
    exposes it; polymarket-client 0.6.0 does not, so AUTO_CONVERT stays off.
    """

    def __init__(self, transport: OfficialFOKExecutor, journal: LiveNegRiskJournal,
                 auto_convert: Optional[bool] = None, auto_redeem: Optional[bool] = None):
        self.transport = transport
        self.journal = journal
        transport.negrisk_journal = journal
        transport.negrisk_executor = self
        if auto_convert is None:
            auto_convert = os.getenv("AUTO_CONVERT_NEGRISK", "0") == "1"
        if auto_redeem is None:
            auto_redeem = os.getenv("AUTO_REDEEM_RESOLVED_POSITIONS", "1") == "1"
        self.auto_convert = bool(auto_convert)
        self.auto_redeem = bool(auto_redeem)

    async def execute(self, opportunity: NegRiskBookOpportunity) -> LiveNegRiskResult:
        if opportunity.shares <= 0.0:
            raise ValueError("cannot execute an empty NegRisk basket")
        await self.transport.preflight(required_usd=opportunity.execution_capital_required)
        basket_id = self.journal.create_basket(opportunity)
        filled: List[Tuple[NegRiskLeg, float, str]] = []
        try:
            for leg in opportunity.legs:
                response_received = False
                response = None
                try:
                    response = await self.transport._call(
                        "place_market_order",
                        token_id=leg.token_id,
                        side="BUY",
                        amount=f"{leg.execution_amount:.6f}",
                        max_spend=f"{leg.execution_amount + leg.execution_fee_cap:.6f}",
                        max_price=f"{leg.worst_price:.6f}",
                        order_type="FOK",
                    )
                    response_received = True
                except Exception as error:
                    await self._unwind_filled(
                        basket_id, filled, error, unknown=True,
                        extra=f"{leg.name} order outcome is unknown: {error}",
                    )
                reported = _filled_shares(response, side="BUY")
                matched = reported if reported is not None else 0.0
                order_id = str(_response_value(response, "order_id", "orderID", default="") or "")
                if (not _response_ok(response) or reported is None
                        or reported + 1e-8 < opportunity.shares):
                    if matched > 1e-8:
                        filled.append((leg, matched, order_id))
                        if order_id:
                            self.journal.set_leg_order(basket_id, leg.token_id, order_id, matched)
                    await self._unwind_filled(
                        basket_id, filled,
                        RuntimeError(f"{leg.name} FOK rejected or incomplete"),
                        unknown=not response_received or (_response_ok(response) and reported is None),
                    )
                if matched > opportunity.shares + 1e-8:
                    filled.append((leg, matched, order_id))
                    if order_id:
                        self.journal.set_leg_order(basket_id, leg.token_id, order_id, matched)
                    await self._unwind_filled(
                        basket_id, filled,
                        RuntimeError(f"{leg.name} FOK filled more shares than requested"),
                        unknown=False,
                    )
                if not order_id:
                    filled.append((leg, matched, ""))
                    await self._unwind_filled(
                        basket_id, filled,
                        RuntimeError(f"{leg.name} response did not include an order ID"),
                        unknown=True,
                    )
                self.journal.set_leg_order(basket_id, leg.token_id, order_id, matched, status="FILLED")
                filled.append((leg, matched, order_id))
        except (UnhedgedPairError, RuntimeError):
            raise
        self.journal.set_status(basket_id, "ASSEMBLED")
        return LiveNegRiskResult(
            basket_id=basket_id,
            shares=opportunity.shares,
            direction=opportunity.direction,
            status="ASSEMBLED",
            order_ids=tuple(order_id for _, _, order_id in filled if order_id),
        )

    async def _unwind_filled(self, basket_id: str,
                             filled: Sequence[Tuple[NegRiskLeg, float, str]],
                             cause: BaseException, unknown: bool,
                             extra: str = "") -> None:
        errors = [extra or str(cause)]
        for leg, shares, _order_id in reversed(list(filled)):
            if shares <= 1e-8:
                continue
            try:
                await self.transport._rollback_leg(leg.token_id, shares, leg.name)
            except Exception as rollback_error:
                errors.append(f"{leg.name} rollback: {rollback_error}")
                self.journal.update(
                    basket_id, rollback_status="FAILED", status="UNHEDGED",
                    error="; ".join(errors),
                )
                raise UnhedgedPairError(
                    "NegRisk basket requires unwind after a later-leg failure; "
                    "rollback was not fully confirmed"
                ) from cause
        status = "UNHEDGED" if unknown else "UNWOUND"
        self.journal.update(
            basket_id, rollback_status="CONFIRMED", status=status, error="; ".join(errors),
        )
        if unknown:
            raise UnhedgedPairError(
                "NegRisk order outcome is unknown; reconcile before placing new orders"
            ) from cause
        raise RuntimeError(f"NegRisk basket was unwound: {cause}") from cause

    def _redeem_condition_ids(self, record: dict) -> List[str]:
        raw = record.get("child_condition_ids") or [record.get("condition_id")]
        seen = set()
        ids = []
        for item in raw:
            value = str(item or "").strip()
            if value and value not in seen:
                seen.add(value)
                ids.append(value)
        return ids

    async def convert_basket(self, record: dict) -> dict:
        """Convert an assembled complete set if the official client exposes it."""
        basket_id = str(record["basket_id"])
        current = self.journal._record(basket_id)
        if current.get("status") == "SETTLED":
            return current
        if current.get("status") != "ASSEMBLED":
            raise RuntimeError(f"basket {basket_id} is not assembled")
        if current.get("settlement_type") == "CONVERT_SUBMITTED":
            raise RuntimeError(f"basket {basket_id} has a submitted convert requiring reconciliation")
        method_name = next(
            (name for name in ("convert_positions", "convertPositions")
             if getattr(self.transport.client, name, None) is not None),
            "",
        )
        if not method_name:
            raise RuntimeError(
                "Official client does not expose convert_positions; "
                "keep AUTO_CONVERT_NEGRISK=0 and redeem after resolution"
            )
        from decimal import Decimal, ROUND_DOWN
        requested = Decimal(str(current.get("requested_shares", 0.0)))
        amount = int((requested * Decimal(1_000_000)).to_integral_value(rounding=ROUND_DOWN))
        if amount <= 0:
            raise RuntimeError(f"basket {basket_id} has no convertible base-unit amount")
        self.journal.update(basket_id, settlement_type="CONVERT_PENDING")
        handle = await self.transport._call(
            method_name,
            condition_id=str(current["condition_id"]),
            amount=amount,
            token_ids=[str(leg.get("token_id")) for leg in current.get("legs") or []],
        )
        transaction_id = str(_response_value(handle, "transaction_id", "transactionID", default="") or "")
        submitted_hash = self.transport._transaction_hash(handle)
        self.journal.update(
            basket_id,
            status="CONVERT_SUBMITTED",
            settlement_type="CONVERT_SUBMITTED",
            settlement_tx_id=transaction_id,
            settlement_tx_hash=submitted_hash,
        )
        wait = getattr(handle, "wait", None)
        if wait is None:
            raise RuntimeError("convert transaction handle has no wait method")
        outcome = wait()
        outcome = await outcome if inspect.isawaitable(outcome) else outcome
        transaction_hash = self.transport._transaction_hash(outcome) or submitted_hash
        if not transaction_hash:
            raise RuntimeError("convert transaction completed without a transaction hash")
        return self.journal.mark_settled(basket_id, transaction_hash, settlement_type="CONVERT")

    async def redeem_basket(self, record: dict) -> dict:
        """Redeem resolved child conditions. Losing legs may have zero balance."""
        basket_id = str(record["basket_id"])
        current = self.journal._record(basket_id)
        if current.get("status") == "SETTLED":
            return current
        if current.get("status") != "RESOLVED_PENDING_REDEMPTION":
            raise RuntimeError(f"basket {basket_id} is not pending redemption")
        if current.get("settlement_type") == "REDEEM_SUBMITTED":
            raise RuntimeError(f"basket {basket_id} has a submitted redemption requiring reconciliation")
        if getattr(self.transport.client, "redeem_positions", None) is None:
            raise RuntimeError("Official client does not expose redeem_positions")
        hashes = []
        errors = []
        last_handle = None
        self.journal.update(basket_id, settlement_type="REDEEM_PENDING")
        for condition_id in self._redeem_condition_ids(current):
            try:
                handle = await self.transport._call("redeem_positions", condition_id=condition_id)
            except Exception as error:
                errors.append(f"{condition_id}: {error}")
                continue
            last_handle = handle
            submitted_hash = self.transport._transaction_hash(handle)
            if submitted_hash:
                hashes.append(submitted_hash)
        if last_handle is None:
            raise RuntimeError(
                f"basket {basket_id} redemption failed on every condition: " + "; ".join(errors)
            )
        transaction_id = str(_response_value(last_handle, "transaction_id", "transactionID", default="") or "")
        self.journal.update(
            basket_id,
            settlement_type="REDEEM_SUBMITTED",
            settlement_tx_id=transaction_id,
            settlement_tx_hash=hashes[-1] if hashes else "",
        )
        wait = getattr(last_handle, "wait", None)
        if wait is None:
            raise RuntimeError("redemption transaction handle has no wait method")
        outcome = wait()
        outcome = await outcome if inspect.isawaitable(outcome) else outcome
        transaction_hash = self.transport._transaction_hash(outcome) or (hashes[-1] if hashes else "")
        if not transaction_hash:
            raise RuntimeError("redemption transaction completed without a transaction hash")
        return self.journal.mark_settled(basket_id, transaction_hash, settlement_type="REDEEM")

    async def settle_baskets(self) -> List[dict]:
        settled = []
        for record in list(self.journal.incomplete_baskets()):
            action = None
            if record.get("status") == "ASSEMBLED" and self.auto_convert:
                action = self.convert_basket
            elif record.get("status") == "RESOLVED_PENDING_REDEMPTION" and self.auto_redeem:
                action = self.redeem_basket
            if action is None:
                continue
            try:
                settled.append(await action(record))
            except Exception as error:
                self.journal.update(record["basket_id"], error=str(error))
        return settled


def negrisk_execution_enabled(live: bool) -> bool:
    """Paper defaults on when ENABLE_NEGRISK_EXECUTION is unset.

    Live stays fail-closed: it needs an explicit execution flag *and*
    ENABLE_NEGRISK_LIVE. That live flag is never treated as on when unset.
    """
    raw = os.getenv("ENABLE_NEGRISK_EXECUTION")
    if raw is None or str(raw).strip() == "":
        enabled = not live
    else:
        enabled = str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    if live and os.getenv("ENABLE_NEGRISK_LIVE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return True
