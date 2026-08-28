#!/usr/bin/env python3
"""Run the deterministic binary-market paper/live arbitrage scanner.

Paper mode is the default. Live mode remains explicitly fail-closed and uses
only the official SDK plus the journal and reconciliation checks.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import signal
import time
from typing import Dict, Iterable, List, Optional

import requests
from .whale_intelligence import WhaleIntelligenceEngine
from .sports_channel import (
    SportsStateTracker, SportsLatencyGate, SportsMarketMap,
    consume_sports_channel, evaluate_sports_candidate,
)
from .market_replay import JsonlEventRecorder
from .macro_model import JsonlMacroFeed, MacroEventModel, MacroRelease
from .crypto_model import CryptoObservation, CryptoStatArbModel, JsonlCryptoFeed
from .polymarket_edge import (
    CalibrationTracker, EdgeEvaluator, NegRiskScanner,
    combo_arb_universe_score, merge_gas_startup_warning, rank_combo_arb_markets,
)
from .negrisk import (
    LiveNegRiskJournal,
    NegRiskBookScanner,
    NegRiskMarket,
    OfficialNegRiskExecutor,
    PaperNegRiskExecutor,
    negrisk_execution_enabled,
    parse_negrisk_markets,
)
from .arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    DirectionalExecutor,
    DirectionalIntent,
    JsonLedger,
    LiveDirectionalJournal,
    LiveOrderJournal,
    OrderBook,
    PaperArbitrageExecutor,
    PaperDirectionalExecutor,
    OfficialFOKExecutor,
    LiveRiskController,
    RiskHaltError,
    UnhedgedPairError,
    handle_market_event,
    consume_user_stream,
    intent_from_best_ask,
    intent_from_inventory_bid,
    _event_name,
    _gamma_outcome_prices,
)


LOG = logging.getLogger("arbitrage-bot")
GAMMA_URL = "https://gamma-api.polymarket.com/markets"
MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw else default
        return value if math.isfinite(value) else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: str = "") -> None:
    """Load KEY=VALUE pairs from a local .env without overriding the process env."""
    dotenv_path = path or os.getenv("DOTENV_PATH", ".env")
    if not dotenv_path or not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def write_health(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    body = dict(payload)
    body["updated_at"] = time.time()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fetch_markets(limit: int, *, get=None, pool: Optional[int] = None) -> List[BinaryMarket]:
    """Fetch a volume-ordered pool, then keep the binary markets closest to combo-arb breakeven."""
    binary, _negrisk = fetch_universe(limit, get=get, pool=pool, negrisk_limit=0)
    return binary


def fetch_universe(limit: int, *, get=None, pool: Optional[int] = None,
                   negrisk_limit: int = 0) -> tuple:
    """Return ranked binary combo-arb markets plus complete NegRisk fields.

    NegRisk rows are never mixed into the Yes/No FOK universe. Scattered
    volume-pool binaries that happen to be marked negRisk are also excluded
    because a truncated field is not a complete set.
    """
    getter = get or _http_get_gamma
    keep = max(1, int(limit))
    pool_size = int(pool if pool is not None else env_int("MARKET_SCAN_POOL", max(keep * 5, 100)))
    pool_size = max(keep, pool_size)
    rows = _fetch_gamma_rows(pool_size, getter)
    negrisk = parse_negrisk_markets(rows)
    negrisk_ids = {market.market_id for market in negrisk}
    binary: List[BinaryMarket] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = BinaryMarket.from_gamma(row)
        if parsed and parsed.active and not parsed.neg_risk:
            binary.append(parsed)
        elif env_bool("ENABLE_NEGRISK_OBSERVE", True):
            market_id = str(row.get("id", row.get("conditionId", "")) or "")
            if market_id not in negrisk_ids:
                _log_negrisk_observation(row)
    if env_bool("ENABLE_NEGRISK_OBSERVE", True):
        for market in negrisk:
            _log_negrisk_market(market)
    ranked = rank_combo_arb_markets(binary)
    selected = ranked[:keep]
    if selected:
        head = selected[0]
        score = combo_arb_universe_score(
            head.category, head.implied_yes, tick_size=head.tick_size,
            taker_fee_rate=head.taker_fee_rate, fee_exponent=head.fee_exponent,
        )
        LOG.info(
            "Universe: %d binary candidates ranked, keeping %d; "
            "top %s ticks_to_breakeven=%d one_tick_net=%.4f implied_yes=%s",
            len(binary), len(selected), head.category, score.ticks_to_breakeven,
            score.one_tick_net,
            "unknown" if head.implied_yes is None else f"{head.implied_yes:.3f}",
        )
    keep_negrisk = max(0, int(negrisk_limit))
    selected_negrisk = negrisk[:keep_negrisk]
    if selected_negrisk:
        LOG.info(
            "NegRisk universe: %d complete fields parsed, keeping %d for the independent path",
            len(negrisk), len(selected_negrisk),
        )
    return selected, selected_negrisk


def _http_get_gamma(params: dict) -> list:
    response = requests.get(GAMMA_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    return rows if isinstance(rows, list) else []


def _fetch_gamma_rows(pool: int, getter) -> list:
    page_size = min(100, max(1, int(pool)))
    rows: list = []
    offset = 0
    while len(rows) < pool:
        batch = getter({
            "closed": "false",
            "active": "true",
            "limit": min(page_size, pool - len(rows)),
            "offset": offset,
            "order": "volume_24hr",
            "ascending": "false",
        })
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(item for item in batch if isinstance(item, dict))
        offset += len(batch)
        if len(batch) < page_size:
            break
    return rows[:pool]


def _log_negrisk_observation(row: dict) -> None:
    """Log a NegRisk complete-set dislocation. Never route it to the binary FOK executor."""
    prices = _gamma_outcome_prices(row)
    if not prices:
        return
    market_id = str(row.get("id", row.get("conditionId", "")) or "")
    opportunity = NegRiskScanner(min_net_margin=0.01).scan(market_id, prices)
    if opportunity is None or not opportunity.tradeable:
        return
    LOG.info(
        "NEGRISK OBSERVE: %s | %s | %s | not routed to the binary FOK executor",
        opportunity.market_id, opportunity.direction, opportunity.note,
    )


def _log_negrisk_market(market: NegRiskMarket) -> None:
    prices = market.implied_yes
    if len(prices) < 2:
        return
    opportunity = NegRiskScanner(min_net_margin=0.01).scan(market.market_id, prices)
    if opportunity is None or not opportunity.tradeable:
        return
    LOG.info(
        "NEGRISK OBSERVE: %s | %s | %s | independent path, not binary FOK",
        market.market_id, opportunity.direction, opportunity.note,
    )


def _events(payload) -> Iterable[dict]:
    values = payload if isinstance(payload, list) else [payload]
    return (value for value in values if isinstance(value, dict))


class PaperMarketRunner:
    def __init__(self, markets: List[BinaryMarket], ledger_path: str,
                 initial_cash: float, scanner: BinaryArbitrageScanner, executor=None,
                 risk_controller: Optional[LiveRiskController] = None,
                 negrisk_markets: Optional[List[NegRiskMarket]] = None,
                 negrisk_scanner: Optional[NegRiskBookScanner] = None,
                 negrisk_executor=None):
        seen_identifiers = {}
        for market in markets:
            for kind, value in (
                ("market", market.market_id), ("condition", market.condition_id),
                ("token", market.yes_token_id), ("token", market.no_token_id),
            ):
                key = (kind, str(value))
                owner = seen_identifiers.get(key)
                if owner is not None:
                    raise ValueError(
                        f"duplicate {kind} identifier {value!r} across markets "
                        f"{owner!r} and {market.market_id!r}"
                    )
                seen_identifiers[key] = market.market_id
        self.negrisk_markets: Dict[str, NegRiskMarket] = {
            market.market_id: market for market in (negrisk_markets or [])
        }
        for market in self.negrisk_markets.values():
            for kind, value in (
                [("market", market.market_id), ("condition", market.condition_id)]
                + [("token", token_id) for token_id in market.token_ids]
            ):
                key = (kind, str(value))
                owner = seen_identifiers.get(key)
                if owner is not None:
                    raise ValueError(
                        f"duplicate {kind} identifier {value!r} across markets "
                        f"{owner!r} and {market.market_id!r}"
                    )
                seen_identifiers[key] = market.market_id
        self.markets: Dict[str, BinaryMarket] = {market.market_id: market for market in markets}
        self.token_to_market = {
            token: market
            for market in markets
            for token in (market.yes_token_id, market.no_token_id)
        }
        for market in self.negrisk_markets.values():
            for token_id in market.token_ids:
                self.token_to_market[token_id] = market
        self.books: Dict[str, OrderBook] = {}
        self.scanner = scanner
        self.negrisk_scanner = negrisk_scanner
        self.negrisk_executor = negrisk_executor
        self.negrisk_journal = getattr(negrisk_executor, "journal", None)
        self.ledger = None if executor else JsonLedger(ledger_path, initial_cash=initial_cash)
        self.executor = executor or PaperArbitrageExecutor(
            self.ledger,
            max_total_exposure_fraction=env_float("MAX_TOTAL_EXPOSURE_FRACTION", 0.25),
            max_market_exposure_fraction=env_float("MAX_MARKET_EXPOSURE_FRACTION", 0.05),
        )
        self.live = executor is not None
        self.live_journal = getattr(executor, "journal", None)
        self.risk_controller = risk_controller
        self.whale_engine = WhaleIntelligenceEngine(
            path=os.getenv("WHALE_STATE_PATH", "whale-intelligence.json"),
            threshold_usd=env_float("WHALE_THRESHOLD_USD", 5000.0),
            window_seconds=env_float("COORDINATION_WINDOW_SECS", 60.0),
            min_unique_wallets=env_int("COORDINATION_MIN_UNIQUE_WALLETS", 7),
            min_settled_markets=env_int("WHALE_MIN_SETTLED_MARKETS", 20),
            min_quality=env_float("WHALE_MIN_QUALITY", 0.58),
            min_pressure=env_float("WHALE_MIN_PRESSURE", 0.60),
            min_qualified_wallets=env_int("WHALE_MIN_QUALIFIED_WALLETS", 1),
            max_concentration=env_float("WHALE_MAX_CONCENTRATION", 0.75),
            min_coordination_trade_usd=env_float("WHALE_MIN_COORDINATION_TRADE_USD", 500.0),
        )
        self.max_book_age_seconds = env_float("MAX_BOOK_AGE_SECONDS", 5.0)
        self.max_leg_skew_ms = max(0, int(env_float("MAX_LEG_SKEW_MS", 1000.0)))
        self.last_fingerprint = set()
        self.directional_executor = None
        self.calibration = None
        self.edge_evaluator = None
        self.last_directional_event = ""

    def invalidate_books(self) -> None:
        """Require fresh snapshots after a market-stream reconnect."""
        for book in self.books.values():
            book.invalidate("market stream reconnect")

    async def process(self, event: dict) -> None:
        event_type = _event_name(event.get("event_type", event.get("type", "")) if isinstance(event, dict) else getattr(event, "type", ""))
        payload = event.get("payload") if isinstance(event, dict) else getattr(event, "payload", None)
        if payload is not None:
            event = payload

        def value(*names, default=None):
            for name in names:
                if isinstance(event, dict) and name in event:
                    return event[name]
                if hasattr(event, name):
                    return getattr(event, name)
            return default

        if event_type == "market_resolved":
            resolved_id = str(value("market", "condition_id", "id", default=""))
            winning = str(value("winning_outcome", "winning_asset_id", "winning_token_id", default="unknown"))
            for resolved_market in self.markets.values():
                if resolved_id in {resolved_market.market_id, resolved_market.condition_id}:
                    if winning == resolved_market.yes_token_id:
                        whale_winner = "Yes"
                    elif winning == resolved_market.no_token_id:
                        whale_winner = "No"
                    else:
                        whale_winner = winning
                    self.whale_engine.settle_market(resolved_market.market_id, whale_winner)
            if self.ledger is None:
                if self.live_journal:
                    settled = 0
                    for resolved_market in self.markets.values():
                        if resolved_id in {resolved_market.market_id, resolved_market.condition_id}:
                            settled += self.live_journal.mark_resolved(
                                resolved_market.market_id, resolved_market.condition_id, winning
                            )
                    if settled:
                        LOG.info("LIVE RESOLUTION: %s pair(s) pending redemption", settled)
                if self.negrisk_journal:
                    marked = 0
                    for resolved_market in self.negrisk_markets.values():
                        if resolved_id in {resolved_market.market_id, resolved_market.condition_id}:
                            marked += self.negrisk_journal.mark_resolved(resolved_id, winning)
                    if marked:
                        LOG.info("LIVE NEGRISK RESOLUTION: %s basket(s) marked settled; convert is still manual", marked)
                return
            for position_id, position in list(self.ledger.state["positions"].items()):
                market = self.markets.get(position.get("market_id")) or self.negrisk_markets.get(position.get("market_id"))
                if market and resolved_id in {market.market_id, market.condition_id} and not position.get("settled"):
                    self.ledger.settle(position_id, winning)
                    LOG.info("PAPER SETTLE: %s | position %s", market.title, position_id)
            return
        if event_type == "last_trade_price":
            token_id = str(value("asset_id", "token_id", default=""))
            market = self.token_to_market.get(token_id)
            if isinstance(market, BinaryMarket):
                outcome = "Yes" if token_id == market.yes_token_id else "No"
                observation = self.whale_engine.record_trade({
                    "trade_id": value("trade_id", "id", default=""),
                    "market_id": market.market_id,
                    "outcome": outcome,
                    "wallet_address": value("wallet_address", "maker", default=""),
                    "side": value("side"),
                    "price": value("price"),
                    "size": value("size"),
                    "timestamp": value("timestamp", default=time.time()),
                    "tx_hash": value("tx_hash", "transaction_hash", default=""),
                })
                if observation and observation.coordination and observation.coordination.eligible:
                    coordination = observation.coordination
                    LOG.info(
                        "WHALE COORDINATION: %s | %d wallets | %s %s | $%.0f | concentration %.0f%% | confidence %.2f",
                        market.title, coordination.unique_wallets, coordination.side,
                        coordination.outcome, coordination.total_notional_usd,
                        coordination.max_wallet_share * 100, coordination.confidence,
                    )
            return
        affected = handle_market_event(event, self.token_to_market, self.books)
        now_ms = int(time.time() * 1000)
        for market_id in affected:
            if market_id in self.negrisk_markets:
                await self._scan_negrisk(market_id, now_ms)
                continue
            market = self.markets.get(market_id)
            if market is None:
                continue
            yes_book = self.books.get(market.yes_token_id)
            no_book = self.books.get(market.no_token_id)
            if not yes_book or not no_book:
                continue
            if not yes_book.synced or not no_book.synced:
                continue
            if self.live and not getattr(self.executor, "user_stream_healthy", False):
                continue
            if not yes_book.timestamp_ms or not no_book.timestamp_ms:
                continue
            oldest_timestamp = min(yes_book.timestamp_ms, no_book.timestamp_ms)
            if now_ms + 30_000 < oldest_timestamp:
                continue
            if now_ms - oldest_timestamp > self.max_book_age_seconds * 1000:
                continue
            if abs(yes_book.timestamp_ms - no_book.timestamp_ms) > self.max_leg_skew_ms:
                continue
            opportunity = self.scanner.scan(market, yes_book, no_book)
            if not opportunity or opportunity.fingerprint in self.last_fingerprint:
                continue
            yes_cost, _, yes_fills = yes_book.walk_asks(opportunity.shares)
            no_cost, _, no_fills = no_book.walk_asks(opportunity.shares)
            if abs(yes_cost - opportunity.yes_cost) > 1e-8 or abs(no_cost - opportunity.no_cost) > 1e-8:
                continue
            try:
                if self.risk_controller:
                    self.risk_controller.check(opportunity)
                result = self.executor.execute(opportunity)
                if inspect.isawaitable(result):
                    result = await result
            except UnhedgedPairError:
                if self.risk_controller:
                    self.risk_controller.halt("unhedged live pair requires manual reconciliation")
                LOG.critical("UNHEDGED LIVE PAIR: stopping the process for manual reconciliation")
                raise
            except ValueError as error:
                LOG.info("Skip %s: %s", market.title, error)
                continue
            yes_book.consume_asks(yes_fills)
            no_book.consume_asks(no_fills)
            self.last_fingerprint.add(opportunity.fingerprint)
            if self.live:
                LOG.info("LIVE ARB HEDGED/PENDING USER CONFIRMATION: %s | %.4f shares | pair %s | YES %s | NO %s",
                         market.title, result.shares, result.pair_id, result.yes_order_id, result.no_order_id)
            else:
                LOG.info("PAPER ARB: %s | %.4f shares | capital $%.4f | net after buffer $%.4f | position %s",
                         market.title, opportunity.shares, opportunity.capital_required,
                         opportunity.net_profit, result.position_id)

    async def _scan_negrisk(self, market_id: str, now_ms: int) -> None:
        if self.negrisk_scanner is None:
            return
        market = self.negrisk_markets[market_id]
        books = []
        for token_id in market.yes_token_ids:
            book = self.books.get(token_id)
            if book is None or not book.synced or not book.timestamp_ms:
                return
            books.append(book)
        if self.live and not getattr(self.executor, "user_stream_healthy", False):
            return
        timestamps = [book.timestamp_ms for book in books]
        oldest = min(timestamps)
        newest = max(timestamps)
        if now_ms + 30_000 < oldest:
            return
        if now_ms - oldest > self.max_book_age_seconds * 1000:
            return
        if newest - oldest > self.max_leg_skew_ms:
            return
        opportunity = self.negrisk_scanner.scan(market, self.books)
        if not opportunity or opportunity.fingerprint in self.last_fingerprint:
            return
        if self.negrisk_executor is None:
            LOG.info(
                "NEGRISK OBSERVE BOOK: %s | %s | %.4f shares | net $%.4f | not executed",
                market.title, opportunity.direction, opportunity.shares, opportunity.net_profit,
            )
            self.last_fingerprint.add(opportunity.fingerprint)
            return
        try:
            if self.risk_controller and self.live:
                self.risk_controller.check_negrisk(opportunity)
            result = self.negrisk_executor.execute(opportunity)
            if inspect.isawaitable(result):
                result = await result
        except UnhedgedPairError:
            if self.risk_controller:
                self.risk_controller.halt("unfinished NegRisk basket requires manual reconciliation")
            LOG.critical("UNHEDGED NEGRISK BASKET: stopping the process for manual reconciliation")
            raise
        except ValueError as error:
            LOG.info("Skip NegRisk %s: %s", market.title, error)
            return
        for leg in opportunity.legs:
            book = self.books.get(leg.token_id)
            if book is not None:
                book.consume_asks(leg.fills)
        self.last_fingerprint.add(opportunity.fingerprint)
        LOG.info(
            "NEGRISK %s: %s | %s | %.4f shares | net $%.4f | basket %s",
            "LIVE" if self.live else "PAPER", market.title, opportunity.direction,
            opportunity.shares, opportunity.net_profit, result.basket_id,
        )

    async def execute_directional(self, intent: DirectionalIntent):
        if self.directional_executor is None:
            return None
        if self.risk_controller and self.live:
            self.risk_controller.check_directional(intent.notional + float(intent.fee_cap), intent.market_id)
        key = f"{intent.source}:{intent.event_id}:{intent.token_id}:{intent.side}"
        if key == self.last_directional_event:
            return None
        result = self.directional_executor.execute(intent)
        if inspect.isawaitable(result):
            result = await result
        self.last_directional_event = key
        LOG.info(
            "DIRECTIONAL %s: %s | %s %s | %.4f @ %.4f | %s | %s",
            "LIVE" if self.live else "PAPER", intent.source, intent.side, intent.token_id,
            result.shares, intent.limit_price, result.status, intent.reason,
        )
        return result


async def run_market_stream(runner: PaperMarketRunner, token_ids: List[str]) -> None:
    try:
        import websockets
    except ImportError as error:
        raise SystemExit("Install websockets for the paper market stream: uv sync") from error

    backoff = 1.0
    recorder = JsonlEventRecorder(os.getenv("MARKET_EVENT_LOG", ""), source="market")
    while True:
        try:
            async with websockets.connect(MARKET_WS_URL, ping_interval=10, ping_timeout=10, close_timeout=5) as websocket:
                runner.invalidate_books()
                await websocket.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
                LOG.info("Subscribed to %d binary-market tokens", len(token_ids))
                backoff = 1.0
                async for message in websocket:
                    try:
                        payload = json.loads(message)
                    except (TypeError, ValueError):
                        continue
                    for event in _events(payload):
                        recorder.record(event)
                        await runner.process(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("Market stream disconnected: %s; reconnecting in %.1fs", error, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


async def run_official_market_stream(executor: OfficialFOKExecutor,
                                     runner: PaperMarketRunner,
                                     token_ids: List[str]) -> None:
    """Consume typed official market events and fail closed on stream loss."""
    try:
        from polymarket.streams import MarketSpec
    except ImportError as error:
        raise RuntimeError("The official SDK does not provide MarketSpec") from error
    subscribe = getattr(executor.client, "subscribe", None)
    if subscribe is None:
        raise RuntimeError("Official client does not expose subscribe")
    backoff = 1.0
    recorder = JsonlEventRecorder(os.getenv("MARKET_EVENT_LOG", ""), source="market")
    while True:
        stream = None
        try:
            runner.invalidate_books()
            stream = subscribe(MarketSpec(token_ids=token_ids, custom_feature_enabled=True))
            if inspect.isawaitable(stream):
                stream = await stream
            if not hasattr(stream, "__aiter__"):
                raise RuntimeError("Official market subscription is not an async iterator")
            backoff = 1.0
            async for event in stream:
                recorder.record(event)
                await runner.process(event)
            raise RuntimeError("official market stream ended")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("Official market stream disconnected: %s; reconnecting in %.1fs", error, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        finally:
            close = getattr(stream, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result


async def run_user_stream(executor: OfficialFOKExecutor,
                          risk: Optional[LiveRiskController] = None) -> None:
    backoff = 1.0
    while True:
        try:
            await consume_user_stream(executor)
            raise RuntimeError("private user stream ended")
        except asyncio.CancelledError:
            raise
        except UnhedgedPairError:
            if risk:
                risk.halt("user stream found an unhedged live pair")
                flatten = getattr(executor, "apply_halt_actions", None)
                if flatten:
                    await flatten(risk)
            LOG.critical("User stream found an unhedged pair; stopping for manual reconciliation")
            raise
        except Exception as error:
            LOG.warning("User stream disconnected: %s; reconnecting in %.1fs", error, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


async def run_reconciliation_loop(executor: OfficialFOKExecutor,
                                  risk: LiveRiskController,
                                  interval_seconds: float = 15.0) -> None:
    """Continuously reconcile live orders; any ambiguity stops new trading."""
    interval = max(1.0, float(interval_seconds))
    cycle = 0
    full_recovery_every = max(1, env_int("LIVE_FULL_RECOVERY_CYCLES", 20))
    while True:
        await asyncio.sleep(interval)
        try:
            risk.record_account_snapshot(await executor.preflight(required_usd=0.0))
            if risk.poll_kill_switch():
                await executor.apply_halt_actions(risk)
                raise RiskHaltError(risk.state.get("halt_reason") or "kill switch")
            reconciled = await executor.reconcile(
                stale_after_seconds=env_float("STALE_ORDER_SECONDS", 30.0),
                recover_orphans=(cycle % full_recovery_every == 0),
                scan_account=(cycle % full_recovery_every == 0),
            )
            cycle += 1
            settled = await executor.settle_hedged_pairs()
            if settled:
                for record in settled:
                    risk.record_realized_pnl(float(record.get("realized_pnl", 0.0)))
                LOG.info("Settled %d live pair(s) after confirmed merge/redemption", len(settled))
            LOG.debug("Live reconciliation completed for %d unfinished pair(s)", len(reconciled))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            risk.halt(f"continuous live reconciliation failed: {error}")
            await executor.apply_halt_actions(risk)
            LOG.critical("LIVE RECONCILIATION HALT: %s", error)
            raise


async def process_sports_observation(observation, runner: PaperMarketRunner,
                                     mapping: SportsMarketMap,
                                     gate: Optional[SportsLatencyGate] = None,
                                     directional=None,
                                     allow_execution: bool = False,
                                     min_edge: float = 0.03):
    """Map one sports observation onto the live books and optionally execute."""
    if runner is None or not mapping.links:
        return None
    link = mapping.resolve(observation.game_id)
    if link is None:
        return None
    market = runner.markets.get(link.market_id)
    if market is None:
        LOG.info("SPORTS UNMAPPED MARKET: game %s -> market %s not in the live universe",
                 observation.game_id, link.market_id)
        return None
    yes_book = runner.books.get(market.yes_token_id)
    market_ts = yes_book.timestamp_ms if yes_book else 0
    price = yes_book.best_ask()[0] if yes_book and yes_book.best_ask() else None
    candidate = evaluate_sports_candidate(
        observation, gate or SportsLatencyGate(), mapping, market_ts, market_price=price,
        now_ms=observation.received_at_ms,
        allow_execution=allow_execution,
        min_edge=min_edge,
        evaluator=getattr(runner, "edge_evaluator", None),
        yes_token_id=market.yes_token_id,
        no_token_id=market.no_token_id,
        calibration=getattr(runner, "calibration", None),
    )
    LOG.info(
        "SPORTS CANDIDATE: %s | %s | eligible=%s executable=%s | %s",
        candidate.market_id, candidate.direction, candidate.eligible,
        candidate.executable, candidate.reason,
    )
    if not candidate.executable or directional is None:
        return candidate
    token_id = candidate.token_id or (
        market.yes_token_id if candidate.direction == "BUY_YES" else market.no_token_id
    )
    intent = intent_from_best_ask(
        market, token_id, runner.books.get(token_id), env_float("MAX_ORDER_USD", 100.0),
        source="sports", event_id=observation.game_id, reason=candidate.reason,
    )
    if intent is None:
        return candidate
    await runner.execute_directional(intent)
    return candidate


async def run_sports_stream(executor: OfficialFOKExecutor, tracker: SportsStateTracker,
                            runner: Optional[PaperMarketRunner] = None,
                            mapping: Optional[SportsMarketMap] = None,
                            directional=None) -> None:
    """Keep the official Sports Channel alive for timestamped observations."""
    backoff = 1.0
    mapping = mapping or SportsMarketMap()
    gate = SportsLatencyGate()
    allow_execution = env_bool("ENABLE_SPORTS_EXECUTION")
    if runner and runner.live and not env_bool("ENABLE_SPORTS_LIVE"):
        allow_execution = False
    min_edge = env_float("SPORTS_MIN_EDGE", 0.03)

    async def observe(event) -> None:
        observation = tracker.observe(event)
        if observation.changed:
            LOG.info(
                "SPORTS STATE: game %s | %s | score %s | period %s",
                observation.game_id, observation.status, observation.score, observation.period,
            )
        await process_sports_observation(
            observation, runner, mapping, gate, directional,
            allow_execution=allow_execution, min_edge=min_edge,
        )

    while True:
        try:
            await consume_sports_channel(executor.client, observe)
            raise RuntimeError("sports stream ended")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("Sports stream disconnected: %s; reconnecting in %.1fs", error, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


def _token_for_direction(market: BinaryMarket, direction: str) -> str:
    normalized = str(direction or "").upper()
    if normalized in {"BUY_NO", "SELL_MARKET", "NO"}:
        return market.no_token_id
    return market.yes_token_id


async def process_macro_release(runner: PaperMarketRunner, model: MacroEventModel,
                                release: MacroRelease, now_ms: Optional[int] = None):
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    market_id = model.market_map.get(release.indicator, "")
    market = runner.markets.get(market_id)
    price = 0.5
    book = None
    if market:
        book = runner.books.get(market.yes_token_id)
        if book and book.best_ask():
            price = book.best_ask()[0]
    if env_bool("ENABLE_CALIBRATION_AUTOTUNE") and runner.calibration:
        model.apply_calibration()
        if runner.edge_evaluator:
            runner.calibration.apply_recommended_edge(runner.edge_evaluator, model.strategy)
    signal = model.predict(release, price, now_ms, market_id=market_id)
    LOG.info(
        "MACRO SIGNAL: %s %s | eligible=%s executable=%s edge=%.4f | %s",
        signal.event_id, signal.market_id, signal.eligible, signal.executable,
        signal.edge, signal.reason,
    )
    if not signal.executable or market is None or book is None:
        return signal
    token_id = _token_for_direction(market, signal.direction)
    intent = intent_from_best_ask(
        market, token_id, runner.books.get(token_id), env_float("MAX_ORDER_USD", 100.0),
        source="macro", event_id=signal.event_id, reason=signal.reason,
    )
    if intent:
        await runner.execute_directional(intent)
    return signal


async def run_macro_feed(runner: PaperMarketRunner, model: MacroEventModel, feed: JsonlMacroFeed) -> None:
    interval = max(0.5, env_float("MACRO_POLL_SECONDS", 2.0))
    live_ok = (not runner.live) or env_bool("ENABLE_MACRO_LIVE")
    model.allow_execution = env_bool("ENABLE_MACRO_EXECUTION") and live_ok
    while True:
        await asyncio.sleep(interval)
        now_ms = int(time.time() * 1000)
        for release in feed.poll():
            await process_macro_release(runner, model, release, now_ms=now_ms)


async def process_crypto_quote(runner: PaperMarketRunner, model: CryptoStatArbModel,
                               quote, now_ms: Optional[int] = None):
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    market = runner.markets.get(quote.market_id)
    if market is None:
        return None
    yes_book = runner.books.get(market.yes_token_id)
    if not yes_book or not yes_book.synced or not yes_book.best_ask():
        return None
    observation = CryptoObservation(
        quote.market_id, yes_book.best_ask()[0], quote.implied_probability, quote.timestamp_ms,
        market_timestamp_ms=yes_book.timestamp_ms,
    )
    signal = model.observe(observation, now_ms, reference_timestamp_ms=quote.timestamp_ms)
    LOG.info(
        "CRYPTO SIGNAL: %s | z=%.3f %s %s eligible=%s executable=%s | %s",
        signal.market_id, signal.zscore, signal.action, signal.direction,
        signal.eligible, signal.executable, signal.reason,
    )
    if not signal.executable:
        return signal
    if signal.action == "EXIT":
        held = model.inventory.get(signal.market_id) or {}
        token_id = held.get("token_id") or market.yes_token_id
        book = runner.books.get(token_id)
        shares = float(held.get("shares") or 0.0)
        intent = intent_from_inventory_bid(
            market, token_id, book, shares,
            source="crypto", event_id=signal.market_id, reason=signal.reason,
        )
        if intent:
            result = await runner.execute_directional(intent)
            if result:
                model.mark_closed(signal.market_id)
        return signal
    token_id = _token_for_direction(market, signal.direction)
    intent = intent_from_best_ask(
        market, token_id, runner.books.get(token_id), env_float("MAX_ORDER_USD", 100.0),
        source="crypto", event_id=signal.market_id, reason=signal.reason,
    )
    if intent:
        result = await runner.execute_directional(intent)
        if result:
            model.mark_open(signal, token_id=token_id, shares=result.shares)
    return signal


async def run_crypto_feed(runner: PaperMarketRunner, model: CryptoStatArbModel,
                          feed: JsonlCryptoFeed) -> None:
    interval = max(0.5, env_float("CRYPTO_POLL_SECONDS", 2.0))
    live_ok = (not runner.live) or env_bool("ENABLE_CRYPTO_LIVE")
    model.allow_execution = env_bool("ENABLE_CRYPTO_EXECUTION") and live_ok
    while True:
        await asyncio.sleep(interval)
        now_ms = int(time.time() * 1000)
        for quote in feed.poll():
            await process_crypto_quote(runner, model, quote, now_ms=now_ms)


async def run_health_loop(path: str, risk: Optional[LiveRiskController],
                          journal: Optional[LiveOrderJournal],
                          directional: Optional[LiveDirectionalJournal],
                          stop: asyncio.Event,
                          negrisk: Optional[LiveNegRiskJournal] = None) -> None:
    interval = max(1.0, env_float("LIVE_HEALTH_INTERVAL_SECONDS", 5.0))
    while not stop.is_set():
        write_health(path, {
            "status": "halted" if risk and risk.state.get("halted") else "running",
            "halt_reason": (risk.state.get("halt_reason") if risk else ""),
            "pair_exposure": journal.open_exposure() if journal else 0.0,
            "directional_exposure": directional.open_exposure() if directional else 0.0,
            "negrisk_exposure": negrisk.open_exposure() if negrisk else 0.0,
            "open_pairs": len(journal.incomplete_pairs()) if journal else 0,
            "open_directional": len(directional.incomplete_trades()) if directional else 0,
            "open_negrisk": len(negrisk.incomplete_baskets()) if negrisk else 0,
        })
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


def _install_stop_signal(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: stop.set())


def _parse_json_map(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _attach_research(runner: PaperMarketRunner, directional_journal: Optional[LiveDirectionalJournal]):
    runner.calibration = CalibrationTracker(
        min_samples=env_int("CALIBRATION_MIN_SAMPLES", 20),
        path=os.getenv("CALIBRATION_PATH", "calibration.json"),
    )
    runner.edge_evaluator = EdgeEvaluator()
    if env_bool("ENABLE_CALIBRATION_AUTOTUNE"):
        for strategy in ("macro-event-v1", "crypto-spread-v1", "sports-latency-v1"):
            runner.calibration.apply_recommended_edge(runner.edge_evaluator, strategy)
    return runner.calibration


def _research_tasks(runner: PaperMarketRunner, executor=None, directional=None) -> set:
    tasks = set()
    sports_map = SportsMarketMap(_parse_json_map(os.getenv("SPORTS_MARKET_MAP", "")))
    if executor is not None and os.getenv("ENABLE_SPORTS_CHANNEL", "1") == "1":
        tasks.add(asyncio.create_task(
            run_sports_stream(executor, SportsStateTracker(), runner, sports_map, directional)
        ))
    macro_path = os.getenv("MACRO_FEED_PATH", "").strip()
    macro_map = _parse_json_map(os.getenv("MACRO_MARKET_MAP", ""))
    if macro_path:
        model = MacroEventModel(
            runner.calibration or CalibrationTracker(),
            market_map=macro_map,
            min_edge=env_float("MACRO_MIN_EDGE", 0.03),
        )
        tasks.add(asyncio.create_task(run_macro_feed(runner, model, JsonlMacroFeed(macro_path))))
    crypto_path = os.getenv("CRYPTO_REFERENCE_FEED_PATH", "").strip()
    if crypto_path:
        model = CryptoStatArbModel(
            runner.calibration or CalibrationTracker(),
            allow_execution=env_bool("ENABLE_CRYPTO_EXECUTION") and ((not runner.live) or env_bool("ENABLE_CRYPTO_LIVE")),
            max_reference_lag_ms=env_int("CRYPTO_MAX_REFERENCE_LAG_MS", 1_000),
        )
        tasks.add(asyncio.create_task(run_crypto_feed(runner, model, JsonlCryptoFeed(crypto_path))))
    return tasks


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Polymarket deterministic binary arbitrage scanner")
    parser.add_argument("--live", action="store_true", help="Use the official FOK executor after explicit environment confirmation")
    parser.add_argument("--preflight", action="store_true", help="Run live account and journal checks without starting streams or placing orders")
    parser.add_argument("--markets", type=int, default=env_int("MARKET_LIMIT", 100))
    parser.add_argument("--ledger", default=os.getenv("PAPER_LEDGER", "paper-ledger.json"))
    parser.add_argument("--live-journal", default=os.getenv("LIVE_ORDER_JOURNAL", "live-orders.json"))
    parser.add_argument("--directional-journal", default=os.getenv("LIVE_DIRECTIONAL_JOURNAL", "live-directional.json"))
    parser.add_argument("--negrisk-journal", default=os.getenv("LIVE_NEGRISK_JOURNAL", "live-negrisk.json"))
    parser.add_argument("--health", action="store_true", help="Print the local health snapshot and exit")
    parser.add_argument("--status", action="store_true", help="Print the local live journal summary and exit")
    parser.add_argument("--cash", type=float, default=env_float("PAPER_CASH", 1000.0))
    parser.add_argument("--max-order", type=float, default=env_float("MAX_ORDER_USD", 100.0))
    parser.add_argument("--min-profit", type=float, default=env_float("MIN_NET_PROFIT_USD", 0.05))
    parser.add_argument("--min-return", type=float, default=env_float("MIN_RETURN_ON_CAPITAL", 0.002))
    parser.add_argument("--buffer", type=float, default=env_float("SAFETY_BUFFER_USD", 0.02))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    health_path = os.getenv("LIVE_HEALTH_PATH", "live-health.json")
    if args.health:
        if not os.path.isfile(health_path):
            print(json.dumps({"status": "missing", "path": health_path}, indent=2, sort_keys=True))
            return 1
        with open(health_path, encoding="utf-8") as handle:
            print(handle.read())
        return 0
    if args.status:
        payload = {
            "pairs": LiveOrderJournal(args.live_journal).summary(),
            "directional": LiveDirectionalJournal(args.directional_journal).summary(),
            "negrisk": LiveNegRiskJournal(args.negrisk_journal).summary(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.preflight and not args.live:
        LOG.error("--preflight must be combined with --live")
        return 2
    if args.live and os.getenv("POLYMARKET_LIVE_CONFIRM") != "I_UNDERSTAND_THE_RISK":
        LOG.error("Set POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK to unlock live FOK execution")
        return 2

    want_negrisk = negrisk_execution_enabled(args.live)
    negrisk_limit = env_int("NEGRISK_MARKET_LIMIT", 10) if want_negrisk else 0
    markets, negrisk_markets = fetch_universe(args.markets, negrisk_limit=negrisk_limit)
    if not markets:
        LOG.error("No active binary markets with Yes/No token IDs were found")
        return 1
    scanner = BinaryArbitrageScanner(
        min_net_profit_usd=args.min_profit,
        min_return=args.min_return,
        safety_buffer_usd=args.buffer,
        max_order_usd=args.max_order,
        max_levels=env_int("LIVE_MAX_BOOK_LEVELS", 1) if args.live else None,
        merge_gas_usd=env_float("MERGE_GAS_USD", 0.0),
    )
    negrisk_scanner = NegRiskBookScanner(
        min_net_profit_usd=args.min_profit,
        min_return=args.min_return,
        safety_buffer_usd=args.buffer,
        max_order_usd=args.max_order,
        max_levels=env_int("LIVE_MAX_BOOK_LEVELS", 1) if args.live else None,
        merge_gas_usd=env_float("MERGE_GAS_USD", 0.0),
    ) if negrisk_markets else None
    gas_warning = merge_gas_startup_warning(
        env_float("MERGE_GAS_USD", 0.0), args.max_order, env_bool("AUTO_MERGE_COMPLETE_SETS"),
    )
    if gas_warning:
        LOG.warning("%s", gas_warning)
    if args.live:
        geoblock = requests.get("https://polymarket.com/api/geoblock", timeout=5).json()
        if geoblock.get("blocked"):
            LOG.error("Official geoblock denied this connection: %s", geoblock)
            return 2

        async def run_live() -> None:
            journal = LiveOrderJournal(args.live_journal)
            directional_journal = LiveDirectionalJournal(args.directional_journal)
            negrisk_journal = LiveNegRiskJournal(args.negrisk_journal)
            executor = await OfficialFOKExecutor.create_from_env(
                journal=journal, directional_journal=directional_journal,
            )
            negrisk_live_executor = (
                OfficialNegRiskExecutor(executor, negrisk_journal)
                if want_negrisk and negrisk_markets else None
            )
            account = await executor.preflight(required_usd=0.01)
            risk = LiveRiskController(
                journal,
                equity_usd=env_float("LIVE_RISK_EQUITY_USD", account["balance"] + journal.open_exposure() + negrisk_journal.open_exposure()),
                state_path=os.getenv("LIVE_RISK_STATE_PATH", "live-risk.json"),
                kill_switch_path=os.getenv("LIVE_KILL_SWITCH_PATH", "live-kill-switch"),
                max_total_exposure_fraction=env_float("LIVE_MAX_TOTAL_EXPOSURE_FRACTION", 0.25),
                max_market_exposure_fraction=env_float("LIVE_MAX_MARKET_EXPOSURE_FRACTION", 0.05),
                max_open_pairs=env_int("LIVE_MAX_OPEN_PAIRS", 10),
                max_daily_loss_usd=env_float("LIVE_MAX_DAILY_LOSS_USD", 0.0),
                extra_journals=[directional_journal],
                max_open_directional=env_int("LIVE_MAX_OPEN_DIRECTIONAL", 5),
                negrisk_journal=negrisk_journal,
                max_open_negrisk=env_int("LIVE_MAX_OPEN_NEGRISK", 2) if negrisk_live_executor else 0,
            )
            directional = DirectionalExecutor(executor, directional_journal, risk=risk)
            try:
                risk.check_startup()
            except RiskHaltError:
                await executor.apply_halt_actions(risk)
                raise
            try:
                await executor.reconcile(
                    stale_after_seconds=env_float("STALE_ORDER_SECONDS", 30.0),
                    scan_account=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                risk.halt(f"startup live reconciliation failed: {error}")
                await executor.apply_halt_actions(risk)
                raise
            if args.preflight:
                print(json.dumps({
                    "preflight": "ok",
                    "account": account,
                    "journal": journal.summary(),
                    "directional": directional_journal.summary(),
                    "negrisk": negrisk_journal.summary(),
                    "risk": risk.state,
                }, indent=2, sort_keys=True))
                await executor.close()
                return
            runner = PaperMarketRunner(
                markets, args.ledger, args.cash, scanner, executor=executor,
                risk_controller=risk,
                negrisk_markets=negrisk_markets,
                negrisk_scanner=negrisk_scanner,
                negrisk_executor=negrisk_live_executor,
            )
            runner.directional_executor = directional
            _attach_research(runner, directional_journal)
            stop = asyncio.Event()
            _install_stop_signal(stop)
            market_task = asyncio.create_task(
                run_official_market_stream(executor, runner, list(runner.token_to_market))
            )
            user_task = asyncio.create_task(run_user_stream(executor, risk))
            reconcile_task = asyncio.create_task(
                run_reconciliation_loop(
                    executor, risk, env_float("LIVE_RECONCILIATION_INTERVAL_SECONDS", 15.0)
                )
            )
            health_task = asyncio.create_task(
                run_health_loop(health_path, risk, journal, directional_journal, stop, negrisk_journal)
            )
            tasks = {market_task, user_task, reconcile_task, health_task}
            tasks |= _research_tasks(runner, executor, directional)
            shutdown_task = asyncio.create_task(stop.wait())
            try:
                done, pending = await asyncio.wait(
                    tasks | {shutdown_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if shutdown_task in done:
                    LOG.info("Shutdown signal received")
                for task in done:
                    if task is shutdown_task:
                        continue
                    task.result()
            finally:
                stop.set()
                for task in tasks | {shutdown_task}:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, shutdown_task, return_exceptions=True)
                try:
                    if risk.state.get("halted"):
                        await executor.apply_halt_actions(risk)
                    elif env_bool("LIVE_CANCEL_ON_SHUTDOWN", True):
                        await executor.cancel_all_open_orders()
                except Exception as flatten_error:
                    LOG.critical("SHUTDOWN CANCEL-ALL FAILED: %s", flatten_error)
                await executor.close()

        try:
            asyncio.run(run_live())
        except KeyboardInterrupt:
            LOG.info("Stopped")
        except RiskHaltError as error:
            LOG.critical("LIVE RISK HALT: %s", error)
            return 3
        return 0

    async def run_paper() -> None:
        negrisk_journal = LiveNegRiskJournal(args.negrisk_journal)
        paper_negrisk = None
        runner = PaperMarketRunner(
            markets, args.ledger, args.cash, scanner,
            negrisk_markets=negrisk_markets,
            negrisk_scanner=negrisk_scanner,
        )
        if want_negrisk and negrisk_markets:
            paper_negrisk = PaperNegRiskExecutor(negrisk_journal, runner.ledger)
            runner.negrisk_executor = paper_negrisk
            runner.negrisk_journal = negrisk_journal
        directional_journal = LiveDirectionalJournal(args.directional_journal)
        runner.directional_executor = PaperDirectionalExecutor(directional_journal)
        _attach_research(runner, directional_journal)
        stop = asyncio.Event()
        _install_stop_signal(stop)
        health_task = asyncio.create_task(
            run_health_loop(health_path, None, None, directional_journal, stop, negrisk_journal)
        )
        market_task = asyncio.create_task(run_market_stream(runner, list(runner.token_to_market)))
        tasks = {market_task, health_task} | _research_tasks(runner)
        shutdown_task = asyncio.create_task(stop.wait())
        try:
            done, pending = await asyncio.wait(
                tasks | {shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if shutdown_task not in done:
                for task in done:
                    task.result()
        finally:
            stop.set()
            for task in tasks | {shutdown_task}:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, shutdown_task, return_exceptions=True)

    try:
        asyncio.run(run_paper())
    except KeyboardInterrupt:
        LOG.info("Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
