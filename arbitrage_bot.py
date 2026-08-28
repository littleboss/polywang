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
import time
from typing import Dict, Iterable, List, Optional

import requests
from whale_intelligence import WhaleIntelligenceEngine
from sports_channel import SportsStateTracker, SportsLatencyGate, SportsMarketMap, consume_sports_channel, evaluate_sports_candidate
from market_replay import JsonlEventRecorder

from arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    JsonLedger,
    LiveOrderJournal,
    OrderBook,
    PaperArbitrageExecutor,
    OfficialFOKExecutor,
    LiveRiskController,
    RiskHaltError,
    UnhedgedPairError,
    handle_market_event,
    consume_user_stream,
    _event_name,
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


def fetch_markets(limit: int) -> List[BinaryMarket]:
    """Fetch active binary markets and keep only markets with two outcomes."""
    response = requests.get(
        GAMMA_URL,
        params={"closed": "false", "active": "true", "limit": limit, "order": "volume_24hr", "ascending": "false"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    markets = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = BinaryMarket.from_gamma(row)
        if parsed and parsed.active:
            markets.append(parsed)
    return markets


def _events(payload) -> Iterable[dict]:
    values = payload if isinstance(payload, list) else [payload]
    return (value for value in values if isinstance(value, dict))


class PaperMarketRunner:
    def __init__(self, markets: List[BinaryMarket], ledger_path: str,
                 initial_cash: float, scanner: BinaryArbitrageScanner, executor=None,
                 risk_controller: Optional[LiveRiskController] = None):
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
        self.markets: Dict[str, BinaryMarket] = {market.market_id: market for market in markets}
        self.token_to_market = {
            token: market
            for market in markets
            for token in (market.yes_token_id, market.no_token_id)
        }
        self.books: Dict[str, OrderBook] = {}
        self.scanner = scanner
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
        self.last_fingerprint = set()

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
                return
            for position_id, position in list(self.ledger.state["positions"].items()):
                market = self.markets.get(position.get("market_id"))
                if market and resolved_id in {market.market_id, market.condition_id} and not position.get("settled"):
                    self.ledger.settle(position_id, winning)
                    LOG.info("PAPER SETTLE: %s | position %s", market.title, position_id)
            return
        if event_type == "last_trade_price":
            token_id = str(value("asset_id", "token_id", default=""))
            market = self.token_to_market.get(token_id)
            if market:
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
            market = self.markets[market_id]
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


async def run_market_stream(runner: PaperMarketRunner, token_ids: List[str]) -> None:
    try:
        import websockets
    except ImportError as error:
        raise SystemExit("Install websockets for the paper market stream: python3 -m pip install websockets") from error

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


async def run_sports_stream(executor: OfficialFOKExecutor, tracker: SportsStateTracker,
                            runner: Optional[PaperMarketRunner] = None,
                            mapping: Optional[SportsMarketMap] = None) -> None:
    """Keep the official Sports Channel alive for timestamped observations."""
    backoff = 1.0
    mapping = mapping or SportsMarketMap()
    gate = SportsLatencyGate()

    async def observe(event) -> None:
        observation = tracker.observe(event)
        if observation.changed:
            LOG.info(
                "SPORTS STATE: game %s | %s | score %s | period %s",
                observation.game_id, observation.status, observation.score, observation.period,
            )
        if not mapping.links or runner is None:
            return
        link = mapping.resolve(observation.game_id)
        if link is None:
            return
        market = runner.markets.get(link.market_id)
        if market is None:
            LOG.info("SPORTS UNMAPPED MARKET: game %s -> market %s not in the live universe",
                     observation.game_id, link.market_id)
            return
        yes_book = runner.books.get(market.yes_token_id)
        market_ts = yes_book.timestamp_ms if yes_book else 0
        price = yes_book.best_ask()[0] if yes_book and yes_book.best_ask() else None
        candidate = evaluate_sports_candidate(
            observation, gate, mapping, market_ts, market_price=price,
        )
        LOG.info(
            "SPORTS CANDIDATE: %s | %s | eligible=%s executable=%s | %s",
            candidate.market_id, candidate.direction, candidate.eligible,
            candidate.executable, candidate.reason,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket deterministic binary arbitrage scanner")
    parser.add_argument("--live", action="store_true", help="Use the official FOK executor after explicit environment confirmation")
    parser.add_argument("--preflight", action="store_true", help="Run live account and journal checks without starting streams or placing orders")
    parser.add_argument("--markets", type=int, default=env_int("MARKET_LIMIT", 100))
    parser.add_argument("--ledger", default=os.getenv("PAPER_LEDGER", "paper-ledger.json"))
    parser.add_argument("--live-journal", default=os.getenv("LIVE_ORDER_JOURNAL", "live-orders.json"))
    parser.add_argument("--status", action="store_true", help="Print the local live journal summary and exit")
    parser.add_argument("--cash", type=float, default=env_float("PAPER_CASH", 1000.0))
    parser.add_argument("--max-order", type=float, default=env_float("MAX_ORDER_USD", 100.0))
    parser.add_argument("--min-profit", type=float, default=env_float("MIN_NET_PROFIT_USD", 0.05))
    parser.add_argument("--min-return", type=float, default=env_float("MIN_RETURN_ON_CAPITAL", 0.002))
    parser.add_argument("--buffer", type=float, default=env_float("SAFETY_BUFFER_USD", 0.02))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.status:
        print(json.dumps(LiveOrderJournal(args.live_journal).summary(), indent=2, sort_keys=True))
        return 0
    if args.preflight and not args.live:
        LOG.error("--preflight must be combined with --live")
        return 2
    if args.live and os.getenv("POLYMARKET_LIVE_CONFIRM") != "I_UNDERSTAND_THE_RISK":
        LOG.error("Set POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK to unlock live FOK execution")
        return 2

    markets = fetch_markets(args.markets)
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
    if args.live:
        geoblock = requests.get("https://polymarket.com/api/geoblock", timeout=5).json()
        if geoblock.get("blocked"):
            LOG.error("Official geoblock denied this connection: %s", geoblock)
            return 2

        async def run_live() -> None:
            journal = LiveOrderJournal(args.live_journal)
            executor = await OfficialFOKExecutor.create_from_env(journal=journal)
            account = await executor.preflight(required_usd=0.01)
            risk = LiveRiskController(
                journal,
                equity_usd=env_float("LIVE_RISK_EQUITY_USD", account["balance"] + journal.open_exposure()),
                state_path=os.getenv("LIVE_RISK_STATE_PATH", "live-risk.json"),
                kill_switch_path=os.getenv("LIVE_KILL_SWITCH_PATH", "live-kill-switch"),
                max_total_exposure_fraction=env_float("LIVE_MAX_TOTAL_EXPOSURE_FRACTION", 0.25),
                max_market_exposure_fraction=env_float("LIVE_MAX_MARKET_EXPOSURE_FRACTION", 0.05),
                max_open_pairs=env_int("LIVE_MAX_OPEN_PAIRS", 10),
                max_daily_loss_usd=env_float("LIVE_MAX_DAILY_LOSS_USD", 0.0),
            )
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
                    "risk": risk.state,
                }, indent=2, sort_keys=True))
                await executor.close()
                return
            runner = PaperMarketRunner(
                markets, args.ledger, args.cash, scanner, executor=executor,
                risk_controller=risk,
            )
            market_task = asyncio.create_task(
                run_official_market_stream(executor, runner, list(runner.token_to_market))
            )
            user_task = asyncio.create_task(run_user_stream(executor, risk))
            reconcile_task = asyncio.create_task(
                run_reconciliation_loop(
                    executor, risk, env_float("LIVE_RECONCILIATION_INTERVAL_SECONDS", 15.0)
                )
            )
            sports_task = None
            sports_map = SportsMarketMap()
            raw_map = os.getenv("SPORTS_MARKET_MAP", "").strip()
            if raw_map:
                try:
                    parsed_map = json.loads(raw_map)
                    if isinstance(parsed_map, dict):
                        sports_map = SportsMarketMap(parsed_map)
                except (TypeError, ValueError) as error:
                    LOG.warning("Ignoring invalid SPORTS_MARKET_MAP: %s", error)
            if os.getenv("ENABLE_SPORTS_CHANNEL", "1") == "1":
                sports_task = asyncio.create_task(
                    run_sports_stream(executor, SportsStateTracker(), runner, sports_map)
                )
            tasks = {market_task, user_task, reconcile_task} | ({sports_task} if sports_task else set())
            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_EXCEPTION
                )
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if risk.state.get("halted"):
                    try:
                        await executor.apply_halt_actions(risk)
                    except Exception as flatten_error:
                        LOG.critical("HALT CANCEL-ALL FAILED: %s", flatten_error)
                await executor.close()

        try:
            asyncio.run(run_live())
        except KeyboardInterrupt:
            LOG.info("Stopped")
        except RiskHaltError as error:
            LOG.critical("LIVE RISK HALT: %s", error)
            return 3
        return 0

    runner = PaperMarketRunner(markets, args.ledger, args.cash, scanner)
    token_ids = list(runner.token_to_market)
    try:
        asyncio.run(run_market_stream(runner, token_ids))
    except KeyboardInterrupt:
        LOG.info("Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
