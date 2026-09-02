import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from polywang.arbitrage_bot import PaperMarketRunner, load_dotenv, run_user_stream, write_health
from polywang.arbitrage_core import BinaryArbitrageScanner, BinaryMarket, UnhedgedPairError


class SettlementMappingTests(unittest.TestCase):
    def test_runner_rejects_duplicate_market_identifiers(self):
        first = BinaryMarket("m1", "c1", "One", "yes-token", "no-token")
        duplicate = BinaryMarket("m2", "c2", "Two", "yes-token", "other-token")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate token"):
                PaperMarketRunner(
                    [first, duplicate], os.path.join(directory, "ledger.json"), 100.0,
                    BinaryArbitrageScanner(),
                )

    def test_winning_asset_id_is_mapped_to_whale_outcome(self):
        market = BinaryMarket("m1", "c1", "Test", "yes-token", "no-token")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False
        ):
            runner = PaperMarketRunner(
                [market], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(),
            )
            asyncio.run(runner.process({
                "event_type": "last_trade_price", "asset_id": "yes-token",
                "trade_id": "t1", "wallet_address": "0x" + "a" * 40,
                "side": "BUY", "price": "0.40", "size": "10",
            }))
            asyncio.run(runner.process({
                "event_type": "market_resolved", "market": "m1",
                "winning_asset_id": "yes-token",
            }))
            snapshot = runner.whale_engine.snapshot("0x" + "a" * 40)
            self.assertEqual(snapshot["settled_markets"], 1)
            self.assertGreater(snapshot["realized_pnl"], 0.0)

    def test_user_stream_persists_halt_when_it_detects_unhedged_pair(self):
        class Executor:
            async def close(self):
                return None

        class Risk:
            def __init__(self):
                self.reason = ""

            def halt(self, reason):
                self.reason = reason

        risk = Risk()
        executor = Executor()

        async def failing_consumer(_executor):
            raise UnhedgedPairError("one leg missing")

        with mock.patch("polywang.arbitrage_bot.consume_user_stream", failing_consumer):
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(run_user_stream(executor, risk))
        self.assertIn("unhedged", risk.reason)

    def test_load_dotenv_does_not_override_existing_env(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, ".env")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("POLYMARKET_LIVE_CONFIRM=from-file\nEXISTING_KEEP=file\n")
            os.environ["EXISTING_KEEP"] = "process"
            os.environ.pop("POLYMARKET_LIVE_CONFIRM", None)
            load_dotenv(path)
            self.assertEqual(os.environ.get("POLYMARKET_LIVE_CONFIRM"), "from-file")
            self.assertEqual(os.environ.get("EXISTING_KEEP"), "process")
            os.environ.pop("POLYMARKET_LIVE_CONFIRM", None)

    def test_health_snapshot_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "live-health.json")
            write_health(path, {"status": "running", "open_pairs": 0})
            with open(path, encoding="utf-8") as handle:
                payload = __import__("json").load(handle)
            self.assertEqual(payload["status"], "running")
            self.assertIn("updated_at", payload)


class UniverseSelectionTests(unittest.TestCase):
    def test_fetch_markets_ranks_by_breakeven_ticks_not_volume_order(self):
        from polywang.arbitrage_bot import fetch_markets
        rows = [
            {
                "id": "vol-mid", "conditionId": "c1", "question": "Busy mid",
                "clobTokenIds": '["y1", "n1"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.50", "0.50"]', "category": "politics",
                "active": True, "closed": False,
            },
            {
                "id": "geo", "conditionId": "c2", "question": "Geopolitics",
                "clobTokenIds": '["y2", "n2"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.55", "0.45"]', "category": "geopolitics",
                "active": True, "closed": False,
            },
            {
                "id": "extreme", "conditionId": "c3", "question": "Extreme politics",
                "clobTokenIds": '["y3", "n3"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.95", "0.05"]', "category": "politics",
                "active": True, "closed": False,
            },
        ]

        def getter(params):
            self.assertEqual(params["order"], "volume24hr")
            return list(rows)

        selected = fetch_markets(2, get=getter, pool=3)
        self.assertEqual([market.market_id for market in selected], ["geo", "extreme"])

    def test_fetch_markets_logs_negrisk_without_selecting_it(self):
        from polywang.arbitrage_bot import fetch_markets
        rows = [
            {
                "id": "nr", "conditionId": "cnr", "question": "Who wins",
                "clobTokenIds": '["a", "b", "c", "d"]',
                "outcomes": '["A", "B", "C", "D"]',
                "outcomePrices": '["0.30", "0.25", "0.20", "0.15"]',
                "negRisk": True, "active": True, "closed": False,
            },
            {
                "id": "bin", "conditionId": "cb", "question": "Binary",
                "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]', "category": "geopolitics",
                "active": True, "closed": False,
            },
        ]
        with self.assertLogs("arbitrage-bot", level="INFO") as captured:
            selected = fetch_markets(5, get=lambda params: rows, pool=5)
        self.assertEqual([market.market_id for market in selected], ["bin"])
        self.assertTrue(any("NEGRISK OBSERVE" in line for line in captured.output))


class ResearchExecutionPathTests(unittest.TestCase):
    def _runner(self, directory, market=None):
        from polywang.arbitrage_core import LiveDirectionalJournal, PaperDirectionalExecutor
        binary = market or BinaryMarket("m1", "c1", "Test", "yes-token", "no-token")
        runner = PaperMarketRunner(
            [binary], os.path.join(directory, "ledger.json"), 100.0,
            BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
        )
        journal = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
        runner.directional_executor = PaperDirectionalExecutor(journal)
        return runner, binary, journal

    def _synced_book(self, ask=0.42, bid=0.40, timestamp_ms=1_700_000_000_000):
        from polywang.arbitrage_core import OrderBook
        book = OrderBook()
        book.asks = {ask: 20.0}
        book.bids = {bid: 20.0}
        book.synced = True
        book.timestamp_ms = timestamp_ms
        return book

    def test_runner_skips_cross_leg_timestamp_skew(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_LEG_SKEW_MS": "500"}, clear=False,
        ):
            market = BinaryMarket("m1", "c1", "Test", "yes-token", "no-token")
            runner = PaperMarketRunner(
                [market], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
            )
            runner.max_book_age_seconds = 1e9
            now = int(__import__("time").time() * 1000)
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "yes-token", "timestamp": str(now),
                "hash": "y", "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            }))
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "no-token", "timestamp": str(now + 2000),
                "hash": "n", "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            }))
            self.assertEqual(runner.ledger.state["positions"], {})

    def test_sports_candidate_fills_through_directional_executor(self):
        from polywang.arbitrage_bot import process_sports_observation
        from polywang.polymarket_edge import CalibrationTracker
        from polywang.sports_channel import SportsLatencyGate, SportsMarketMap, SportsStateTracker
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_ORDER_USD": "10"}, clear=False,
        ):
            market = BinaryMarket("m-home", "c1", "Test", "yes-token", "no-token")
            runner, _, journal = self._runner(directory, market)
            runner.books["yes-token"] = self._synced_book(ask=0.42)
            calibration = CalibrationTracker(min_samples=2)
            calibration.record("sports-latency-v1", 0.9, 1)
            calibration.record("sports-latency-v1", 0.1, 0)
            runner.calibration = calibration
            observation = SportsStateTracker().observe({
                "gameId": "g1", "status": "LIVE", "live": True, "ended": False,
                "score": "2-0", "period": "70'", "source_timestamp": 1_700_000_004_000,
            }, received_at_ms=1_700_000_005_000)
            mapping = SportsMarketMap({"g1": {"market_id": "m-home", "yes_means": "home"}})
            candidate = asyncio.run(process_sports_observation(
                observation, runner, mapping,
                SportsLatencyGate(max_age_seconds=5, min_delay_ms=100, max_delay_ms=5_000),
                runner.directional_executor, allow_execution=True, min_edge=0.03,
            ))
            self.assertTrue(candidate.executable)
            self.assertGreater(journal.open_exposure(), 0.0)
            self.assertAlmostEqual(journal.inventory_by_token()["yes-token"], 20.0)

    def test_macro_vendor_jsonl_fills_through_directional_executor(self):
        from polywang.arbitrage_bot import process_macro_release
        from polywang.macro_model import MacroEventModel, MacroRelease
        from polywang.polymarket_edge import CalibrationTracker
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_ORDER_USD": "10"}, clear=False,
        ):
            runner, _, journal = self._runner(directory)
            runner.books["yes-token"] = self._synced_book(ask=0.50)
            tracker = CalibrationTracker(min_samples=2)
            tracker.record("macro-event-v1", 0.9, 1)
            tracker.record("macro-event-v1", 0.1, 0)
            runner.calibration = tracker
            model = MacroEventModel(
                tracker, min_edge=0.01, surprise_weight=1.0, market_map={"cpi": "m1"},
            )
            model.allow_execution = True
            release = MacroRelease.from_payload({
                "id": "e-print", "indicator": "cpi", "print": 5.0,
                "forecast": 4.0, "stdev": 0.5, "timestamp": 1_700_000_000,
            })
            signal = asyncio.run(process_macro_release(
                runner, model, release, now_ms=1_700_000_001_000,
            ))
            self.assertTrue(signal.executable)
            self.assertGreater(journal.open_exposure(), 0.0)

    def test_crypto_buy_then_exit_sells_inventory(self):
        from polywang.arbitrage_bot import process_crypto_quote
        from polywang.crypto_model import CryptoObservation, CryptoReferenceQuote, CryptoStatArbModel
        from polywang.polymarket_edge import CalibrationTracker
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_ORDER_USD": "10"}, clear=False,
        ):
            runner, _, journal = self._runner(directory)
            history_time = 1_700_000_000_000
            book = self._synced_book(ask=0.20, bid=0.19, timestamp_ms=history_time + 12)
            runner.books["yes-token"] = book
            tracker = CalibrationTracker(min_samples=1)
            tracker.record("crypto-spread-v1", 0.9, 1)
            model = CryptoStatArbModel(
                tracker, entry_zscore=1.0, exit_zscore=0.5, allow_execution=True,
                max_reference_lag_ms=1_000,
            )
            for index in range(12):
                market_p = 0.50 + (0.01 if index % 2 == 0 else -0.01)
                model.observe(
                    CryptoObservation("m1", market_p, 0.5, history_time + index,
                                      market_timestamp_ms=history_time + index),
                    history_time + index,
                )
            buy = asyncio.run(process_crypto_quote(
                runner, model,
                CryptoReferenceQuote("m1", history_time + 12, 0.5, source="fixture"),
                now_ms=history_time + 12,
            ))
            self.assertEqual(buy.action, "ENTER")
            self.assertTrue(buy.executable)
            self.assertIn("yes-token", journal.inventory_by_token())
            book.asks = {0.50: 20.0}
            book.bids = {0.49: 20.0}
            book.timestamp_ms = history_time + 13
            exit_signal = asyncio.run(process_crypto_quote(
                runner, model,
                CryptoReferenceQuote("m1", history_time + 13, 0.5, source="fixture"),
                now_ms=history_time + 13,
            ))
            self.assertEqual(exit_signal.action, "EXIT")
            self.assertEqual(journal.inventory_by_token(), {})
            self.assertIsNone(model.inventory.get("m1"))


class ScanRejectAndGammaTests(unittest.TestCase):
    def test_gamma_market_fetch_uses_volume24hr_order_param(self):
        from polywang.arbitrage_bot import GAMMA_VOLUME_ORDER, _fetch_gamma_rows
        self.assertEqual(GAMMA_VOLUME_ORDER, "volume24hr")
        seen = []

        def getter(params):
            seen.append(dict(params))
            return []

        _fetch_gamma_rows(5, getter)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["order"], "volume24hr")
        self.assertNotEqual(seen[0]["order"], "volume_24hr")

    def test_fetch_markets_query_order_is_volume24hr(self):
        from polywang.arbitrage_bot import fetch_markets

        def getter(params):
            self.assertEqual(params["order"], "volume24hr")
            return [{
                "id": "bin", "conditionId": "cb", "question": "Binary",
                "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]', "category": "geopolitics",
                "active": True, "closed": False,
            }]

        selected = fetch_markets(1, get=getter, pool=1)
        self.assertEqual([market.market_id for market in selected], ["bin"])

    def test_default_floors_and_buffer_unchanged(self):
        scanner = BinaryArbitrageScanner()
        self.assertEqual(scanner.min_net_profit_usd, 0.05)
        self.assertEqual(scanner.min_return, 0.002)
        self.assertEqual(scanner.safety_buffer_usd, 0.02)

    def test_scan_reject_counter_flush_sums_to_attempts_and_logs_touch(self):
        from polywang.arbitrage_bot import ScanRejectCounter
        lines = []

        class Capture:
            def info(self, message, *args):
                lines.append(message % args if args else message)

        counter = ScanRejectCounter(flush_interval_s=3600.0, logger=Capture())
        counter.note_dual_synced("m1")
        counter.observe_touch_sum(0.97)
        counter.observe_touch_sum(0.99)
        counter.observe_net(-0.04)
        counter.record("net_below_floor")
        counter.record("stale_book")
        counter.record("leg_skew")
        self.assertEqual(counter.attempts, 3)
        self.assertEqual(sum(counter.counts.values()), 3)
        counter.flush()
        self.assertEqual(len(lines), 1)
        self.assertIn("attempts=3", lines[0])
        self.assertIn("rejects=3", lines[0])
        self.assertIn("accepted=0", lines[0])
        self.assertIn("dual_synced_markets=1", lines[0])
        self.assertIn("best_yes_ask+no_ask=0.9700", lines[0])
        self.assertIn("best_net=-0.0400", lines[0])
        self.assertIn("net_below_floor=1", lines[0])
        self.assertIn("stale_book=1", lines[0])
        self.assertIn("leg_skew=1", lines[0])
        self.assertEqual(counter.attempts, 0)

    def test_paper_process_logs_rejects_and_does_not_write_live_orders(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_LEG_SKEW_MS": "500"}, clear=False,
        ):
            ledger = os.path.join(directory, "ledger.json")
            live_orders = os.path.join(directory, "live-orders.json")
            market = BinaryMarket("m1", "c1", "Test", "yes-token", "no-token")
            # Default floors: fee drag leaves net below 0.05 on a 0.48+0.48 book.
            runner = PaperMarketRunner(
                [market], ledger, 100.0, BinaryArbitrageScanner(),
            )
            runner.max_book_age_seconds = 1e9
            runner.scan_rejects.flush_interval_s = 3600.0
            now = int(__import__("time").time() * 1000)
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "yes-token", "timestamp": str(now),
                "hash": "y", "asks": [{"price": "0.49", "size": "10"}], "bids": [],
            }))
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "no-token", "timestamp": str(now),
                "hash": "n", "asks": [{"price": "0.49", "size": "10"}], "bids": [],
            }))
            self.assertEqual(runner.ledger.state["positions"], {})
            self.assertGreater(runner.scan_rejects.attempts, 0)
            self.assertEqual(
                sum(runner.scan_rejects.counts.values()) + runner.scan_rejects.accepted,
                runner.scan_rejects.attempts,
            )
            self.assertGreater(runner.scan_rejects.counts["net_below_floor"], 0)
            self.assertIsNotNone(runner.scan_rejects.best_touch_sum)
            self.assertAlmostEqual(runner.scan_rejects.best_touch_sum, 0.98)
            self.assertEqual(runner.scan_rejects.dual_synced_markets, {"m1"})
            with self.assertLogs("arbitrage-bot", level="INFO") as captured:
                runner.scan_rejects.flush()
            self.assertTrue(any("SCAN REJECTS:" in line for line in captured.output))
            self.assertTrue(any("best_yes_ask+no_ask=" in line for line in captured.output))
            self.assertFalse(os.path.exists(live_orders))
            self.assertNotEqual(ledger, live_orders)

    def test_paper_process_counts_leg_skew(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "MAX_LEG_SKEW_MS": "500"}, clear=False,
        ):
            market = BinaryMarket("m1", "c1", "Test", "yes-token", "no-token")
            runner = PaperMarketRunner(
                [market], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
            )
            runner.max_book_age_seconds = 1e9
            runner.scan_rejects.flush_interval_s = 3600.0
            now = int(__import__("time").time() * 1000)
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "yes-token", "timestamp": str(now),
                "hash": "y", "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            }))
            asyncio.run(runner.process({
                "event_type": "book", "asset_id": "no-token", "timestamp": str(now + 2000),
                "hash": "n", "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            }))
            self.assertEqual(runner.scan_rejects.counts["leg_skew"], 1)
            self.assertEqual(runner.scan_rejects.attempts, 1)
            self.assertIsNotNone(runner.scan_rejects.best_touch_sum)


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t


class _FakeMarketSocket:
    """Async websocket stand-in that records subscribe frames and yields scripted messages."""

    def __init__(self, messages=None, hang: bool = True):
        self._messages = [json.dumps(item) if isinstance(item, (dict, list)) else item
                          for item in (messages or [])]
        self.sent = []
        self.closed = False
        self.hang = hang

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        if not self.hang:
            raise RuntimeError("fake socket closed")
        await asyncio.Future()


class _RecordingConnect:
    def __init__(self, message_factory=None):
        self.calls = []
        self.sockets = []
        self.subscribes = []
        self._message_factory = message_factory or (lambda _index: [])

    def __call__(self, url, **kwargs):
        index = len(self.calls)
        self.calls.append({"url": url, "kwargs": dict(kwargs)})
        socket = _FakeMarketSocket(self._message_factory(index))
        original_send = socket.send

        async def send(data):
            await original_send(data)
            self.subscribes.append(json.loads(data))

        socket.send = send
        self.sockets.append(socket)
        return socket


class _FakeStreamRunner:
    def __init__(self):
        self.processed = []
        self.invalidated = []

    def invalidate_books(self, token_ids=None):
        self.invalidated.append(list(token_ids) if token_ids is not None else None)

    async def process(self, event):
        self.processed.append(event)


class MarketStreamSubscribeTests(unittest.TestCase):
    def _ids(self, count):
        return [f"tok-{index}" for index in range(count)]

    async def _run_until(self, coro, predicate, timeout=2.0):
        task = asyncio.create_task(coro)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                if task.done():
                    task.result()
                    raise AssertionError("market stream ended before the wait condition")
                await asyncio.sleep(0.01)
            raise AssertionError("timed out waiting for the market-stream condition")
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def test_chunk_asset_ids_never_exceeds_50_and_covers_200(self):
        from polywang.arbitrage_bot import MARKET_WS_MAX_ASSETS_PER_SUBSCRIBE, chunk_asset_ids
        self.assertEqual(MARKET_WS_MAX_ASSETS_PER_SUBSCRIBE, 50)
        self.assertEqual(chunk_asset_ids([]), [])
        fifty = chunk_asset_ids(self._ids(50))
        self.assertEqual(len(fifty), 1)
        self.assertEqual(len(fifty[0]), 50)
        two_hundred = chunk_asset_ids(self._ids(200))
        self.assertEqual(len(two_hundred), 4)
        self.assertTrue(all(len(chunk) <= 50 for chunk in two_hundred))
        self.assertEqual(sum(len(chunk) for chunk in two_hundred), 200)
        self.assertEqual([len(chunk) for chunk in chunk_asset_ids(self._ids(51))], [50, 1])

    def test_idle_watch_books_only_dump_is_idle_after_15s(self):
        from polywang.arbitrage_bot import MARKET_WS_IDLE_RECONNECT_SECONDS, MarketStreamIdleWatch
        self.assertEqual(MARKET_WS_IDLE_RECONNECT_SECONDS, 15.0)
        clock = _FakeClock(0.0)
        watch = MarketStreamIdleWatch(15.0, now=clock)
        watch.mark_subscribed()
        self.assertFalse(watch.note({"event_type": "book", "asset_id": "yes"}))
        self.assertFalse(watch.note({"event_type": "book", "asset_id": "no"}))
        clock.t = 14.9
        self.assertFalse(watch.is_idle())
        clock.t = 15.0
        self.assertTrue(watch.is_idle())
        self.assertEqual(watch.last_event_type, "book")
        self.assertAlmostEqual(watch.last_event_age(), 15.0)

    def test_idle_watch_price_change_and_later_book_reset_timer(self):
        from polywang.arbitrage_bot import MarketStreamIdleWatch
        clock = _FakeClock(0.0)
        watch = MarketStreamIdleWatch(15.0, now=clock)
        watch.mark_subscribed()
        watch.note({"event_type": "book", "asset_id": "yes"})
        clock.t = 5.0
        self.assertTrue(watch.note({
            "event_type": "price_change",
            "price_changes": [{"asset_id": "yes", "price": "0.40", "size": "1", "side": "SELL"}],
        }))
        clock.t = 19.9
        self.assertFalse(watch.is_idle())
        clock.t = 20.0
        self.assertTrue(watch.note({"event_type": "book", "asset_id": "yes"}))
        clock.t = 34.9
        self.assertFalse(watch.is_idle())

    def test_one_connection_never_receives_more_than_50_ids(self):
        from polywang.arbitrage_bot import MARKET_WS_URL, run_market_stream
        connect = _RecordingConnect()
        runner = _FakeStreamRunner()
        asyncio.run(self._run_until(
            run_market_stream(runner, self._ids(51), connect=connect, idle_seconds=3600.0),
            lambda: len(connect.subscribes) >= 2,
        ))
        sizes = [len(payload["assets_ids"]) for payload in connect.subscribes]
        self.assertTrue(sizes)
        self.assertTrue(all(size <= 50 for size in sizes))
        self.assertEqual(max(sizes), 50)
        self.assertEqual(sum(sizes), 51)
        self.assertTrue(all(payload["type"] == "market" for payload in connect.subscribes))
        self.assertTrue(all(call["url"] == MARKET_WS_URL for call in connect.calls))
        self.assertTrue(all(call["kwargs"]["ping_interval"] == 10 for call in connect.calls))
        self.assertTrue(all(call["kwargs"]["ping_timeout"] == 10 for call in connect.calls))

    def test_200_ids_open_multiple_connections_of_at_most_50(self):
        from polywang.arbitrage_bot import run_market_stream
        connect = _RecordingConnect()
        runner = _FakeStreamRunner()
        token_ids = self._ids(200)
        asyncio.run(self._run_until(
            run_market_stream(runner, token_ids, connect=connect, idle_seconds=3600.0),
            lambda: len(connect.subscribes) >= 4,
        ))
        self.assertGreaterEqual(len(connect.calls), 4)
        self.assertEqual(len(connect.subscribes), 4)
        subscribed = []
        for payload in connect.subscribes:
            self.assertLessEqual(len(payload["assets_ids"]), 50)
            self.assertEqual(payload["type"], "market")
            subscribed.extend(payload["assets_ids"])
        self.assertEqual(subscribed, token_ids)
        self.assertEqual(len(runner.invalidated), 4)

    def test_idle_after_books_only_dump_reconnects_and_logs(self):
        from polywang.arbitrage_bot import run_market_stream

        books = [
            {"event_type": "book", "asset_id": "tok-0", "asks": [], "bids": []},
            {"event_type": "book", "asset_id": "tok-1", "asks": [], "bids": []},
        ]

        def messages(index):
            return list(books) if index == 0 else []

        connect = _RecordingConnect(messages)
        runner = _FakeStreamRunner()
        with self.assertLogs("arbitrage-bot", level="WARNING") as captured:
            asyncio.run(self._run_until(
                run_market_stream(
                    runner, ["tok-0", "tok-1"],
                    connect=connect, idle_seconds=0.05,
                ),
                lambda: len(connect.calls) >= 2 and any(
                    "silent" in line and "last_event=book" in line for line in captured.output
                ),
            ))
        self.assertGreaterEqual(len(connect.calls), 2)
        self.assertTrue(connect.sockets[0].closed)
        self.assertEqual([event["event_type"] for event in runner.processed], ["book", "book"])
        self.assertTrue(any(
            "shard 0 silent" in line and "tokens=2" in line and "last_event=book" in line
            for line in captured.output
        ))

    def test_price_change_after_books_is_processed_and_keeps_socket(self):
        from polywang.arbitrage_bot import run_market_stream

        live_tape = [
            {"event_type": "book", "asset_id": "tok-0", "asks": [], "bids": []},
            {
                "event_type": "price_change", "timestamp": "1",
                "price_changes": [{"asset_id": "tok-0", "price": "0.41", "size": "2", "side": "SELL"}],
            },
        ]
        connect = _RecordingConnect(lambda _index: live_tape)
        runner = _FakeStreamRunner()
        asyncio.run(self._run_until(
            run_market_stream(
                runner, ["tok-0"], connect=connect, idle_seconds=0.2,
            ),
            lambda: any(event.get("event_type") == "price_change" for event in runner.processed),
        ))
        self.assertEqual(len(connect.calls), 1)
        self.assertEqual(runner.processed[-1]["event_type"], "price_change")

    def test_jsonl_recorder_and_process_see_every_event(self):
        from polywang.arbitrage_bot import run_market_stream
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "market-events.jsonl")
            connect = _RecordingConnect(lambda _index: [
                {"event_type": "book", "asset_id": "tok-0", "asks": [], "bids": []},
                {
                    "event_type": "price_change", "timestamp": "1",
                    "price_changes": [{"asset_id": "tok-0", "price": "0.41", "size": "2", "side": "SELL"}],
                },
            ])
            runner = _FakeStreamRunner()
            def recorded_rows():
                if not os.path.isfile(path):
                    return 0
                with open(path, encoding="utf-8") as handle:
                    return sum(1 for line in handle if line.strip())

            with mock.patch.dict(os.environ, {"MARKET_EVENT_LOG": path}, clear=False):
                asyncio.run(self._run_until(
                    run_market_stream(runner, ["tok-0"], connect=connect, idle_seconds=3600.0),
                    lambda: recorded_rows() >= 2,
                ))
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual([row["event_type"] for row in rows], ["book", "price_change"])
            self.assertEqual(rows[0]["source"], "market")
            self.assertEqual([event["event_type"] for event in runner.processed], ["book", "price_change"])

    def test_invalidate_books_only_drops_the_reconnecting_shard(self):
        from polywang.arbitrage_core import OrderBook
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            first = BinaryMarket("m1", "c1", "One", "yes-a", "no-a")
            second = BinaryMarket("m2", "c2", "Two", "yes-b", "no-b")
            runner = PaperMarketRunner(
                [first, second], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(),
            )
            for token_id in ("yes-a", "no-a", "yes-b", "no-b"):
                book = OrderBook()
                book.asks = {0.40: 10.0}
                book.synced = True
                runner.books[token_id] = book
            runner.invalidate_books(["yes-a", "no-a"])
            self.assertFalse(runner.books["yes-a"].synced)
            self.assertFalse(runner.books["no-a"].synced)
            self.assertTrue(runner.books["yes-b"].synced)
            self.assertTrue(runner.books["no-b"].synced)


if __name__ == "__main__":
    unittest.main()
