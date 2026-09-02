#!/usr/bin/env python3
"""Independent NegRisk complete-set path. No network."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from polywang.arbitrage_bot import (
    PaperMarketRunner,
    fetch_markets,
    fetch_universe,
    resolve_negrisk_journal_path,
)
from polywang.arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    LiveOrderJournal,
    LiveRiskController,
    OfficialFOKExecutor,
    OrderBook,
    RiskHaltError,
    UnhedgedPairError,
)
from polywang.negrisk import (
    LiveNegRiskJournal,
    NegRiskBookScanner,
    NegRiskMarket,
    OfficialNegRiskExecutor,
    PaperNegRiskExecutor,
    collect_event_lookups,
    fetch_complete_negrisk_events,
    negrisk_execution_enabled,
    parse_negrisk_markets,
)


def nway_payload(**overrides):
    row = {
        "id": "nr1",
        "conditionId": "cnr",
        "question": "Who wins",
        "clobTokenIds": '["tok-a", "tok-b", "tok-c"]',
        "outcomes": '["A", "B", "C"]',
        "outcomePrices": '["0.20", "0.20", "0.20"]',
        "category": "geopolitics",
        "active": True,
        "closed": False,
    }
    row.update(overrides)
    return row


def child_binary(market_id, yes, no, title, price="0.20"):
    implied_no = f"{1.0 - float(price):.2f}"
    return {
        "id": market_id,
        "conditionId": f"c-{market_id}",
        "question": title,
        "clobTokenIds": json.dumps([yes, no]),
        "outcomes": '["Yes", "No"]',
        "outcomePrices": json.dumps([price, implied_no]),
        "category": "geopolitics",
        "active": True,
        "closed": False,
        "groupItemTitle": title,
        "negRisk": True,
    }


def synced_book(asks, timestamp_ms=1, digest=""):
    book = OrderBook()
    book.asks = dict(asks)
    book.synced = True
    book.timestamp_ms = timestamp_ms
    book.hash = digest or str(asks)
    return book


class FakeNegRiskClient:
    def __init__(self, fail_token=None, raise_token=None, rollback_ok=True, fill="10"):
        self.fail_token = fail_token
        self.raise_token = raise_token
        self.rollback_ok = rollback_ok
        self.fill = fill
        self.calls = []

    async def get_balance_allowance(self, **kwargs):
        return {"balance": "1000", "allowance": "1000"}

    async def place_market_order(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["side"] == "SELL":
            return {
                "ok": self.rollback_ok,
                "order_id": "rb",
                "making_amount": kwargs.get("shares", self.fill),
            }
        token = kwargs["token_id"]
        if token == self.raise_token:
            raise TimeoutError("request timed out after submission")
        if token == self.fail_token:
            return {"ok": False, "order_id": f"bad-{token}", "taking_amount": "0"}
        return {"ok": True, "order_id": f"ord-{token}", "taking_amount": self.fill}


class ParseTests(unittest.TestCase):
    def test_nway_market_is_a_complete_field(self):
        market = NegRiskMarket.from_gamma(nway_payload())
        self.assertIsNotNone(market)
        self.assertEqual(market.yes_token_ids, ("tok-a", "tok-b", "tok-c"))
        self.assertFalse(market.has_no_tokens)
        self.assertEqual(market.source, "nway")

    def test_yes_no_binary_is_not_a_negrisk_field(self):
        payload = {
            "id": "m1", "conditionId": "c1", "question": "Yes or no",
            "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
            "negRisk": True, "active": True, "closed": False,
        }
        self.assertIsNone(NegRiskMarket.from_gamma(payload))
        binary = BinaryMarket.from_gamma(payload)
        self.assertIsNotNone(binary)
        self.assertTrue(binary.neg_risk)

    def test_complete_event_with_nested_markets_has_no_tokens(self):
        payload = {
            "id": "evt",
            "title": "Election",
            "active": True,
            "closed": False,
            "category": "geopolitics",
            "markets": [
                child_binary("m-a", "ya", "na", "A"),
                child_binary("m-b", "yb", "nb", "B"),
                child_binary("m-c", "yc", "nc", "C"),
            ],
        }
        market = NegRiskMarket.from_gamma(payload)
        self.assertIsNotNone(market)
        self.assertTrue(market.has_no_tokens)
        self.assertEqual(market.source, "event")
        self.assertEqual(market.yes_token_ids, ("ya", "yb", "yc"))

    def test_scattered_volume_pool_binaries_are_not_grouped(self):
        rows = [
            child_binary("m-a", "ya", "na", "A"),
            child_binary("m-b", "yb", "nb", "B"),
        ]
        self.assertEqual(parse_negrisk_markets(rows), [])

    def test_scattered_binaries_become_a_field_only_after_event_fetch(self):
        rows = [
            {**child_binary("m-a", "ya", "na", "A"), "eventId": "evt-1"},
            {**child_binary("m-b", "yb", "nb", "B"), "eventId": "evt-1"},
        ]
        self.assertEqual(collect_event_lookups(rows), [("id", "evt-1")])
        self.assertEqual(parse_negrisk_markets(rows), [])

        def get_event(kind, value):
            self.assertEqual((kind, value), ("id", "evt-1"))
            return {
                "id": "evt",
                "title": "Election",
                "active": True,
                "closed": False,
                "category": "geopolitics",
                "markets": [
                    child_binary("m-a", "ya", "na", "A"),
                    child_binary("m-b", "yb", "nb", "B"),
                    child_binary("m-c", "yc", "nc", "C"),
                ],
            }

        events = fetch_complete_negrisk_events(collect_event_lookups(rows), get_event)
        markets = parse_negrisk_markets(rows, extra_events=events)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].source, "event")
        self.assertEqual(markets[0].yes_token_ids, ("ya", "yb", "yc"))

        binary, negrisk = fetch_universe(
            5, get=lambda params: rows, pool=5, negrisk_limit=5, get_event=get_event,
        )
        self.assertEqual(binary, [])
        self.assertEqual([market.market_id for market in negrisk], ["evt"])

        empty, still_empty = fetch_universe(
            5, get=lambda params: rows, pool=5, negrisk_limit=0,
        )
        self.assertEqual(empty, [])
        self.assertEqual(still_empty, [])


class BookScannerTests(unittest.TestCase):
    def setUp(self):
        self.market = NegRiskMarket.from_gamma(nway_payload())
        self.scanner = NegRiskBookScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
        )

    def test_buy_all_yes_when_asks_sum_below_one(self):
        books = {
            "tok-a": synced_book({0.20: 10}),
            "tok-b": synced_book({0.20: 10}),
            "tok-c": synced_book({0.20: 10}),
        }
        opportunity = self.scanner.scan(self.market, books)
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.direction, "BUY_ALL_YES")
        self.assertEqual(opportunity.shares, 10)
        self.assertAlmostEqual(opportunity.gross_profit, 4.0, places=6)
        self.assertFalse(opportunity.is_risk_free)
        self.assertEqual(len(opportunity.legs), 3)

    def test_buy_all_no_is_skipped_without_no_tokens(self):
        books = {
            "tok-a": synced_book({0.90: 10}),
            "tok-b": synced_book({0.90: 10}),
            "tok-c": synced_book({0.90: 10}),
        }
        # YES asks sum to 2.70, so BUY_ALL_YES is not profitable; NO tokens
        # do not exist, so the scanner must stand down rather than invent them.
        self.assertIsNone(self.scanner.scan(self.market, books))

    def test_buy_all_no_uses_child_no_tokens_on_a_complete_event(self):
        market = NegRiskMarket.from_gamma({
            "id": "evt", "title": "Election", "active": True, "closed": False,
            "category": "geopolitics",
            "markets": [
                child_binary("m-a", "ya", "na", "A", "0.40"),
                child_binary("m-b", "yb", "nb", "B", "0.40"),
                child_binary("m-c", "yc", "nc", "C", "0.40"),
            ],
        })
        books = {
            "ya": synced_book({0.40: 10}),
            "yb": synced_book({0.40: 10}),
            "yc": synced_book({0.40: 10}),
            "na": synced_book({0.20: 10}),
            "nb": synced_book({0.20: 10}),
            "nc": synced_book({0.20: 10}),
        }
        opportunity = self.scanner.scan(market, books)
        self.assertEqual(opportunity.direction, "BUY_ALL_NO")
        # Payout is n-1 = 2.00; cost is 0.60; gross 1.40 per share.
        self.assertAlmostEqual(opportunity.payout_per_share, 2.0, places=9)
        self.assertGreater(opportunity.net_profit, 0.0)

    def test_unsynced_or_missing_leg_is_not_scanned(self):
        books = {
            "tok-a": synced_book({0.20: 10}),
            "tok-b": synced_book({0.20: 10}),
        }
        self.assertIsNone(self.scanner.scan(self.market, books))
        books["tok-c"] = synced_book({0.20: 10})
        books["tok-c"].synced = False
        self.assertIsNone(self.scanner.scan(self.market, books))

    def test_merge_gas_can_kill_the_edge(self):
        books = {
            "tok-a": synced_book({0.20: 10}),
            "tok-b": synced_book({0.20: 10}),
            "tok-c": synced_book({0.20: 10}),
        }
        costly = NegRiskBookScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0, merge_gas_usd=5.0,
        )
        self.assertIsNone(costly.scan(self.market, books))


class ExecutorTests(unittest.TestCase):
    def opportunity(self):
        market = NegRiskMarket.from_gamma(nway_payload())
        books = {
            "tok-a": synced_book({0.20: 10}),
            "tok-b": synced_book({0.20: 10}),
            "tok-c": synced_book({0.20: 10}),
        }
        return NegRiskBookScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
        ).scan(market, books)

    def test_sequential_fok_buys_every_yes_leg(self):
        client = FakeNegRiskClient()
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            transport = OfficialFOKExecutor(client)
            result = asyncio.run(OfficialNegRiskExecutor(transport, journal).execute(self.opportunity()))
        self.assertEqual(result.status, "ASSEMBLED")
        self.assertEqual([call["order_type"] for call in client.calls], ["FOK", "FOK", "FOK"])
        self.assertEqual([call["token_id"] for call in client.calls], ["tok-a", "tok-b", "tok-c"])
        self.assertTrue(all(call["side"] == "BUY" for call in client.calls))

    def test_later_leg_failure_unwinds_filled_legs_with_fak(self):
        client = FakeNegRiskClient(fail_token="tok-c")
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            transport = OfficialFOKExecutor(client)
            with self.assertRaises(RuntimeError):
                asyncio.run(OfficialNegRiskExecutor(transport, journal).execute(self.opportunity()))
            record = list(journal.state["baskets"].values())[0]
            self.assertEqual(record["status"], "UNWOUND")
        sides = [call["side"] for call in client.calls]
        self.assertEqual(sides[:3], ["BUY", "BUY", "BUY"])
        self.assertEqual(sides[3:], ["SELL", "SELL"])
        self.assertTrue(all(call["order_type"] == "FAK" for call in client.calls[3:]))

    def test_unknown_first_leg_is_unhedged(self):
        client = FakeNegRiskClient(raise_token="tok-a")
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            transport = OfficialFOKExecutor(client)
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(OfficialNegRiskExecutor(transport, journal).execute(self.opportunity()))
            record = list(journal.state["baskets"].values())[0]
            self.assertEqual(record["status"], "UNHEDGED")

    def test_binary_fok_executor_is_not_the_negrisk_path(self):
        opportunity = self.opportunity()
        with self.assertRaises(ValueError):
            asyncio.run(OfficialFOKExecutor(FakeNegRiskClient()).execute(opportunity))


class RiskAndRunnerTests(unittest.TestCase):
    def test_startup_halts_on_unfinished_basket(self):
        with tempfile.TemporaryDirectory() as directory:
            nr_path = os.path.join(directory, "nr.json")
            pair_path = os.path.join(directory, "pairs.json")
            nr = LiveNegRiskJournal(nr_path)
            opportunity = ExecutorTests().opportunity()
            basket_id = nr.create_basket(opportunity)
            nr.set_leg_order(basket_id, "tok-a", "ord-a", 10)
            nr.set_status(basket_id, "PARTIAL")
            pairs = LiveOrderJournal(pair_path)
            risk = LiveRiskController(
                pairs, equity_usd=1000.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=os.path.join(directory, "missing-kill"),
                negrisk_journal=nr,
            )
            with self.assertRaises(RiskHaltError):
                risk.check_startup()

    def test_assembled_inventory_is_known_to_live_recon(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            client = FakeNegRiskClient()
            transport = OfficialFOKExecutor(client)
            asyncio.run(OfficialNegRiskExecutor(transport, journal).execute(ExecutorTests().opportunity()))
            self.assertIn("tok-a", transport._known_live_token_ids())
            self.assertIn("ord-tok-a", transport._known_live_order_ids())

    def test_paper_runner_fills_a_complete_set_without_touching_binary_fok(self):
        market = NegRiskMarket.from_gamma(nway_payload())
        binary = BinaryMarket("m1", "c1", "Binary", "yes-token", "no-token", category="geopolitics")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            nr_journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            nr_exec = PaperNegRiskExecutor(nr_journal)
            runner = PaperMarketRunner(
                [binary], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
                negrisk_markets=[market],
                negrisk_scanner=NegRiskBookScanner(
                    min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
                ),
                negrisk_executor=nr_exec,
            )
            nr_exec.ledger = runner.ledger
            runner.max_book_age_seconds = 1e9
            now = int(__import__("time").time() * 1000)
            for token in ("tok-a", "tok-b", "tok-c"):
                asyncio.run(runner.process({
                    "event_type": "book", "asset_id": token, "timestamp": str(now),
                    "hash": token, "asks": [{"price": "0.20", "size": "10"}], "bids": [],
                }))
            baskets = list(nr_journal.state["baskets"].values())
            self.assertEqual(len(baskets), 1)
            self.assertEqual(baskets[0]["status"], "ASSEMBLED")
            self.assertEqual(baskets[0]["direction"], "BUY_ALL_YES")
            self.assertTrue(any(pos.get("kind") == "negrisk" for pos in runner.ledger.state["positions"].values()))
            self.assertEqual(runner.live_journal, None)
            live_orders = os.path.join(directory, "live-orders.json")
            self.assertFalse(os.path.exists(live_orders))
            self.assertNotEqual(os.path.basename(nr_journal.path), "live-orders.json")

    def test_paper_negrisk_journal_default_is_not_live_orders(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPER_NEGRISK_JOURNAL", None)
            os.environ.pop("LIVE_NEGRISK_JOURNAL", None)
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            paper_path = resolve_negrisk_journal_path(False)
            self.assertEqual(paper_path, "paper-negrisk.json")
            self.assertNotEqual(paper_path, "live-orders.json")
            self.assertNotEqual(resolve_negrisk_journal_path(True), "live-orders.json")
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
            with self.assertRaises(ValueError):
                resolve_negrisk_journal_path(False, "live-orders.json")
            with self.assertRaises(ValueError):
                resolve_negrisk_journal_path(False, "/tmp/live-orders.json")

    def test_paper_logs_under_one_negrisk_basket_without_live_orders(self):
        market = NegRiskMarket.from_gamma(nway_payload())
        binary = BinaryMarket("m1", "c1", "Binary", "yes-token", "no-token", category="geopolitics")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            journal_path = os.path.join(directory, "paper-negrisk.json")
            live_orders = os.path.join(directory, "live-orders.json")
            nr_journal = LiveNegRiskJournal(journal_path)
            nr_exec = PaperNegRiskExecutor(nr_journal)
            runner = PaperMarketRunner(
                [binary], os.path.join(directory, "ledger.json"), 100.0,
                BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
                negrisk_markets=[market],
                negrisk_scanner=NegRiskBookScanner(
                    min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
                ),
                negrisk_executor=nr_exec,
            )
            nr_exec.ledger = runner.ledger
            runner.max_book_age_seconds = 1e9
            now = int(__import__("time").time() * 1000)
            with self.assertLogs("arbitrage-bot", level="INFO") as captured:
                for token in ("tok-a", "tok-b", "tok-c"):
                    asyncio.run(runner.process({
                        "event_type": "book", "asset_id": token, "timestamp": str(now),
                        "hash": token, "asks": [{"price": "0.20", "size": "10"}], "bids": [],
                    }))
            baskets = list(nr_journal.state["baskets"].values())
            self.assertEqual(len(baskets), 1)
            self.assertEqual(baskets[0]["status"], "ASSEMBLED")
            # 0.20 + 0.20 + 0.20 = 0.60 < 1.00 complete-set payout
            self.assertLess(sum(0.20 for _ in ("A", "B", "C")), 1.0)
            self.assertTrue(any("NEGRISK PAPER" in line for line in captured.output))
            self.assertTrue(os.path.exists(journal_path))
            self.assertFalse(os.path.exists(live_orders))
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
            self.assertEqual(runner.live_journal, None)

    def test_fetch_markets_still_excludes_negrisk(self):
        rows = [
            nway_payload(),
            {
                "id": "bin", "conditionId": "cb", "question": "Binary",
                "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]', "category": "geopolitics",
                "active": True, "closed": False,
            },
        ]
        selected = fetch_markets(5, get=lambda params: rows, pool=5)
        self.assertEqual([market.market_id for market in selected], ["bin"])
        binary, negrisk = fetch_universe(5, get=lambda params: rows, pool=5, negrisk_limit=5)
        self.assertEqual([market.market_id for market in binary], ["bin"])
        self.assertEqual([market.market_id for market in negrisk], ["nr1"])

    def test_startup_allows_assembled_and_pending_redemption(self):
        with tempfile.TemporaryDirectory() as directory:
            nr = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            opportunity = ExecutorTests().opportunity()
            assembled = nr.create_basket(opportunity)
            nr.set_status(assembled, "ASSEMBLED")
            pending = nr.create_basket(opportunity)
            nr.set_status(pending, "RESOLVED_PENDING_REDEMPTION")
            converting = nr.create_basket(opportunity)
            nr.set_status(converting, "CONVERT_SUBMITTED")
            pairs = LiveOrderJournal(os.path.join(directory, "pairs.json"))
            risk = LiveRiskController(
                pairs, equity_usd=1000.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=os.path.join(directory, "missing-kill"),
                negrisk_journal=nr,
            )
            risk.check_startup()

    def test_resolution_redeem_settles_and_convert_is_fail_closed(self):
        class RedeemHandle:
            transaction_hash = "0xredeem"
            transaction_id = "rtx"

            async def wait(self):
                return self

        class RedeemOnlyClient(FakeNegRiskClient):
            def __init__(self):
                super().__init__()
                self.redeem_calls = []

            async def redeem_positions(self, **kwargs):
                self.redeem_calls.append(kwargs)
                return RedeemHandle()

        opportunity = ExecutorTests().opportunity()
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveNegRiskJournal(os.path.join(directory, "nr.json"))
            client = RedeemOnlyClient()
            transport = OfficialFOKExecutor(client)
            executor = OfficialNegRiskExecutor(
                transport, journal, auto_convert=True, auto_redeem=True,
            )
            asyncio.run(executor.execute(opportunity))
            basket = list(journal.state["baskets"].values())[0]
            self.assertEqual(basket["status"], "ASSEMBLED")
            with self.assertRaises(RuntimeError) as raised:
                asyncio.run(executor.convert_basket(basket))
            self.assertIn("convert_positions", str(raised.exception))
            self.assertEqual(journal._record(basket["basket_id"])["status"], "ASSEMBLED")
            self.assertEqual(journal.mark_resolved(opportunity.market_id, "A"), 1)
            settled = asyncio.run(executor.settle_baskets())
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["status"], "SETTLED")
            self.assertEqual(settled[0]["settlement_type"], "REDEEM")
            self.assertEqual(client.redeem_calls, [{"condition_id": "cnr"}])

    def test_execution_flags_paper_default_on_live_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_NEGRISK_EXECUTION", None)
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            self.assertTrue(negrisk_execution_enabled(False))
            self.assertFalse(negrisk_execution_enabled(True))
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
        with mock.patch.dict(os.environ, {"ENABLE_NEGRISK_EXECUTION": "0"}, clear=False):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            self.assertFalse(negrisk_execution_enabled(False))
            self.assertFalse(negrisk_execution_enabled(True))
        with mock.patch.dict(os.environ, {"ENABLE_NEGRISK_EXECUTION": "1"}, clear=False):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            self.assertTrue(negrisk_execution_enabled(False))
            self.assertFalse(negrisk_execution_enabled(True))
        with mock.patch.dict(os.environ, {
            "ENABLE_NEGRISK_EXECUTION": "1", "ENABLE_NEGRISK_LIVE": "1",
        }, clear=False):
            self.assertTrue(negrisk_execution_enabled(True))

    def test_paper_negrisk_cannot_reuse_the_same_resting_level(self):
        market = NegRiskMarket.from_gamma(nway_payload())
        binary = BinaryMarket("m1", "c1", "Binary", "yes-token", "no-token", category="geopolitics")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            os.environ.pop("POLYMARKET_LIVE_CONFIRM", None)
            nr_journal = LiveNegRiskJournal(os.path.join(directory, "paper-negrisk.json"))
            nr_exec = PaperNegRiskExecutor(nr_journal)
            runner = PaperMarketRunner(
                [binary], os.path.join(directory, "ledger.json"), 1000.0,
                BinaryArbitrageScanner(),
                negrisk_markets=[market],
                negrisk_scanner=NegRiskBookScanner(),
                negrisk_executor=nr_exec,
            )
            nr_exec.ledger = runner.ledger
            runner.max_book_age_seconds = 1e9
            now = int(__import__("time").time() * 1000)

            def feed(round_id):
                for token in ("tok-a", "tok-b", "tok-c"):
                    asyncio.run(runner.process({
                        "event_type": "book", "asset_id": token, "timestamp": str(now),
                        "hash": f"{token}-{round_id}",
                        "asks": [{"price": "0.20", "size": "10"}], "bids": [],
                    }))

            feed(1)
            self.assertEqual(len(nr_journal.state["baskets"]), 1)
            first_cash = float(runner.ledger.state["cash"])
            # QUANT-08 RCA: 13 identical CS snapshots in ~70ms, each with a new hash.
            for round_id in range(2, 14):
                feed(round_id)
            self.assertEqual(len(nr_journal.state["baskets"]), 1)
            self.assertAlmostEqual(float(runner.ledger.state["cash"]), first_cash)
            restored = synced_book({0.20: 10})
            runner.paper_ask_depth.apply_to_book("tok-a", restored)
            with self.assertRaisesRegex(ValueError, "insufficient paper book depth"):
                runner.paper_ask_depth.ensure_shares(restored, 10.0)
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
            self.assertNotEqual(os.getenv("POLYMARKET_LIVE_CONFIRM"), "I_UNDERSTAND_THE_RISK")
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.scanner.min_return, 0.002)
            self.assertEqual(runner.scanner.safety_buffer_usd, 0.02)
            self.assertEqual(runner.negrisk_scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.negrisk_scanner.min_return, 0.002)
            self.assertEqual(runner.negrisk_scanner.safety_buffer_usd, 0.02)


if __name__ == "__main__":
    unittest.main()
