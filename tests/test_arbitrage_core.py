import os
import tempfile
import unittest
import asyncio
import time
from types import SimpleNamespace
from unittest import mock

from polywang.arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    JsonLedger,
    LiveOrderJournal,
    MatchingEngineRestartError,
    CancelOnlyError,
    OrderBook,
    OfficialFOKExecutor,
    LiveRiskController,
    RiskHaltError,
    PaperArbitrageExecutor,
    UnhedgedPairError,
    handle_market_event,
    maker_gtc_enabled,
    maker_limit_price,
)


def market():
    return BinaryMarket("m1", "c1", "Test", "yes-token", "no-token", category="politics")


class MarketAndBookTests(unittest.TestCase):
    def test_gamma_market_parses_json_fields(self):
        parsed = BinaryMarket.from_gamma({
            "id": "7", "question": "Test", "clobTokenIds": '["y", "n"]',
            "outcomes": '["Yes", "No"]', "category": "politics", "active": True,
        })
        self.assertEqual(parsed.yes_token_id, "y")
        self.assertEqual(parsed.no_token_id, "n")

    def test_gamma_market_parses_market_specific_fee_schedule(self):
        parsed = BinaryMarket.from_gamma({
            "id": "7", "conditionId": "c7", "question": "Test",
            "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
            "feeSchedule": {"rate": "0.02", "exponent": 2},
        })
        self.assertAlmostEqual(parsed.taker_fee_rate, 0.02)
        self.assertEqual(parsed.fee_exponent, 2.0)

    def test_gamma_market_parses_implied_yes_from_outcome_prices(self):
        parsed = BinaryMarket.from_gamma({
            "id": "7", "conditionId": "c7", "question": "Test",
            "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.95", "0.05"]', "category": "politics",
        })
        self.assertAlmostEqual(parsed.implied_yes, 0.95)
        self.assertAlmostEqual(parsed.implied_no, 0.05)
        combo = BinaryMarket.from_gamma({
            "id": "8", "conditionId": "c8", "question": "Combo",
            "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.48", "0.48"]', "category": "geopolitics",
        })
        self.assertAlmostEqual(combo.implied_yes, 0.48)
        self.assertAlmostEqual(combo.implied_no, 0.48)
        missing = BinaryMarket.from_gamma({
            "id": "7", "conditionId": "c7", "question": "Test",
            "clobTokenIds": '["y", "n"]', "outcomes": '["Yes", "No"]',
        })
        self.assertIsNone(missing.implied_yes)
        self.assertIsNone(missing.implied_no)

    def test_gamma_market_rejects_incomplete_or_duplicate_identifiers(self):
        self.assertIsNone(BinaryMarket.from_gamma({
            "id": "", "conditionId": "c7", "clobTokenIds": '["y", "n"]',
            "outcomes": '["Yes", "No"]',
        }))
        self.assertIsNone(BinaryMarket.from_gamma({
            "id": "7", "conditionId": "c7", "clobTokenIds": '["same", "same"]',
            "outcomes": '["Yes", "No"]',
        }))

    def test_book_and_nested_price_change_are_applied(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        affected = handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1000",
            "hash": "a", "bids": [{"price": "0.40", "size": "5"}],
            "asks": [{"price": "0.45", "size": "5"}],
        }, mapping, books)
        self.assertEqual(affected, ["m1"])
        handle_market_event({
            "event_type": "price_change", "timestamp": "1001",
            "price_changes": [{"asset_id": "yes-token", "price": "0.44", "size": "2", "side": "SELL", "hash": "b"}],
        }, mapping, books)
        self.assertEqual(books["yes-token"].best_ask(), (0.44, 2.0))
        self.assertEqual(books["yes-token"].hash, "b")

    def test_invalid_price_change_side_is_ignored(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "timestamp": "1001",
            "price_changes": [{"asset_id": "yes-token", "price": "0.44", "size": "2", "side": "BAD"}],
        }, mapping, books), [])

    def test_sdk_style_typed_market_event_is_supported(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        event = SimpleNamespace(
            type="book",
            payload=SimpleNamespace(
                token_id="yes-token", timestamp=SimpleNamespace(timestamp=lambda: 1_700_000_000),
                hash="sdk-hash", bids=[SimpleNamespace(price="0.40", size="5")],
                asks=[SimpleNamespace(price="0.45", size="5")],
            ),
        )
        self.assertEqual(handle_market_event(event, mapping, books), ["m1"])
        self.assertEqual(books["yes-token"].best_ask(), (0.45, 5.0))

    def test_incremental_change_requires_snapshot_and_rejects_older_event(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "asset_id": "yes-token", "timestamp": "1001",
            "price_changes": [{"asset_id": "yes-token", "price": "0.44", "size": "2", "side": "SELL"}],
        }, mapping, books), [])
        self.assertEqual(handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1000",
            "hash": "a", "bids": [], "asks": [{"price": "0.45", "size": "5"}],
        }, mapping, books), ["m1"])
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "asset_id": "yes-token", "timestamp": "999",
            "price_changes": [{"asset_id": "yes-token", "price": "0.40", "size": "2", "side": "SELL"}],
        }, mapping, books), [])
        self.assertEqual(books["yes-token"].best_ask(), (0.45, 5.0))

    def test_sequence_gap_invalidates_the_book_instead_of_applying_a_partial_update(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1000",
            "sequence": 10, "hash": "a", "schema_version": "1",
            "bids": [], "asks": [{"price": "0.45", "size": "5"}],
        }, mapping, books)
        self.assertTrue(books["yes-token"].synced)
        affected = handle_market_event({
            "event_type": "price_change", "timestamp": "1002", "sequence": 12,
            "price_changes": [{"asset_id": "yes-token", "price": "0.40", "size": "9", "side": "SELL", "hash": "c"}],
        }, mapping, books)
        self.assertEqual(affected, [])
        book = books["yes-token"]
        self.assertFalse(book.synced)
        self.assertEqual(book.asks, {})
        self.assertIn("sequence gap", book.gap_reason)
        self.assertIsNone(BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(
            m, book, book))

    def test_contiguous_sequence_and_duplicate_sequence_are_applied(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1000",
            "sequence": 10, "hash": "a", "asks": [{"price": "0.45", "size": "5"}], "bids": [],
        }, mapping, books)
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "timestamp": "1001", "sequence": 11,
            "price_changes": [{"asset_id": "yes-token", "price": "0.44", "size": "2", "side": "SELL", "hash": "b"}],
        }, mapping, books), ["m1"])
        self.assertEqual(books["yes-token"].best_ask(), (0.44, 2.0))
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "timestamp": "1001", "sequence": 11,
            "price_changes": [{"asset_id": "yes-token", "price": "0.43", "size": "1", "side": "SELL", "hash": "b2"}],
        }, mapping, books), ["m1"])
        self.assertEqual(books["yes-token"].best_ask(), (0.43, 1.0))

    def test_unknown_schema_version_and_hash_chain_break_unsync_the_book(self):
        m = market()
        books = {}
        mapping = {m.yes_token_id: m, m.no_token_id: m}
        handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1000",
            "hash": "a", "asks": [{"price": "0.45", "size": "5"}], "bids": [],
        }, mapping, books)
        self.assertEqual(handle_market_event({
            "event_type": "book", "asset_id": "yes-token", "timestamp": "1001",
            "schema_version": "99.0", "hash": "b",
            "asks": [{"price": "0.10", "size": "99"}], "bids": [],
        }, mapping, books), [])
        self.assertFalse(books["yes-token"].synced)
        books["yes-token"].replace_snapshot({
            "timestamp": "1002", "hash": "c",
            "asks": [{"price": "0.45", "size": "5"}], "bids": [],
        })
        self.assertEqual(handle_market_event({
            "event_type": "price_change", "timestamp": "1003", "prev_hash": "not-c",
            "price_changes": [{"asset_id": "yes-token", "price": "0.40", "size": "1", "side": "SELL", "hash": "d"}],
        }, mapping, books), [])
        self.assertFalse(books["yes-token"].synced)
        self.assertIn("hash chain", books["yes-token"].gap_reason)


class ScannerTests(unittest.TestCase):
    def book(self, asks):
        b = OrderBook()
        b.asks = dict(asks)
        b.timestamp_ms = 1
        b.synced = True
        return b

    def test_requires_both_legs_and_includes_taker_fees(self):
        opportunity = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(
            market(), self.book([(0.40, 100)]), self.book([(0.40, 100)]))
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.shares, 100)
        self.assertGreater(opportunity.gross_profit, opportunity.net_profit)

    def test_thin_second_leg_limits_size(self):
        opportunity = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(
            market(), self.book([(0.40, 100)]), self.book([(0.40, 2)]))
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.shares, 2)

    def test_expensive_pair_is_rejected(self):
        opportunity = BinaryArbitrageScanner(min_net_profit_usd=0.01).scan(
            market(), self.book([(0.50, 100)]), self.book([(0.50, 100)]))
        self.assertIsNone(opportunity)

    def test_expensive_pair_sets_net_below_floor_reason(self):
        scanner = BinaryArbitrageScanner(min_net_profit_usd=0.05, min_return=0.0, safety_buffer_usd=0.02)
        opportunity = scanner.scan(market(), self.book([(0.49, 100)]), self.book([(0.49, 100)]))
        self.assertIsNone(opportunity)
        self.assertEqual(scanner.last_reject_reason, "net_below_floor")
        self.assertAlmostEqual(scanner.last_touch_sum, 0.98)
        self.assertIsNotNone(scanner.last_best_net)
        self.assertLess(scanner.last_best_net, 0.05)

    def test_missing_touch_sets_no_touch_reason(self):
        scanner = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0)
        empty = self.book([])
        opportunity = scanner.scan(market(), self.book([(0.40, 10)]), empty)
        self.assertIsNone(opportunity)
        self.assertEqual(scanner.last_reject_reason, "no_touch")

    def test_max_order_is_checked_against_depth_walked_cost(self):
        yes, no = self.book([(0.30, 100), (0.35, 100)]), self.book([(0.30, 100), (0.35, 100)])
        opportunity = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0, max_order_usd=100.0,
        ).scan(BinaryMarket("m1", "c1", "Test", "yes-token", "no-token", category="geopolitics"), yes, no)
        self.assertLessEqual(opportunity.capital_required, 100.0 + 1e-8)

    def test_execution_reservation_uses_highest_fee_across_allowed_levels(self):
        yes = self.book([(0.40, 5), (0.50, 5)])
        no = self.book([(0.40, 10)])
        opportunity = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
        ).scan(BinaryMarket("m1", "c1", "Test", "yes-token", "no-token", category="sports"), yes, no)
        self.assertIsNotNone(opportunity)
        self.assertGreater(opportunity.yes_execution_fee_cap, opportunity.yes_fee)
        self.assertAlmostEqual(
            opportunity.execution_capital_required,
            opportunity.yes_execution_amount + opportunity.no_execution_amount
            + opportunity.yes_execution_fee_cap + opportunity.no_execution_fee_cap,
            places=9,
        )

    def test_negative_risk_market_is_not_assumed_to_be_binary_arb(self):
        negative_risk = BinaryMarket("m2", "c2", "NR", "y", "n", category="politics", neg_risk=True)
        self.assertIsNone(BinaryArbitrageScanner(min_net_profit_usd=0.01).scan(
            negative_risk, self.book([(0.30, 10)]), self.book([(0.30, 10)])))

    def test_merge_gas_is_subtracted_from_scan_profit_and_pair_is_not_risk_free(self):
        cheap = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0, merge_gas_usd=0.0,
        ).scan(market(), self.book([(0.40, 10)]), self.book([(0.40, 10)]))
        costly = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0, merge_gas_usd=5.0,
        ).scan(market(), self.book([(0.40, 10)]), self.book([(0.40, 10)]))
        self.assertIsNotNone(cheap)
        self.assertFalse(cheap.is_risk_free)
        self.assertIn("not atomic", cheap.residual_risk)
        self.assertIsNone(costly)
        still_profit = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0, merge_gas_usd=0.05,
        ).scan(market(), self.book([(0.40, 10)]), self.book([(0.40, 10)]))
        self.assertAlmostEqual(still_profit.net_profit, cheap.net_profit - 0.05, places=6)

    def test_maker_scan_zeros_taker_fees_so_politics_mid_can_clear(self):
        yes, no = self.book([(0.49, 100)]), self.book([(0.50, 100)])
        taker = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0).scan(
            market(), yes, no, is_taker=True)
        maker = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0).scan(
            market(), yes, no, is_taker=False)
        self.assertIsNone(taker)
        self.assertIsNotNone(maker)
        self.assertEqual(maker.yes_fee, 0.0)
        self.assertEqual(maker.no_fee, 0.0)
        self.assertEqual(maker.yes_execution_fee_cap, 0.0)
        self.assertFalse(maker.is_taker)
        self.assertEqual(maker.order_style, "GTC")
        self.assertAlmostEqual(maker_limit_price(0.50, 0.01), 0.49, places=9)

    def test_unsynced_book_is_not_scanned(self):
        yes, no = self.book([(0.40, 10)]), self.book([(0.40, 10)])
        yes.synced = False
        self.assertIsNone(BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(
            market(), yes, no))


class LedgerTests(unittest.TestCase):
    def test_paper_pair_is_atomic_and_recovers_on_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.json")
            ledger = JsonLedger(path, initial_cash=1000)
            ledger.state["initial_cash"] = 1000
            opportunity = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(
                market(), OrderBook(), OrderBook())
            # Construct through real books to keep this test close to execution.
            yes, no = OrderBook(), OrderBook()
            yes.asks, no.asks = {0.40: 10}, {0.40: 10}
            yes.synced = no.synced = True
            yes.timestamp_ms = no.timestamp_ms = 1
            opportunity = BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(market(), yes, no)
            position = PaperArbitrageExecutor(ledger, max_total_exposure_fraction=1.0,
                                               max_market_exposure_fraction=1.0).execute(opportunity)
            self.assertAlmostEqual(ledger.state["cash"], 992.0 - opportunity.yes_fee - opportunity.no_fee, places=6)
            reloaded = JsonLedger(path, initial_cash=1)
            self.assertIn(position.position_id, reloaded.state["positions"])
            reloaded.settle(position.position_id, "Yes")
            # Ten YES + ten NO shares pay out ten dollars as a complete pair.
            self.assertAlmostEqual(
                reloaded.state["cash"],
                1000 + opportunity.gross_profit - opportunity.yes_fee - opportunity.no_fee,
                places=6,
            )


class FakeClient:
    def __init__(self, no_ok=True, rollback_ok=True, yes_fill="10"):
        self.no_ok = no_ok
        self.rollback_ok = rollback_ok
        self.yes_fill = yes_fill
        self.calls = []

    async def get_balance_allowance(self, **kwargs):
        return {"balance": "1000", "allowance": "1000"}

    async def list_open_orders(self, **kwargs):
        return []

    async def list_account_trades(self, **kwargs):
        return []

    async def cancel_order(self, **kwargs):
        self.calls.append({"cancel_order": kwargs})
        return {"ok": True, "order_id": kwargs.get("order_id")}

    async def place_market_order(self, **kwargs):
        self.calls.append(kwargs)
        side = kwargs["side"]
        if side == "BUY" and kwargs["token_id"] == "no-token":
            return {"ok": self.no_ok, "order_id": "no-1", "taking_amount": "10" if self.no_ok else "0"}
        if side == "SELL":
            return {"ok": self.rollback_ok, "order_id": "rollback-1",
                    "making_amount": "10" if self.rollback_ok else "0",
                    "taking_amount": "4" if self.rollback_ok else "0"}
        return {"ok": True, "order_id": "yes-1", "taking_amount": self.yes_fill}


class LiveExecutorTests(unittest.TestCase):
    def opportunity(self):
        yes, no = OrderBook(), OrderBook()
        yes.asks, no.asks = {0.40: 10}, {0.40: 10}
        yes.synced = no.synced = True
        yes.timestamp_ms = no.timestamp_ms = 1
        return BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0).scan(market(), yes, no)

    def test_fok_pair_uses_price_caps_and_returns_two_orders(self):
        client = FakeClient()
        result = asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))
        self.assertEqual((result.yes_order_id, result.no_order_id), ("yes-1", "no-1"))
        self.assertEqual([call["order_type"] for call in client.calls], ["FOK", "FOK"])
        self.assertEqual(client.calls[0]["max_price"], "0.400000")
        self.assertIn("max_spend", client.calls[0])

    def test_protected_buy_is_sized_to_target_shares_at_worst_price(self):
        yes, no = OrderBook(), OrderBook()
        yes.asks, no.asks = {0.40: 5, 0.45: 5}, {0.40: 5, 0.45: 5}
        yes.synced = no.synced = True
        yes.timestamp_ms = no.timestamp_ms = 1
        opportunity = BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
        ).scan(market(), yes, no)
        client = FakeClient()
        asyncio.run(OfficialFOKExecutor(client).execute(opportunity))
        self.assertEqual(client.calls[0]["max_price"], "0.450000")
        self.assertEqual(client.calls[0]["amount"], "4.500000")

    def test_second_leg_failure_attempts_fak_rollback(self):
        client = FakeClient(no_ok=False)
        with self.assertRaises(RuntimeError):
            asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))
        self.assertEqual([call["side"] for call in client.calls], ["BUY", "BUY", "SELL"])
        self.assertEqual(client.calls[-1]["order_type"], "FAK")

    def test_unknown_first_order_outcome_is_unhedged(self):
        class UnknownYesClient(FakeClient):
            async def place_market_order(self, **kwargs):
                if kwargs["side"] == "BUY" and kwargs["token_id"] == "yes-token":
                    raise TimeoutError("request timed out after submission")
                return await super().place_market_order(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = UnknownYesClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(executor.execute(self.opportunity()))
            record = next(iter(journal.state["pairs"].values()))
            self.assertEqual(record["status"], "UNHEDGED")
            self.assertIn("outcome is unknown", record["error"])

    def test_unknown_second_order_outcome_never_becomes_rolled_back(self):
        class UnknownNoClient(FakeClient):
            async def place_market_order(self, **kwargs):
                if kwargs["side"] == "BUY" and kwargs["token_id"] == "no-token":
                    self.calls.append(kwargs)
                    raise TimeoutError("request timed out after submission")
                return await super().place_market_order(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = UnknownNoClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(executor.execute(self.opportunity()))
            record = next(iter(journal.state["pairs"].values()))
            self.assertEqual(record["status"], "UNHEDGED")
            self.assertEqual(record["rollback_status"], "CONFIRMED")
            self.assertEqual([call["side"] for call in client.calls], ["BUY", "BUY", "SELL"])

    def test_confirmed_rollback_is_logged_and_counts_toward_daily_loss(self):
        class LossClient(FakeClient):
            async def place_market_order(self, **kwargs):
                response = await super().place_market_order(**kwargs)
                if kwargs["side"] == "SELL":
                    response["taking_amount"] = "3"
                return response

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(LossClient(no_ok=False), journal=journal)
            with self.assertRaises(RuntimeError):
                asyncio.run(executor.execute(self.opportunity()))
            record = next(iter(journal.state["pairs"].values()))
            self.assertEqual(record["status"], "ROLLED_BACK")
            self.assertAlmostEqual(record["realized_pnl"], -1.0)
            self.assertEqual(record["rollback_details"]["yes"]["proceeds_usd"], 3.0)
            risk = LiveRiskController(
                journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="", max_daily_loss_usd=1.0,
            )
            with self.assertRaisesRegex(RiskHaltError, "daily loss"):
                risk.check_startup()

    def test_rollback_failure_is_escalated_as_unhedged(self):
        client = FakeClient(no_ok=False, rollback_ok=False)
        with self.assertRaises(UnhedgedPairError):
            asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))

    def test_partial_first_leg_is_rolled_back_and_never_marked_rejected(self):
        client = FakeClient(yes_fill="3")
        with self.assertRaises(RuntimeError):
            asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))
        self.assertEqual([call["side"] for call in client.calls], ["BUY", "SELL"])

    def test_second_leg_partial_fill_is_rolled_back_before_first_leg(self):
        class PartialNoClient(FakeClient):
            async def place_market_order(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs["token_id"] == "no-token":
                    return {"ok": True, "order_id": "no-partial", "taking_amount": "3"}
                if kwargs["side"] == "SELL":
                    return {"ok": True, "order_id": "rollback", "making_amount": "10"}
                return {"ok": True, "order_id": "yes-1", "taking_amount": "10"}

        client = PartialNoClient()
        with self.assertRaises(RuntimeError):
            asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))
        self.assertEqual([call["side"] for call in client.calls], ["BUY", "BUY", "SELL", "SELL"])

    def test_price_improvement_overfill_is_rolled_back(self):
        class OverfillClient(FakeClient):
            async def place_market_order(self, **kwargs):
                if kwargs["side"] == "SELL":
                    self.calls.append(kwargs)
                    return {"ok": True, "order_id": "rollback", "making_amount": kwargs["shares"]}
                return await super().place_market_order(**kwargs)

        client = OverfillClient(yes_fill="11")
        with self.assertRaises(RuntimeError):
            asyncio.run(OfficialFOKExecutor(client).execute(self.opportunity()))
        self.assertEqual([call["side"] for call in client.calls], ["BUY", "SELL"])

    def test_live_client_requires_a_real_private_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PRIVATE_KEY"):
                asyncio.run(OfficialFOKExecutor.create_from_env())

    def test_journal_tracks_pair_and_duplicate_trade_once(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            opportunity = self.opportunity()
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(opportunity))
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["status"], "PENDING")
            self.assertEqual(record["condition_id"], "c1")
            journal.add_fill(result.pair_id, "yes", 10, "trade-1")
            journal.add_fill(result.pair_id, "yes", 10, "trade-1")
            self.assertEqual(record["yes_trade_ids"], ["trade-1"])
            self.assertAlmostEqual(record["yes_matched_shares"], 10.0)
            self.assertEqual(LiveOrderJournal(journal.path).state["pairs"][result.pair_id]["status"], "PENDING")

    def test_journal_summary_exposes_exposure_and_unhedged_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            journal.set_status(pair_id, "UNHEDGED", "manual review")
            summary = journal.summary()
            self.assertEqual(summary["by_status"], {"UNHEDGED": 1})
            self.assertEqual(summary["unhedged_pairs"], [pair_id])
            self.assertGreater(summary["open_exposure"], 0.0)

    def test_journal_marks_actual_pnl_when_fill_economics_are_available(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            opportunity = self.opportunity()
            pair_id = journal.create_pair(opportunity)
            journal.add_fill(pair_id, "yes", 10, "ty", price=0.39, fee_usd=0.01, tx_hash="0xy")
            journal.add_fill(pair_id, "no", 10, "tn", price=0.38, fee_usd=0.02, tx_hash="0xn")
            journal.set_status(pair_id, "HEDGED")
            journal.mark_resolved("m1", "c1", "Yes")
            record = journal.state["pairs"][pair_id]
            self.assertEqual(record["status"], "RESOLVED_PENDING_REDEMPTION")
            journal.mark_redeemed(pair_id, "0xredeem")
            self.assertEqual(record["pnl_quality"], "ACTUAL")
            self.assertAlmostEqual(record["realized_pnl"], 2.27, places=6)
            self.assertEqual(record["yes_transaction_hashes"], ["0xy"])

    def test_auto_merge_settles_only_after_both_legs_are_hedged(self):
        class MergeHandle:
            async def wait(self):
                return {"transaction_hash": "0xmerge"}

        class MergeClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.merge_calls = []

            async def merge_positions(self, **kwargs):
                self.merge_calls.append(kwargs)
                return MergeHandle()

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = MergeClient()
            executor = OfficialFOKExecutor(client, journal=journal, auto_merge=True)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            settled = asyncio.run(executor.settle_hedged_pairs())
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["status"], "SETTLED")
            self.assertEqual(settled[0]["settlement_tx_hash"], "0xmerge")
            self.assertEqual(client.merge_calls[0]["condition_id"], "c1")
            self.assertEqual(client.merge_calls[0]["amount"], 10_000_000)

    def test_market_resolution_requires_confirmed_redemption(self):
        class RedeemHandle:
            async def wait(self):
                return {"transaction_hash": "0xredeem"}

        class RedeemClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.redeem_calls = []

            async def redeem_positions(self, **kwargs):
                self.redeem_calls.append(kwargs)
                return RedeemHandle()

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = RedeemClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            self.assertEqual(journal.mark_resolved("m1", "c1", "Yes"), 1)
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["status"], "RESOLVED_PENDING_REDEMPTION")
            self.assertNotIn("settled_at", record)
            self.assertAlmostEqual(journal.open_exposure(), record["capital_reserved"])
            settled = asyncio.run(executor.settle_hedged_pairs())
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["status"], "SETTLED")
            self.assertEqual(settled[0]["settlement_type"], "MARKET_REDEMPTION")
            self.assertEqual(client.redeem_calls, [{"condition_id": "c1"}])

    def test_submitted_redemption_is_not_repeated_after_wait_timeout(self):
        class TimeoutHandle:
            async def wait(self):
                raise TimeoutError("chain receipt not observed")

        class TimeoutClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.redeem_calls = 0

            async def redeem_positions(self, **kwargs):
                self.redeem_calls += 1
                return TimeoutHandle()

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = TimeoutClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            journal.mark_resolved("m1", "c1", "Yes")
            self.assertEqual(asyncio.run(executor.settle_hedged_pairs()), [])
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["status"], "RESOLVED_PENDING_REDEMPTION")
            self.assertEqual(record["settlement_type"], "REDEEM_SUBMITTED")
            self.assertEqual(client.redeem_calls, 1)
            self.assertEqual(asyncio.run(executor.settle_hedged_pairs()), [])
            self.assertEqual(client.redeem_calls, 1)

    def test_confirmed_submitted_redemption_is_resumed_without_resubmission(self):
        class TimeoutHandle:
            transaction_hash = "0xredeem"

            async def wait(self):
                raise TimeoutError("chain receipt not observed")

        class ResumeClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.redeem_calls = 0

            async def redeem_positions(self, **kwargs):
                self.redeem_calls += 1
                return TimeoutHandle()

            async def get_transaction_receipt(self, **kwargs):
                return {"status": "0x1"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = ResumeClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            journal.mark_resolved("m1", "c1", "Yes")
            self.assertEqual(asyncio.run(executor.settle_hedged_pairs()), [])
            settled = asyncio.run(executor.settle_hedged_pairs())
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["status"], "SETTLED")
            self.assertEqual(client.redeem_calls, 1)

    def test_confirmed_gasless_redemption_is_resumed_from_transaction_id(self):
        class TimeoutHandle:
            transaction_id = "relayer-1"
            transaction_hash = None

            async def wait(self):
                raise TimeoutError("relayer receipt not observed")

        class Relayer:
            async def get_json(self, path):
                self.path = path
                return {
                    "state": "STATE_CONFIRMED",
                    "transaction_id": "relayer-1",
                    "transaction_hash": "0xgasless",
                }

        class GaslessClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.redeem_calls = 0
                self._ctx = type("Context", (), {"relayer": Relayer()})()

            async def redeem_positions(self, **kwargs):
                self.redeem_calls += 1
                return TimeoutHandle()

            async def get_relayer_transaction(self, **kwargs):
                return type("Transaction", (), {
                    "state": "STATE_CONFIRMED",
                    "transaction_hash": "0xgasless",
                })()

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            client = GaslessClient()
            executor = OfficialFOKExecutor(client, journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            journal.mark_resolved("m1", "c1", "Yes")
            self.assertEqual(asyncio.run(executor.settle_hedged_pairs()), [])
            settled = asyncio.run(executor.settle_hedged_pairs())
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["settlement_tx_hash"], "0xgasless")
            self.assertEqual(client.redeem_calls, 1)

    def test_submitted_merge_is_not_misclassified_when_tokens_are_burned(self):
        class TimeoutHandle:
            async def wait(self):
                raise TimeoutError("chain receipt not observed")

        class MergeClient(FakeClient):
            async def merge_positions(self, **kwargs):
                return TimeoutHandle()

            async def get_balance_allowance(self, **kwargs):
                if kwargs.get("asset_type") == "CONDITIONAL":
                    return {"balance": "0", "allowances": {"conditional-spender": "0"}}
                return await super().get_balance_allowance(**kwargs)

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(MergeClient(), journal=journal, auto_merge=True)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            self.assertEqual(asyncio.run(executor.settle_hedged_pairs()), [])
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["settlement_type"], "MERGE_SUBMITTED")
            reconciled = asyncio.run(executor.reconcile(stale_after_seconds=9999))
            self.assertEqual(reconciled[0]["status"], "HEDGED")

    def test_user_trade_event_updates_leg_and_settlement(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            executor.handle_user_event({
                "event_type": "trade", "order_id": result.no_order_id,
                "token_id": "no-token", "trade_id": "trade-no", "size": "10",
            })
            executor.handle_user_event({
                "event_type": "order", "order_id": result.yes_order_id,
                "token_id": "yes-token", "size_matched": "10",
            })
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["no_trade_ids"], ["trade-no"])
            self.assertAlmostEqual(record["no_matched_shares"], 10.0)
            self.assertEqual(journal.mark_resolved("m1", "c1", "Yes"), 1)
            self.assertEqual(record["status"], "RESOLVED_PENDING_REDEMPTION")
            journal.mark_redeemed(result.pair_id, "0xredeem")
            self.assertEqual(record["status"], "SETTLED")

    def test_user_event_without_original_order_id_is_not_attributed_by_token(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            executor.handle_user_event({
                "event_type": "trade", "token_id": "yes-token", "id": "unrelated-trade",
                "size": "10", "price": "0.4", "side": "SELL",
            })
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["yes_matched_shares"], 0.0)

    def test_user_event_with_wrong_side_halts_instead_of_recording_a_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            with self.assertRaises(UnhedgedPairError):
                executor.handle_user_event({
                    "event_type": "trade", "order_id": result.yes_order_id,
                    "token_id": "yes-token", "trade_id": "bad-side",
                    "side": "SELL", "size": "10",
                })
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(record["yes_matched_shares"], 0.0)

    def test_reconcile_detects_one_leg_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            client = executor.client
            original = client.place_market_order

            async def get_order(**kwargs):
                return {"status": "FILLED", "size_matched": "10" if kwargs["order_id"] == result.yes_order_id else "0"}

            client.get_order = get_order
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(executor.reconcile(stale_after_seconds=9999))

    def test_reconcile_halts_orphaned_pair_intent_without_order_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            with self.assertRaisesRegex(UnhedgedPairError, "no submitted order IDs"):
                asyncio.run(executor.reconcile(stale_after_seconds=9999))
            record = journal.state["pairs"][pair_id]
            self.assertEqual(record["status"], "UNHEDGED")

    def test_reconcile_detects_missing_conditional_token_balance(self):
        class MissingTokenClient(FakeClient):
            async def get_balance_allowance(self, **kwargs):
                if kwargs.get("asset_type") == "CONDITIONAL":
                    return {"balance": "0", "allowances": {"conditional-spender": "0"}}
                return await super().get_balance_allowance(**kwargs)

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(MissingTokenClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            with self.assertRaisesRegex(UnhedgedPairError, "conditional balance"):
                asyncio.run(executor.reconcile(stale_after_seconds=9999))

    def test_reconcile_recovers_fills_missing_from_user_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(FakeClient(), journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))

            async def get_order(**kwargs):
                return {"status": "FILLED", "size_matched": "10"}

            async def list_account_trades(**kwargs):
                return [{"id": "recovered-no", "taker_order_id": result.no_order_id,
                         "asset_id": "no-token", "size": "10"}]

            executor.client.get_order = get_order
            executor.client.list_account_trades = list_account_trades
            reconciled = asyncio.run(executor.reconcile(stale_after_seconds=9999))
            self.assertEqual(reconciled[0]["status"], "HEDGED")
            self.assertEqual(reconciled[0]["no_trade_ids"], ["recovered-no"])

    def test_reconcile_uses_persisted_trade_watermark_with_overlap(self):
        class WatermarkClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.trade_queries = []

            async def list_account_trades(self, **kwargs):
                self.trade_queries.append(kwargs)
                return []

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            result = asyncio.run(OfficialFOKExecutor(FakeClient(), journal=journal).execute(self.opportunity()))
            journal.state["trade_watermarks"]["c1"] = 1_700_000_100
            journal.save()
            client = WatermarkClient()
            executor = OfficialFOKExecutor(client, journal=LiveOrderJournal(journal.path))
            asyncio.run(executor.reconcile(stale_after_seconds=9999))
            self.assertEqual(client.trade_queries, [{"market": "c1", "after": "1700000040"}])

    def test_regular_reconcile_can_skip_open_order_orphan_scan(self):
        class KnownOrderClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.open_order_queries = 0

            async def list_open_orders(self, **kwargs):
                self.open_order_queries += 1
                return []

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            result = asyncio.run(OfficialFOKExecutor(FakeClient(), journal=journal).execute(self.opportunity()))
            client = KnownOrderClient()
            executor = OfficialFOKExecutor(client, journal=LiveOrderJournal(journal.path))
            asyncio.run(executor.reconcile(stale_after_seconds=9999, recover_orphans=False))
            self.assertEqual(client.open_order_queries, 0)

    def test_reconcile_recovers_order_ids_after_crash_before_journal_write(self):
        class OrphanClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.created = time.time()

            async def list_account_trades(self, **kwargs):
                return [
                    {"id": "ty", "taker_order_id": "yes-orphan", "market": "c1",
                     "asset_id": "yes-token", "side": "BUY", "size": "10", "price": "0.4",
                     "fee_rate_bps": "0", "match_time": self.created},
                    {"id": "tn", "taker_order_id": "no-orphan", "market": "c1",
                     "asset_id": "no-token", "side": "BUY", "size": "10", "price": "0.4",
                     "fee_rate_bps": "0", "match_time": self.created},
                ]

            async def get_order(self, **kwargs):
                order_id = kwargs["order_id"]
                token = "yes-token" if order_id == "yes-orphan" else "no-token"
                return {"market": "c1", "asset_id": token, "side": "BUY",
                        "status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            executor = OfficialFOKExecutor(OrphanClient(), journal=journal)
            reconciled = asyncio.run(executor.reconcile(stale_after_seconds=9999))
            self.assertEqual(reconciled[0]["pair_id"], pair_id)
            self.assertEqual(reconciled[0]["status"], "HEDGED")
            self.assertEqual(reconciled[0]["yes_order_id"], "yes-orphan")
            self.assertEqual(reconciled[0]["no_order_id"], "no-orphan")

    def test_425_retries_and_503_is_not_retried(self):
        class StatusClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.restart_calls = 0

            async def ping(self):
                self.restart_calls += 1
                if self.restart_calls == 1:
                    return {"status_code": 425}
                return {"ok": True}

            async def unavailable(self):
                return {"status_code": 503}

        client = StatusClient()
        executor = OfficialFOKExecutor(client, max_retries=1)
        self.assertEqual(asyncio.run(executor._call("ping")), {"ok": True})
        self.assertEqual(client.restart_calls, 2)
        with self.assertRaises(CancelOnlyError):
            asyncio.run(executor._call("unavailable"))

    def test_sdk_exception_status_field_is_retried(self):
        class ExceptionStatusClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.calls_count = 0

            async def ping(self):
                self.calls_count += 1
                if self.calls_count == 1:
                    error = RuntimeError("restart")
                    error.status = 425
                    raise error
                return {"ok": True}

        client = ExceptionStatusClient()
        self.assertEqual(asyncio.run(OfficialFOKExecutor(client, max_retries=1)._call("ping")), {"ok": True})
        self.assertEqual(client.calls_count, 2)

    def test_live_risk_controller_persists_exposure_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            risk = LiveRiskController(
                journal, equity_usd=100.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=os.path.join(directory, "kill"), max_total_exposure_fraction=0.10,
                max_market_exposure_fraction=0.10,
            )
            opportunity = self.opportunity()
            risk.check(opportunity)
            journal.create_pair(opportunity)
            with self.assertRaises(RiskHaltError):
                risk.check(opportunity)

    def test_live_risk_controller_rejects_non_finite_journal_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            journal.state["pairs"][pair_id]["capital_reserved"] = float("nan")
            journal.save()
            risk = LiveRiskController(
                journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="",
            )
            with self.assertRaisesRegex(RiskHaltError, "integrity"):
                risk.check_startup()

    def test_live_risk_controller_records_realized_loss_before_next_trade(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            risk = LiveRiskController(
                journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="", max_daily_loss_usd=5.0,
            )
            risk.record_realized_pnl(-5.0)
            with self.assertRaisesRegex(RiskHaltError, "daily loss"):
                risk.check_startup()

    def test_live_risk_controller_rejects_non_finite_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            risk = LiveRiskController(
                journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="", max_daily_loss_usd=float("nan"),
            )
            with self.assertRaisesRegex(RiskHaltError, "configuration"):
                risk.check_startup()

    def test_live_risk_controller_halts_on_kill_switch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            kill_path = os.path.join(directory, "kill")
            with open(kill_path, "w", encoding="utf-8") as handle:
                handle.write("operator halt\n")
            risk = LiveRiskController(
                journal, equity_usd=100.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=kill_path,
            )
            with self.assertRaises(RiskHaltError):
                risk.check_startup()
            restored = LiveRiskController(
                journal, equity_usd=100.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="",
            )
            with self.assertRaises(RiskHaltError):
                restored.check_startup()

    def test_live_risk_controller_halts_on_persisted_unhedged_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            journal.set_status(pair_id, "UNHEDGED")
            risk = LiveRiskController(
                journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="",
            )
            with self.assertRaisesRegex(RiskHaltError, "unhedged"):
                risk.check_startup()

    def test_preflight_uses_collateral_base_units_and_rejects_ambiguous_spenders(self):
        class AllowanceClient(FakeClient):
            async def get_balance_allowance(self, **kwargs):
                return {"balance": "2000000", "allowances": {"spender-a": "3000000"}}

        snapshot = asyncio.run(OfficialFOKExecutor(AllowanceClient()).preflight(required_usd=2.0))
        self.assertEqual(snapshot, {"balance": 2.0, "allowance": 3.0})

        class AmbiguousClient(AllowanceClient):
            async def get_balance_allowance(self, **kwargs):
                return {"balance": "2000000", "allowances": {"a": "3000000", "b": "3000000"}}

        with self.assertRaisesRegex(RuntimeError, "multiple collateral spenders"):
            asyncio.run(OfficialFOKExecutor(AmbiguousClient()).preflight(required_usd=1.0))

    def test_preflight_rejects_non_finite_balance(self):
        class InvalidBalanceClient(FakeClient):
            async def get_balance_allowance(self, **kwargs):
                return {"balance": "nan", "allowance": "1000"}

        with self.assertRaisesRegex(RuntimeError, "balance/allowance response is invalid"):
            asyncio.run(OfficialFOKExecutor(InvalidBalanceClient()).preflight(required_usd=0.0))

    def test_account_scan_halts_on_external_open_orders_and_positions(self):
        class ExternalClient(FakeClient):
            async def list_open_orders(self, **kwargs):
                return [{"id": "foreign-order", "token_id": "other-token", "side": "BUY"}]

            async def list_positions(self):
                return [{"token_id": "stray-token", "size": "4.0"}]

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            asyncio.run(OfficialFOKExecutor(FakeClient(), journal=journal).execute(self.opportunity()))
            executor = OfficialFOKExecutor(ExternalClient(), journal=LiveOrderJournal(journal.path))
            with self.assertRaisesRegex(UnhedgedPairError, "open orders outside"):
                asyncio.run(executor.reconcile(stale_after_seconds=9999, scan_account=True))

        class PositionOnlyClient(FakeClient):
            async def list_positions(self):
                return [{"asset_id": "stray-token", "shares": "1.5"}]

            async def get_order(self, **kwargs):
                return {"status": "FILLED", "size_matched": "10"}

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            asyncio.run(OfficialFOKExecutor(FakeClient(), journal=journal).execute(self.opportunity()))
            executor = OfficialFOKExecutor(PositionOnlyClient(), journal=LiveOrderJournal(journal.path))
            with self.assertRaisesRegex(UnhedgedPairError, "inventory outside"):
                asyncio.run(executor.reconcile(stale_after_seconds=9999, scan_account=True))

    def test_kill_switch_cancel_all_leaves_matched_inventory_for_manual_review(self):
        class FlattenClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.open = [{"id": "resting-1", "token_id": "yes-token"}]
                self.cancelled = []

            async def list_open_orders(self, **kwargs):
                return list(self.open)

            async def cancel_order(self, **kwargs):
                order_id = kwargs["order_id"]
                self.cancelled.append(order_id)
                self.open = [order for order in self.open if order["id"] != order_id]
                return {"ok": True, "order_id": order_id}

            async def get_balance_allowance(self, **kwargs):
                if kwargs.get("asset_type") == "CONDITIONAL":
                    return {"balance": "10000000", "allowances": {"c": "0"}}
                return await super().get_balance_allowance(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            kill_path = os.path.join(directory, "kill")
            with open(kill_path, "w", encoding="utf-8") as handle:
                handle.write("stop\n")
            result = asyncio.run(OfficialFOKExecutor(FakeClient(), journal=journal).execute(self.opportunity()))
            journal.set_status(result.pair_id, "HEDGED")
            client = FlattenClient()
            executor = OfficialFOKExecutor(client, journal=LiveOrderJournal(journal.path))
            risk = LiveRiskController(
                executor.journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=kill_path,
            )
            with self.assertRaises(RiskHaltError):
                risk.check_startup()
            flatten = asyncio.run(executor.apply_halt_actions(risk))
            self.assertEqual(client.cancelled, ["resting-1"])
            self.assertEqual(flatten["cancelled_order_ids"], ["resting-1"])
            self.assertTrue(risk.state["flatten_completed"])
            self.assertEqual(len(flatten["inventory"]["pairs"]), 1)
            self.assertIn("not auto-sold", flatten["note"])
            asyncio.run(executor.apply_halt_actions(risk))
            self.assertEqual(client.cancelled, ["resting-1"])

    def test_kill_switch_faks_directional_inventory_but_not_hedged_pairs(self):
        from polywang.arbitrage_core import DirectionalIntent, LiveDirectionalJournal, PaperDirectionalExecutor

        class FlattenClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.open = []

            async def list_open_orders(self, **kwargs):
                return list(self.open)

            async def cancel_order(self, **kwargs):
                return {"ok": True, "order_id": kwargs["order_id"]}

            async def get_balance_allowance(self, **kwargs):
                if kwargs.get("asset_type") == "CONDITIONAL":
                    return {"balance": "10000000", "allowances": {"c": "0"}}
                return await super().get_balance_allowance(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            pairs = LiveOrderJournal(os.path.join(directory, "live.json"))
            directional = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
            asyncio.run(OfficialFOKExecutor(FakeClient(), journal=pairs).execute(self.opportunity()))
            next(iter(pairs.state["pairs"].values()))["status"] = "HEDGED"
            pairs.save()
            PaperDirectionalExecutor(directional).execute(DirectionalIntent(
                token_id="yes-token", side="BUY", shares=5, limit_price=0.40, market_id="m1",
            ))
            self.assertAlmostEqual(directional.inventory_by_token()["yes-token"], 5.0)
            client = FlattenClient()
            executor = OfficialFOKExecutor(
                client, journal=LiveOrderJournal(pairs.path), directional_journal=directional,
            )
            risk = LiveRiskController(
                executor.journal, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="", extra_journals=[directional],
            )
            risk.halt("manual kill")
            flatten = asyncio.run(executor.apply_halt_actions(risk))
            sells = [call for call in client.calls if call.get("side") == "SELL"]
            self.assertEqual(len(sells), 1)
            self.assertEqual(sells[0]["order_type"], "FAK")
            self.assertEqual(sells[0]["token_id"], "yes-token")
            self.assertNotIn("yes-token", directional.inventory_by_token())
            self.assertEqual(len(flatten["inventory"]["pairs"]), 1)
            self.assertIn("not auto-sold", flatten["note"])
            self.assertEqual(len(flatten["directional_flatten"]["flattened"]), 1)


class FakeGTCClient(FakeClient):
    def __init__(self, yes_ok=True, no_ok=True):
        super().__init__(no_ok=no_ok)
        self.yes_ok = yes_ok
        self.limit_calls = []
        self.orders = {}

    async def place_limit_order(self, **kwargs):
        self.limit_calls.append(kwargs)
        token = kwargs["token_id"]
        ok = self.yes_ok if token == "yes-token" else self.no_ok
        order_id = f"gtc-{token}" if ok else ""
        response = {
            "ok": ok,
            "order_id": order_id,
            "taking_amount": "0",
            "message": "" if ok else "would take",
        }
        if ok:
            self.orders[order_id] = {
                "ok": True,
                "order_id": order_id,
                "token_id": token,
                "side": "BUY",
                "status": "OPEN",
                "taking_amount": "0",
            }
        return response

    async def get_order(self, **kwargs):
        return self.orders[kwargs["order_id"]]

    async def cancel_order(self, **kwargs):
        self.calls.append({"cancel_order": kwargs})
        order = self.orders.get(kwargs["order_id"])
        if order:
            order["status"] = "CANCELLED"
        return {"ok": True, "order_id": kwargs.get("order_id")}


class MakerGTCTests(unittest.TestCase):
    def opportunity(self):
        yes, no = OrderBook(), OrderBook()
        yes.asks, no.asks = {0.49: 10}, {0.50: 10}
        yes.synced = no.synced = True
        yes.timestamp_ms = no.timestamp_ms = 1
        return BinaryArbitrageScanner(
            min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
        ).scan(market(), yes, no, is_taker=False)

    def test_maker_flag_defaults_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_MAKER_GTC", None)
            self.assertFalse(maker_gtc_enabled())

    def test_gtc_places_post_only_limits_and_stays_resting(self):
        opportunity = self.opportunity()
        self.assertIsNotNone(opportunity)
        client = FakeGTCClient()
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            result = asyncio.run(OfficialFOKExecutor(client, journal=journal).execute(opportunity))
            record = journal.state["pairs"][result.pair_id]
            self.assertEqual(result.status, "RESTING")
            self.assertEqual(record["status"], "RESTING")
            self.assertEqual(record["order_style"], "GTC")
            self.assertEqual([call["post_only"] for call in client.limit_calls], [True, True])
            self.assertEqual(client.limit_calls[0]["price"], "0.480000")
            self.assertEqual(client.limit_calls[1]["price"], "0.490000")
            self.assertFalse(any(call.get("order_type") == "FOK" for call in client.calls))

    def test_gtc_post_only_reject_does_not_fall_back_to_fok(self):
        client = FakeGTCClient(yes_ok=False)
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            with self.assertRaises(RuntimeError) as raised:
                asyncio.run(OfficialFOKExecutor(client, journal=journal).execute(self.opportunity()))
            self.assertIn("post-only", str(raised.exception))
            record = list(journal.state["pairs"].values())[0]
            self.assertEqual(record["status"], "REJECTED")
        self.assertEqual(len(client.limit_calls), 1)
        self.assertFalse(any(call.get("order_type") == "FOK" for call in client.calls))

    def test_gtc_timeout_cancels_and_unwinds_a_one_sided_fill(self):
        opportunity = self.opportunity()
        client = FakeGTCClient()
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(client, journal=journal, rest_seconds=0.0)
            result = asyncio.run(executor.execute(opportunity))
            client.orders["gtc-yes-token"]["taking_amount"] = "10"
            client.orders["gtc-yes-token"]["status"] = "FILLED"
            journal.state["pairs"][result.pair_id]["created_at"] = time.time() - 60
            journal.save()
            asyncio.run(executor.reconcile(stale_after_seconds=0.0, recover_orphans=False))
            record = journal.state["pairs"][result.pair_id]
            sells = [call for call in client.calls if call.get("side") == "SELL"]
            self.assertTrue(sells)
            self.assertEqual(sells[0]["order_type"], "FAK")
            self.assertEqual(record["status"], "ROLLED_BACK")

    def test_startup_allows_resting_gtc_but_halts_on_unhedged(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            pair_id = journal.create_pair(self.opportunity())
            journal.update(pair_id, order_style="GTC", status="RESTING",
                           yes_order_id="gtc-yes", no_order_id="gtc-no")
            risk = LiveRiskController(
                journal, equity_usd=1000.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path=os.path.join(directory, "missing-kill"),
            )
            risk.check_startup()
            journal.set_status(pair_id, "UNHEDGED", "one-sided fill")
            with self.assertRaises(RiskHaltError):
                risk.check_startup()

    def test_halt_unwinds_unbalanced_resting_gtc(self):
        client = FakeGTCClient()
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveOrderJournal(os.path.join(directory, "live.json"))
            executor = OfficialFOKExecutor(client, journal=journal)
            result = asyncio.run(executor.execute(self.opportunity()))
            journal.set_matched(result.pair_id, "yes", 10)
            journal.set_status(result.pair_id, "RESTING")
            risk = LiveRiskController(
                journal, equity_usd=1000.0, state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="",
            )
            risk.halt("kill")
            asyncio.run(executor.apply_halt_actions(risk))
            sells = [call for call in client.calls if call.get("side") == "SELL"]
            self.assertEqual(len(sells), 1)
            self.assertEqual(sells[0]["token_id"], "yes-token")
            self.assertEqual(journal.state["pairs"][result.pair_id]["status"], "ROLLED_BACK")


if __name__ == "__main__":
    unittest.main()
