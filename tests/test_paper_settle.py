#!/usr/bin/env python3
"""QUANT-20260914-01: paper settlement reconciler. No network."""

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from polywang.arbitrage_bot import PaperMarketRunner
from polywang.arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    JsonLedger,
    OrderBook,
    PaperArbitrageExecutor,
)
from polywang.monitor import compute_health_payload
from polywang.negrisk import (
    LiveNegRiskJournal,
    NegRiskBookScanner,
    NegRiskMarket,
    PaperNegRiskExecutor,
)
from polywang.paper_settle import (
    CASH_CLOSE_TOLERANCE,
    DEFAULT_RECONCILE_INTERVAL_SEC,
    DEFAULT_STUCK_GRACE_HOURS,
    PaperSettlementReconciler,
    extract_winning_outcome,
    parse_gamma_resolution,
    paper_settle_grace_hours,
    paper_settle_interval_sec,
    run_paper_settle_loop,
)


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nway_payload(**overrides):
    row = {
        "id": "nr-sports",
        "conditionId": "c-nr-sports",
        "question": "Blue Jays/Athletics",
        "clobTokenIds": '["tok-a", "tok-b", "tok-c"]',
        "outcomes": '["Blue Jays", "Athletics", "Other"]',
        "outcomePrices": '["0.20", "0.20", "0.20"]',
        "category": "sports",
        "active": True,
        "closed": False,
    }
    row.update(overrides)
    return row


def synced_book(asks, timestamp_ms=1, digest=""):
    book = OrderBook()
    book.asks = dict(asks)
    book.synced = True
    book.timestamp_ms = timestamp_ms
    book.hash = digest or str(asks)
    return book


def binary_market(**overrides):
    payload = {
        "id": "ceasefire",
        "conditionId": "c-ceasefire",
        "question": "US x Iran Effective Ceasefire by September 4?",
        "clobTokenIds": '["yes-token", "no-token"]',
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.40", "0.60"]',
        "category": "geopolitics",
        "active": True,
        "closed": False,
        "endDate": "2026-09-04T00:00:00Z",
    }
    payload.update(overrides)
    market = BinaryMarket.from_gamma(payload)
    assert market is not None
    return market


def open_binary_pair(runner, market):
    yes, no = OrderBook(), OrderBook()
    yes.asks, no.asks = {0.40: 10}, {0.40: 10}
    yes.synced = no.synced = True
    yes.timestamp_ms = no.timestamp_ms = 1
    opportunity = BinaryArbitrageScanner(
        min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
    ).scan(market, yes, no)
    assert opportunity is not None
    return PaperArbitrageExecutor(
        runner.ledger, max_total_exposure_fraction=1.0, max_market_exposure_fraction=1.0,
    ).execute(opportunity)


def open_nr_basket(runner, market):
    books = {token: synced_book({0.20: 10}) for token in market.yes_token_ids}
    opportunity = NegRiskBookScanner(
        min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
    ).scan(market, books)
    assert opportunity is not None
    return runner.negrisk_executor.execute(opportunity)


class FakeGamma:
    def __init__(self, rows):
        self.rows = dict(rows)
        self.calls = []

    def __call__(self, identifier):
        self.calls.append(str(identifier))
        return self.rows.get(str(identifier))


class GammaParseTests(unittest.TestCase):
    def test_binary_closed_yes_is_resolved_winner(self):
        snapshot = parse_gamma_resolution({
            "id": "ceasefire",
            "conditionId": "c-ceasefire",
            "question": "US x Iran Effective Ceasefire by September 4?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
            "closed": True,
            "active": False,
            "endDate": "2026-09-04T00:00:00Z",
            "umaResolutionStatus": "resolved",
        })
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.resolved)
        self.assertEqual(snapshot.winning_outcome, "Yes")
        self.assertTrue(snapshot.uma_known_or_finalized)
        self.assertEqual(extract_winning_outcome({
            "outcomes": ["Yes", "No"], "outcomePrices": ["0", "1"],
        }), "No")

    def test_uma_known_and_finalized_spellings(self):
        for status in ("known", "finalized", "resolved", "settled", "proposed"):
            snapshot = parse_gamma_resolution({
                "id": "m", "conditionId": "c",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.50", "0.50"]',
                "closed": False,
                "umaResolutionStatus": status,
                "endDate": "2026-09-04T00:00:00Z",
            })
            self.assertTrue(snapshot.uma_known_or_finalized, status)
            self.assertFalse(snapshot.resolved)

    def test_env_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPER_SETTLE_RECONCILE_INTERVAL_SEC", None)
            os.environ.pop("PAPER_SETTLE_STUCK_GRACE_HOURS", None)
            self.assertEqual(paper_settle_interval_sec(), DEFAULT_RECONCILE_INTERVAL_SEC)
            self.assertEqual(paper_settle_grace_hours(), DEFAULT_STUCK_GRACE_HOURS)


class PaperSettlementReconcilerTests(unittest.TestCase):
    def _runner(self, directory, binary=None, negrisk=None, cash=1000.0):
        binary = binary or binary_market()
        journal = LiveNegRiskJournal(os.path.join(directory, "paper-negrisk.json"))
        executor = PaperNegRiskExecutor(journal) if negrisk is not None else None
        runner = PaperMarketRunner(
            [binary], os.path.join(directory, "ledger.json"), cash,
            BinaryArbitrageScanner(min_net_profit_usd=0.05, min_return=0.002, safety_buffer_usd=0.02),
            negrisk_markets=[negrisk] if negrisk is not None else None,
            negrisk_scanner=NegRiskBookScanner(
                min_net_profit_usd=0.05, min_return=0.002, safety_buffer_usd=0.02,
            ),
            negrisk_executor=executor,
        )
        if executor is not None:
            executor.ledger = runner.ledger
        return runner

    def test_resolved_expired_binary_emits_settle_pair_and_closes_cash(self):
        market = binary_market()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            runner = self._runner(directory, binary=market)
            cash_before = float(runner.ledger.state["cash"])
            position = open_binary_pair(runner, market)
            cost = float(runner.ledger.state["positions"][position.position_id]["cost"])
            fees = float(runner.ledger.state["positions"][position.position_id]["fees"])
            reserved = cash_before - float(runner.ledger.state["cash"])
            self.assertGreater(reserved, 0.0)
            # Market rotated out of the scanned universe — the original bug.
            runner.markets.pop(market.market_id, None)
            gamma = FakeGamma({
                "ceasefire": {
                    "id": "ceasefire",
                    "conditionId": "c-ceasefire",
                    "question": market.title,
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["1", "0"]',
                    "closed": True,
                    "active": False,
                    "endDate": "2026-09-04T00:00:00Z",
                    "umaResolutionStatus": "resolved",
                },
            })
            report = asyncio.run(PaperSettlementReconciler(runner, getter=gamma).reconcile())
            raw = runner.ledger.state["positions"][position.position_id]
            self.assertTrue(raw["settled"])
            self.assertIn(position.position_id, report.settled_positions)
            settle = next(
                trade for trade in runner.ledger.state["trades"] if trade.get("type") == "SETTLE_PAIR"
            )
            self.assertEqual(settle["winning_outcome"], "Yes")
            self.assertAlmostEqual(settle["payout"], raw["payout"], places=6)
            self.assertAlmostEqual(raw["payout"], float(raw["shares"]) * 1.0, places=6)
            cash_after = float(runner.ledger.state["cash"])
            self.assertAlmostEqual(cash_after, cash_before - reserved + raw["payout"], places=6)
            self.assertLess(abs(cash_after - (cash_before - cost - fees + raw["payout"])), CASH_CLOSE_TOLERANCE)
            health = compute_health_payload(ledger=runner.ledger)
            self.assertEqual(health["open_pairs"], 0)
            self.assertLess(abs(health["pair_exposure"]), CASH_CLOSE_TOLERANCE)
            self.assertEqual(health["unhedged_leg_count"], 0)
            self.assertEqual(health["settlement_stuck"], 0)
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.scanner.min_return, 0.002)
            self.assertEqual(runner.scanner.safety_buffer_usd, 0.02)
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
            self.assertFalse(getattr(runner, "live", False))

    def test_stale_assembled_sports_settles_when_gamma_resolved(self):
        nr = NegRiskMarket.from_gamma(nway_payload())
        binary = binary_market()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            runner = self._runner(directory, binary=binary, negrisk=nr)
            cash_before = float(runner.ledger.state["cash"])
            opened = open_nr_basket(runner, nr)
            journal = runner.negrisk_journal
            basket = journal._record(opened.basket_id)
            self.assertEqual(basket["status"], "ASSEMBLED")
            reserved = float(basket["capital_reserved"])
            self.assertGreater(reserved, 0.0)
            runner.negrisk_markets.pop(nr.market_id, None)
            gamma = FakeGamma({
                "nr-sports": {
                    **nway_payload(
                        closed=True, active=False,
                        outcomePrices='["1", "0", "0"]',
                        umaResolutionStatus="resolved",
                        endDate="2026-09-09T00:00:00Z",
                    ),
                },
            })
            report = asyncio.run(PaperSettlementReconciler(runner, getter=gamma).reconcile())
            basket = journal._record(opened.basket_id)
            self.assertEqual(basket["status"], "SETTLED")
            self.assertEqual(basket["settlement_type"], "PAPER_LEDGER")
            self.assertAlmostEqual(float(basket["capital_reserved"]), 0.0, places=6)
            self.assertIn(opened.basket_id, report.settled_baskets)
            position = next(
                row for row in runner.ledger.state["positions"].values()
                if row.get("basket_id") == opened.basket_id
            )
            self.assertTrue(position["settled"])
            payout = float(position["payout"])
            self.assertAlmostEqual(payout, float(position["shares"]) * float(position["payout_per_share"]), places=6)
            cash_after = float(runner.ledger.state["cash"])
            # Journal reserved is execution capital; ledger cash uses cost. Close both.
            self.assertLess(abs(cash_after - (cash_before - float(position["cost"]) + payout)), CASH_CLOSE_TOLERANCE)
            self.assertLess(abs(float(basket["capital_reserved"])), CASH_CLOSE_TOLERANCE)
            health = compute_health_payload(ledger=runner.ledger, negrisk=journal)
            self.assertEqual(health["open_negrisk"], 0)
            self.assertLess(abs(health["negrisk_exposure"]), CASH_CLOSE_TOLERANCE)
            self.assertEqual(health["unhedged_leg_count"], 0)
            self.assertEqual(journal.open_exposure(), 0.0)
            self.assertEqual(runner.negrisk_scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)

    def test_past_grace_uma_finalized_force_settles_complete_set(self):
        market = binary_market()
        end_ts = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            runner = self._runner(directory, binary=market)
            position = open_binary_pair(runner, market)
            gamma = FakeGamma({
                "ceasefire": {
                    "id": "ceasefire",
                    "conditionId": "c-ceasefire",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.50", "0.50"]',
                    "closed": False,
                    "active": True,
                    "endDate": _iso(end_ts),
                    "umaResolutionStatus": "finalized",
                },
            })
            now = end_ts + 49 * 3600
            report = asyncio.run(
                PaperSettlementReconciler(runner, getter=gamma, grace_hours=48).reconcile(now=now)
            )
            raw = runner.ledger.state["positions"][position.position_id]
            self.assertTrue(raw["settled"])
            settle = next(
                trade for trade in runner.ledger.state["trades"] if trade.get("type") == "SETTLE_PAIR"
            )
            self.assertEqual(settle["winning_outcome"], "UMA_FINALIZED")
            self.assertIn(position.position_id, report.settled_positions)
            self.assertEqual(report.unhedged_leg_count, 0)

    def test_past_grace_without_uma_marks_settlement_stuck(self):
        market = binary_market()
        nr = NegRiskMarket.from_gamma(nway_payload())
        end_ts = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": "", "PAPER_SETTLE_STUCK_ALERT": "1"}, clear=False,
        ):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            runner = self._runner(directory, binary=market, negrisk=nr)
            position = open_binary_pair(runner, market)
            opened = open_nr_basket(runner, nr)
            payload = {
                "id": "ceasefire",
                "conditionId": "c-ceasefire",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
                "closed": False,
                "endDate": _iso(end_ts),
                "umaResolutionStatus": "",
            }
            sports = nway_payload(
                closed=False, active=True,
                endDate=_iso(end_ts),
                umaResolutionStatus="",
            )
            gamma = FakeGamma({"ceasefire": payload, "nr-sports": sports})
            now = end_ts + 72 * 3600
            with mock.patch("polywang.paper_settle.record_monitor_exception") as alert:
                report = asyncio.run(
                    PaperSettlementReconciler(
                        runner, getter=gamma, grace_hours=48, alert=True,
                    ).reconcile(now=now)
                )
            self.assertGreaterEqual(len(report.stuck), 2)
            raw = runner.ledger.state["positions"][position.position_id]
            self.assertFalse(raw.get("settled"))
            self.assertTrue(raw.get("settlement_stuck"))
            basket = runner.negrisk_journal._record(opened.basket_id)
            self.assertEqual(basket["status"], "ASSEMBLED")
            self.assertTrue(basket.get("settlement_stuck"))
            health = compute_health_payload(ledger=runner.ledger, negrisk=runner.negrisk_journal)
            self.assertGreaterEqual(health["settlement_stuck"], 2)
            self.assertEqual(health["unhedged_leg_count"], 0)
            self.assertGreaterEqual(alert.call_count, 1)
            self.assertFalse(any(
                trade.get("type") == "SETTLE_PAIR" for trade in runner.ledger.state["trades"]
            ))

    def test_live_runner_is_rejected_and_loop_is_noop(self):
        market = binary_market()
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory, binary=market)
            runner.live = True
            with self.assertRaisesRegex(RuntimeError, "must not run in live"):
                PaperSettlementReconciler(runner, getter=FakeGamma({}))
            runner.live = False
            runner.ledger = None
            with self.assertRaisesRegex(RuntimeError, "must not run in live"):
                PaperSettlementReconciler(runner, getter=FakeGamma({}))

            class LiveRunner:
                live = True
                ledger = None

            async def _run():
                stop = asyncio.Event()
                stop.set()
                await run_paper_settle_loop(LiveRunner(), stop, getter=FakeGamma({}))

            asyncio.run(_run())

    def test_existing_stream_resolution_still_settles_without_poll(self):
        nr = NegRiskMarket.from_gamma(nway_payload())
        binary = binary_market()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            runner = self._runner(directory, binary=binary, negrisk=nr)
            opened = open_nr_basket(runner, nr)
            asyncio.run(runner.process({
                "event_type": "market_resolved",
                "market": "nr-sports",
                "winning_outcome": "Blue Jays",
            }))
            basket = runner.negrisk_journal._record(opened.basket_id)
            self.assertEqual(basket["status"], "SETTLED")
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.negrisk_scanner.min_return, 0.002)


if __name__ == "__main__":
    unittest.main()
