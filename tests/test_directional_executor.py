import asyncio
import os
import tempfile
import unittest

from polywang.arbitrage_core import (
    BinaryMarket,
    DirectionalExecutor,
    DirectionalIntent,
    LiveDirectionalJournal,
    LiveOrderJournal,
    LiveRiskController,
    OfficialFOKExecutor,
    OrderBook,
    PaperDirectionalExecutor,
    RiskHaltError,
    intent_from_best_ask,
    intent_from_inventory_bid,
)
from test_arbitrage_core import FakeClient, market


class DirectionalExecutorTests(unittest.TestCase):
    def test_paper_buy_and_sell_update_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
            executor = PaperDirectionalExecutor(journal)
            buy = executor.execute(DirectionalIntent(
                token_id="yes-token", side="BUY", shares=5, limit_price=0.40,
                market_id="m1", source="sports", event_id="g1",
            ))
            self.assertEqual(buy.status, "FILLED")
            self.assertAlmostEqual(journal.inventory_by_token()["yes-token"], 5.0)
            sell = executor.execute(DirectionalIntent(
                token_id="yes-token", side="SELL", shares=5, limit_price=0.41,
                market_id="m1", source="sports", event_id="g1-exit",
            ))
            self.assertEqual(sell.status, "FILLED")
            self.assertEqual(journal.inventory_by_token(), {})

    def test_live_buy_uses_official_fok_caps_and_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = LiveOrderJournal(os.path.join(directory, "pairs.json"))
            directional = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
            risk = LiveRiskController(
                pairs, equity_usd=100.0,
                state_path=os.path.join(directory, "risk.json"),
                kill_switch_path="", extra_journals=[directional],
                max_total_exposure_fraction=0.10,
            )
            client = FakeClient()
            live = OfficialFOKExecutor(client, journal=pairs, directional_journal=directional)
            executor = DirectionalExecutor(live, directional, risk=risk)
            result = asyncio.run(executor.execute(DirectionalIntent(
                token_id="yes-token", side="BUY", shares=10, limit_price=0.40,
                market_id="m1", source="macro",
            )))
            self.assertEqual(result.status, "FILLED")
            self.assertEqual(client.calls[0]["order_type"], "FOK")
            self.assertEqual(client.calls[0]["max_price"], "0.400000")
            self.assertGreater(directional.open_exposure(), 0.0)
            with self.assertRaises(RiskHaltError):
                risk.check_directional(20.0, "m1")

    def test_user_stream_sell_is_attributed_to_the_directional_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = LiveOrderJournal(os.path.join(directory, "pairs.json"))
            directional = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
            live = OfficialFOKExecutor(FakeClient(), journal=pairs, directional_journal=directional)
            trade_id = directional.create(DirectionalIntent(
                token_id="yes-token", side="SELL", shares=4, limit_price=0.39, market_id="m1",
            ))
            directional.set_order_id(trade_id, "dir-sell-1")
            updated = live.handle_user_event({
                "event_type": "trade", "order_id": "dir-sell-1", "token_id": "yes-token",
                "side": "SELL", "size": "4", "price": "0.39", "trade_id": "t-sell",
            })
            self.assertEqual(updated["status"], "FILLED")
            self.assertAlmostEqual(updated["matched_shares"], 4.0)

    def test_account_recon_treats_directional_inventory_as_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = LiveOrderJournal(os.path.join(directory, "pairs.json"))
            directional = LiveDirectionalJournal(os.path.join(directory, "dir.json"))
            PaperDirectionalExecutor(directional).execute(DirectionalIntent(
                token_id="yes-token", side="BUY", shares=3, limit_price=0.40, market_id="m1",
            ))
            live = OfficialFOKExecutor(FakeClient(), journal=pairs, directional_journal=directional)
            self.assertIn("yes-token", live._known_live_token_ids())

    def test_intent_helpers_require_synced_books(self):
        binary = market()
        asks = OrderBook()
        asks.asks = {0.42: 8}
        asks.synced = True
        intent = intent_from_best_ask(binary, binary.yes_token_id, asks, 10.0, source="sports")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.side, "BUY")
        unsynced = OrderBook()
        unsynced.asks = {0.42: 8}
        self.assertIsNone(intent_from_best_ask(binary, binary.yes_token_id, unsynced, 10.0, source="sports"))
        bids = OrderBook()
        bids.bids = {0.40: 5}
        bids.synced = True
        sell = intent_from_inventory_bid(binary, binary.yes_token_id, bids, 4.0, source="crypto")
        self.assertEqual(sell.shares, 4.0)
        self.assertEqual(sell.side, "SELL")


if __name__ == "__main__":
    unittest.main()
