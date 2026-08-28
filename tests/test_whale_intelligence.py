import os
import tempfile
import unittest

from polywang.whale_intelligence import WhaleIntelligenceEngine, normalize_wallet


WALLET_A = "0x" + "a" * 40
WALLET_B = "0x" + "b" * 40


class WhaleIntelligenceTests(unittest.TestCase):
    def test_anonymous_or_malformed_addresses_are_not_wallets(self):
        self.assertEqual(normalize_wallet("0x0"), "")
        self.assertEqual(normalize_wallet("0xalice"), "")
        engine = WhaleIntelligenceEngine()
        self.assertIsNone(engine.record_trade({
            "id": "anonymous-1", "market_id": "m1", "outcome": "Yes",
            "wallet_address": "0x0", "price": 0.4, "size": 20000,
        }))
        self.assertEqual(engine.state["anonymous_events"], 1)

    def test_missing_side_is_not_assumed_to_be_buy(self):
        engine = WhaleIntelligenceEngine()
        self.assertIsNone(engine.record_trade({
            "id": "no-side", "market_id": "m1", "outcome": "Yes",
            "wallet_address": WALLET_A, "price": 0.4, "size": 20000,
        }))
        self.assertIsNone(engine.snapshot(WALLET_A))

    def test_duplicate_trade_id_is_ignored(self):
        engine = WhaleIntelligenceEngine(threshold_usd=1, min_unique_wallets=1)
        payload = {"id": "t1", "market_id": "m1", "outcome": "Yes",
                   "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 10}
        self.assertIsNotNone(engine.record_trade(payload))
        self.assertIsNone(engine.record_trade(payload))
        self.assertEqual(engine.snapshot(WALLET_A)["trades"], 1)

    def test_settlement_builds_a_shrunk_wallet_quality_score(self):
        engine = WhaleIntelligenceEngine(threshold_usd=1, min_unique_wallets=1)
        for index in range(2):
            market_id = f"history-{index}"
            engine.record_trade({"id": f"buy-{index}", "market_id": market_id, "outcome": "Yes",
                                 "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 10})
            self.assertEqual(engine.settle_market(market_id, "Yes"), 1)
        snapshot = engine.snapshot(WALLET_A)
        self.assertEqual(snapshot["settled_markets"], 2)
        self.assertGreater(snapshot["realized_pnl"], 0)
        self.assertGreater(snapshot["quality"], 0.5)

    def test_signal_requires_history_and_quality_weighted_flow(self):
        engine = WhaleIntelligenceEngine(
            threshold_usd=1, min_unique_wallets=2, min_settled_markets=2,
            min_quality=0.55, min_pressure=0.6,
        )
        for index in range(2):
            market_id = f"history-{index}"
            engine.record_trade({"id": f"buy-{index}", "market_id": market_id, "outcome": "Yes",
                                 "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 10})
            engine.settle_market(market_id, "Yes")

        first = engine.record_trade({"id": "flow-a-1", "market_id": "live", "outcome": "Yes",
                                     "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 20})
        self.assertFalse(first.eligible)
        engine.record_trade({"id": "flow-b-1", "market_id": "live", "outcome": "Yes",
                             "wallet_address": WALLET_B, "side": "BUY", "price": 0.4, "size": 20})
        qualified = engine.record_trade({"id": "flow-a-2", "market_id": "live", "outcome": "Yes",
                                         "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 20})
        self.assertTrue(qualified.eligible)
        self.assertEqual(qualified.unique_wallets, 2)
        self.assertGreaterEqual(qualified.market_pressure, 0.6)

    def test_coordination_requires_same_outcome_and_direction(self):
        engine = WhaleIntelligenceEngine(
            threshold_usd=1, min_coordination_trade_usd=0.1,
            min_unique_wallets=2, min_settled_markets=1,
        )
        engine.record_trade({"id": "yes", "market_id": "m1", "outcome": "Yes",
                             "wallet_address": WALLET_A, "side": "BUY", "price": 0.5, "size": 10})
        engine.record_trade({"id": "no", "market_id": "m1", "outcome": "No",
                             "wallet_address": WALLET_B, "side": "BUY", "price": 0.5, "size": 10})
        signal = engine.coordination_signal("m1")
        self.assertFalse(signal.eligible)
        self.assertEqual(signal.unique_wallets, 1)

    def test_coordination_rejects_one_wallet_dominating_the_burst(self):
        engine = WhaleIntelligenceEngine(
            threshold_usd=100, min_coordination_trade_usd=1,
            min_unique_wallets=3, min_settled_markets=1,
            max_concentration=0.75,
        )
        for index, (wallet, size) in enumerate(((WALLET_A, 180), (WALLET_B, 10), ("0x" + "c" * 40, 10))):
            engine.record_trade({"id": f"t-{index}", "market_id": "m2", "outcome": "Yes",
                                 "wallet_address": wallet, "side": "BUY", "price": 0.5, "size": size})
        signal = engine.coordination_signal("m2")
        self.assertFalse(signal.eligible)
        self.assertGreater(signal.max_wallet_share, 0.75)

    def test_state_persists_wallet_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "whales.json")
            engine = WhaleIntelligenceEngine(path=path)
            engine.record_trade({"id": "t1", "market_id": "m1", "outcome": "Yes",
                                 "wallet_address": WALLET_A, "side": "BUY", "price": 0.4, "size": 10})
            restored = WhaleIntelligenceEngine(path=path)
            self.assertEqual(restored.snapshot(WALLET_A)["trades"], 1)


if __name__ == "__main__":
    unittest.main()
