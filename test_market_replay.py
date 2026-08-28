import json
import os
import tempfile
import unittest

from market_replay import BinaryMarketReplay, JsonlEventRecorder
from arbitrage_core import BinaryArbitrageScanner, BinaryMarket


class ReplayTests(unittest.TestCase):
    def test_recorder_writes_replayable_jsonl_with_receipt_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.jsonl")
            JsonlEventRecorder(path).record({
                "type": "book", "asset_id": "yes", "timestamp": "1700000000000",
                "bids": [], "asks": [],
            }, received_at_ms=1700000000123)
            with open(path, encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["received_at_ms"], 1700000000123)
            self.assertEqual(row["source"], "market")

    def test_iso_timestamp_from_typed_event_is_accepted(self):
        market = BinaryMarket("m1", "c1", "Test", "yes", "no")
        replay = BinaryMarketReplay([market])
        self.assertEqual(replay.process({
            "event_type": "book", "asset_id": "yes",
            "timestamp": "2023-11-14T22:13:20+00:00", "hash": "y1",
            "asks": [{"price": "0.40", "size": "10"}], "bids": [],
        }), [])
        self.assertEqual(replay.books["yes"].timestamp_ms, 1700000000000)

    def test_replay_uses_snapshot_and_incremental_book_events(self):
        market = BinaryMarket("m1", "c1", "Test", "yes", "no", category="geopolitics")
        replay = BinaryMarketReplay(
            [market],
            scanner=BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
        )
        replay.process({
            "event_type": "book", "asset_id": "yes", "timestamp": "1700000000000", "hash": "y1",
            "asks": [{"price": "0.40", "size": "10"}], "bids": [],
        }, 0)
        self.assertEqual(replay.process({
            "event_type": "book", "asset_id": "no", "timestamp": "1700000000000", "hash": "n1",
            "asks": [{"price": "0.40", "size": "10"}], "bids": [],
        }, 1)[0].fingerprint, "m1:y1:n1:10.000000000000")
        opportunities = replay.process({
            "event_type": "price_change", "timestamp": "1700000001000",
            "price_changes": [{"asset_id": "yes", "price": "0.39", "size": "10", "side": "SELL", "hash": "y2"}],
        }, 2)
        self.assertTrue(opportunities)
        self.assertIn(":y2:n1:", opportunities[0].fingerprint)

    def test_consume_fills_changes_subsequent_replay_capacity(self):
        market = BinaryMarket("m1", "c1", "Test", "yes", "no", category="geopolitics")
        replay = BinaryMarketReplay(
            [market],
            scanner=BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
            consume_fills=True,
        )
        for token in ("yes", "no"):
            replay.process({
                "event_type": "book", "asset_id": token, "timestamp": "1700000000000",
                "hash": token, "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            })
        first = replay.opportunities
        self.assertEqual(len(first), 1)
        replay.process({
                "event_type": "price_change", "timestamp": "1700000001000",
                "price_changes": [
                {"asset_id": "yes", "price": "0.40", "size": "0", "side": "SELL", "hash": "yes-2"},
                {"asset_id": "no", "price": "0.40", "size": "0", "side": "SELL", "hash": "no-2"},
            ],
        })
        self.assertEqual(len(replay.opportunities), 1)

    def test_latency_replay_counts_opportunity_that_disappears_before_execution(self):
        market = BinaryMarket("m1", "c1", "Test", "yes", "no", category="geopolitics")
        replay = BinaryMarketReplay(
            [market],
            scanner=BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
            consume_fills=True, execution_latency_ms=1_000,
        )
        base = 1_700_000_000_000
        for token in ("yes", "no"):
            replay.process({
                "event_type": "book", "asset_id": token, "timestamp": str(base),
                "received_at_ms": base, "hash": token,
                "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            })
        replay.process({
            "event_type": "price_change", "timestamp": str(base + 500),
            "received_at_ms": base + 500,
            "price_changes": [{"asset_id": "yes", "price": "0.40", "size": "0", "side": "SELL"}],
        })
        replay.process({
            "event_type": "price_change", "timestamp": str(base + 1_500),
            "received_at_ms": base + 1_500,
            "price_changes": [{"asset_id": "no", "price": "0.40", "size": "10", "side": "SELL"}],
        })
        self.assertEqual(replay.report()["execution"]["signals"], 1)
        self.assertEqual(replay.report()["execution"]["executed"], 0)
        self.assertEqual(replay.report()["execution"]["latency_missed"], 1)
        self.assertEqual(replay.report()["executed_opportunities"], 0)
        self.assertEqual(replay.report()["executed_net_profit"], 0)

    def test_zero_latency_reports_simulated_executed_profit_separately(self):
        market = BinaryMarket("m1", "c1", "Test", "yes", "no", category="geopolitics")
        replay = BinaryMarketReplay(
            [market],
            scanner=BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
            consume_fills=True,
        )
        for token in ("yes", "no"):
            replay.process({
                "event_type": "book", "asset_id": token, "timestamp": "1700000000000",
                "hash": token, "asks": [{"price": "0.40", "size": "10"}], "bids": [],
            })
        report = replay.report()
        self.assertEqual(report["execution"]["signals"], 1)
        self.assertEqual(report["execution"]["executed"], 1)
        self.assertEqual(report["executed_opportunities"], 1)
        self.assertGreater(report["executed_net_profit"], 0.0)


if __name__ == "__main__":
    unittest.main()
