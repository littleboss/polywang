import asyncio
import os
import tempfile
import unittest
from unittest import mock

from arbitrage_bot import PaperMarketRunner, run_user_stream
from arbitrage_core import BinaryArbitrageScanner, BinaryMarket, UnhedgedPairError


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

        with mock.patch("arbitrage_bot.consume_user_stream", failing_consumer):
            with self.assertRaises(UnhedgedPairError):
                asyncio.run(run_user_stream(executor, risk))
        self.assertIn("unhedged", risk.reason)


if __name__ == "__main__":
    unittest.main()
