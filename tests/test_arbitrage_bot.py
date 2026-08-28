import asyncio
import os
import tempfile
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


if __name__ == "__main__":
    unittest.main()
