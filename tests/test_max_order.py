#!/usr/bin/env python3
"""QUANT-20260915-02: dynamic max order from paper ledger cash."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from polywang.arbitrage_bot import PaperMarketRunner, ScanRejectCounter
from polywang.arbitrage_core import (
    BinaryArbitrageScanner,
    BinaryMarket,
    JsonLedger,
    MaxOrderPolicy,
    clamp_order_usd,
    effective_max_order_usd,
    ledger_available_cash,
)
from polywang.negrisk import NegRiskBookScanner


def _binary(**overrides):
    fields = dict(
        market_id="m1",
        condition_id="c1",
        title="Test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        category="geopolitics",
    )
    fields.update(overrides)
    return BinaryMarket(**fields)


class EffectiveMaxOrderTests(unittest.TestCase):
    def test_cash_1000_is_about_25(self):
        self.assertAlmostEqual(effective_max_order_usd(1000.0), 25.0, places=6)

    def test_cash_10000_hits_cap(self):
        self.assertAlmostEqual(effective_max_order_usd(10000.0), 100.0, places=6)
        self.assertAlmostEqual(effective_max_order_usd(20000.0), 100.0, places=6)

    def test_very_low_cash_not_below_floor(self):
        self.assertAlmostEqual(effective_max_order_usd(0.0), 5.0, places=6)
        self.assertAlmostEqual(effective_max_order_usd(1.0), 5.0, places=6)
        self.assertAlmostEqual(effective_max_order_usd(80.0), 5.0, places=6)

    def test_recompute_when_cash_changes(self):
        policy = MaxOrderPolicy()
        self.assertAlmostEqual(policy.effective(1000.0), 25.0, places=6)
        self.assertAlmostEqual(policy.effective(400.0), 10.0, places=6)
        self.assertAlmostEqual(policy.effective(10000.0), 100.0, places=6)
        self.assertAlmostEqual(policy.effective(10.0), 5.0, places=6)

    def test_compat_fixed_max_order_when_fraction_disabled(self):
        self.assertAlmostEqual(
            effective_max_order_usd(
                1000.0, fraction=0.0, fixed_max_order_usd=7.0,
            ),
            7.0,
            places=6,
        )
        self.assertAlmostEqual(
            effective_max_order_usd(
                50.0, fraction=0.0, fixed_max_order_usd=40.0,
            ),
            40.0,
            places=6,
        )

    def test_fraction_on_ignores_fixed_override(self):
        self.assertAlmostEqual(
            effective_max_order_usd(
                1000.0, fraction=0.025, fixed_max_order_usd=7.0,
            ),
            25.0,
            places=6,
        )

    def test_clamp_swaps_inverted_bounds(self):
        self.assertAlmostEqual(clamp_order_usd(50.0, 100.0, 5.0), 50.0, places=6)

    def test_prefers_cash_after_reserved_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = JsonLedger(os.path.join(directory, "ledger.json"), initial_cash=1000.0)
            self.assertAlmostEqual(ledger.available_cash(), 1000.0, places=6)
            self.assertAlmostEqual(ledger_available_cash(ledger), 1000.0, places=6)
            ledger.state["cash"] = 900.0
            ledger.state["cash_after_reserved"] = 400.0
            self.assertAlmostEqual(ledger_available_cash(ledger), 400.0, places=6)
            self.assertAlmostEqual(
                effective_max_order_usd(ledger_available_cash(ledger)), 10.0, places=6,
            )
            del ledger.state["cash_after_reserved"]
            self.assertAlmostEqual(ledger_available_cash(ledger), 900.0, places=6)

    def test_no_ledger_returns_none(self):
        self.assertIsNone(ledger_available_cash(None))


class MaxOrderPolicyEnvTests(unittest.TestCase):
    def test_from_env_defaults_enable_dynamic(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_ORDER_FRACTION_OF_CASH", None)
            os.environ.pop("MAX_ORDER_FLOOR_USD", None)
            os.environ.pop("MAX_ORDER_CAP_USD", None)
            os.environ.pop("MAX_ORDER_USD", None)
            policy = MaxOrderPolicy.from_env()
            self.assertAlmostEqual(policy.fraction, 0.025, places=6)
            self.assertAlmostEqual(policy.floor_usd, 5.0, places=6)
            self.assertAlmostEqual(policy.cap_usd, 100.0, places=6)
            self.assertTrue(policy.dynamic)
            self.assertAlmostEqual(policy.effective(1000.0), 25.0, places=6)

    def test_from_env_compat_uses_explicit_max_order_usd(self):
        with mock.patch.dict(
            os.environ,
            {"MAX_ORDER_FRACTION_OF_CASH": "0", "MAX_ORDER_USD": "12"},
            clear=False,
        ):
            policy = MaxOrderPolicy.from_env()
            self.assertFalse(policy.dynamic)
            self.assertAlmostEqual(policy.effective(1000.0), 12.0, places=6)

    def test_from_env_cli_max_order_used_when_fraction_off(self):
        with mock.patch.dict(
            os.environ, {"MAX_ORDER_FRACTION_OF_CASH": "0"}, clear=False,
        ):
            os.environ.pop("MAX_ORDER_USD", None)
            policy = MaxOrderPolicy.from_env(cli_max_order=9.0)
            self.assertAlmostEqual(policy.effective(500.0), 9.0, places=6)

    def test_from_env_fraction_wins_over_explicit_max_order_usd(self):
        with mock.patch.dict(
            os.environ,
            {"MAX_ORDER_FRACTION_OF_CASH": "0.025", "MAX_ORDER_USD": "7"},
            clear=False,
        ):
            policy = MaxOrderPolicy.from_env(cli_max_order=7.0)
            self.assertTrue(policy.dynamic)
            self.assertAlmostEqual(policy.effective(1000.0), 25.0, places=6)


class RunnerEffectiveMaxTests(unittest.TestCase):
    def _runner(self, directory, cash=1000.0, policy=None):
        policy = policy if policy is not None else MaxOrderPolicy()
        runner = PaperMarketRunner(
            [_binary()],
            os.path.join(directory, "ledger.json"),
            cash,
            BinaryArbitrageScanner(
                min_net_profit_usd=0.05, min_return=0.002, safety_buffer_usd=0.02,
            ),
            negrisk_scanner=NegRiskBookScanner(
                min_net_profit_usd=0.05, min_return=0.002, safety_buffer_usd=0.02,
            ),
            max_order_policy=policy,
        )
        runner.executor.max_total_exposure_fraction = 1.0
        runner.executor.max_market_exposure_fraction = 1.0
        runner.max_book_age_seconds = 1e9
        runner.scan_rejects.flush_interval_s = 3600.0
        return runner

    def test_binary_and_negrisk_share_helper_and_recompute(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            runner = self._runner(directory, cash=1000.0)
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.scanner.min_return, 0.002)
            self.assertEqual(runner.scanner.safety_buffer_usd, 0.02)
            first = runner._refresh_effective_max_order()
            self.assertAlmostEqual(first, 25.0, places=6)
            self.assertAlmostEqual(runner.scanner.max_order_usd, 25.0, places=6)
            self.assertAlmostEqual(runner.negrisk_scanner.max_order_usd, 25.0, places=6)
            self.assertEqual(runner.scanner.max_order_usd, runner.negrisk_scanner.max_order_usd)

            runner.ledger.state["cash"] = 10000.0
            self.assertAlmostEqual(runner._refresh_effective_max_order(), 100.0, places=6)
            self.assertAlmostEqual(runner.scanner.max_order_usd, 100.0, places=6)
            self.assertAlmostEqual(runner.negrisk_scanner.max_order_usd, 100.0, places=6)

            runner.ledger.state["cash"] = 20.0
            self.assertAlmostEqual(runner._refresh_effective_max_order(), 5.0, places=6)
            self.assertAlmostEqual(runner.scanner.max_order_usd, 5.0, places=6)
            self.assertAlmostEqual(runner.negrisk_scanner.max_order_usd, 5.0, places=6)

            runner.ledger.state["cash_after_reserved"] = 400.0
            self.assertAlmostEqual(runner._refresh_effective_max_order(), 10.0, places=6)

    def test_compat_fixed_policy_does_not_follow_cash(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            policy = MaxOrderPolicy(fraction=0.0, fixed_max_order_usd=8.0)
            runner = self._runner(directory, cash=1000.0, policy=policy)
            self.assertAlmostEqual(runner._refresh_effective_max_order(), 8.0, places=6)
            runner.ledger.state["cash"] = 10000.0
            self.assertAlmostEqual(runner._refresh_effective_max_order(), 8.0, places=6)
            self.assertAlmostEqual(runner.scanner.max_order_usd, 8.0, places=6)
            self.assertAlmostEqual(runner.negrisk_scanner.max_order_usd, 8.0, places=6)

    def test_scan_and_open_logs_record_effective_max(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            runner = self._runner(directory, cash=1000.0)
            now = int(__import__("time").time() * 1000)
            with self.assertLogs("arbitrage-bot", level="INFO") as captured:
                asyncio.run(runner.process({
                    "event_type": "book", "asset_id": "yes-token", "timestamp": str(now),
                    "hash": "y", "asks": [{"price": "0.40", "size": "20"}], "bids": [],
                }))
                asyncio.run(runner.process({
                    "event_type": "book", "asset_id": "no-token", "timestamp": str(now),
                    "hash": "n", "asks": [{"price": "0.40", "size": "20"}], "bids": [],
                }))
                runner.scan_rejects.flush()
            open_lines = [line for line in captured.output if "PAPER ARB:" in line]
            self.assertTrue(open_lines)
            self.assertTrue(any("effective_max_order_usd $25.0000" in line for line in open_lines))
            scan_lines = [line for line in captured.output if "SCAN REJECTS:" in line]
            self.assertTrue(scan_lines)
            self.assertTrue(any("effective_max_order_usd=25.0000" in line for line in scan_lines))
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.05)
            self.assertEqual(runner.scanner.min_return, 0.002)
            self.assertEqual(runner.scanner.safety_buffer_usd, 0.02)
            self.assertGreater(len(runner.ledger.state["positions"]), 0)

    def test_scan_reject_counter_includes_effective_max_field(self):
        lines = []

        class Capture:
            def info(self, message, *args):
                lines.append(message % args if args else message)

        counter = ScanRejectCounter(flush_interval_s=3600.0, logger=Capture())
        counter.record("no_touch")
        counter.flush()
        self.assertIn("effective_max_order_usd=n/a", lines[0])
        counter.note_effective_max_order(25.0)
        counter.record("no_depth")
        counter.flush()
        self.assertIn("effective_max_order_usd=25.0000", lines[1])


if __name__ == "__main__":
    unittest.main()
