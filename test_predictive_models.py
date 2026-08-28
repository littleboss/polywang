import os
import tempfile
import unittest

from crypto_model import CryptoObservation, CryptoStatArbModel
from macro_model import MacroEventModel, MacroRelease
from polymarket_edge import CalibrationTracker


class PredictiveModelTests(unittest.TestCase):
    def test_predictive_payloads_reject_non_finite_values(self):
        self.assertIsNone(MacroRelease.from_payload({
            "id": "e", "indicator": "cpi", "actual": "nan", "consensus": 1,
            "historical_std": 1, "released_at_ms": 1700000000000,
        }))
        self.assertIsNone(CryptoObservation.from_payload({
            "market_id": "m", "market_probability": "inf",
            "reference_probability": 0.5, "timestamp_ms": 1700000000000,
        }))

    def test_macro_model_rejects_until_calibration_is_ready(self):
        tracker = CalibrationTracker(min_samples=2)
        model = MacroEventModel(tracker, min_edge=0.01)
        release = MacroRelease("e1", "cpi", 5.0, 4.0, 0.5, 1_700_000_000_000)
        signal = model.predict(release, 0.5, 1_700_000_001_000)
        self.assertFalse(signal.eligible)
        self.assertIn("calibration", signal.reason)

    def test_macro_model_becomes_ready_only_after_evidence(self):
        tracker = CalibrationTracker(min_samples=2)
        model = MacroEventModel(tracker, min_edge=0.01, surprise_weight=1.0)
        release = MacroRelease("e1", "cpi", 5.0, 4.0, 0.5, 1_700_000_000_000)
        tracker.record(model.strategy, 0.9, 1)
        tracker.record(model.strategy, 0.1, 0)
        self.assertTrue(model.predict(release, 0.5, 1_700_000_001_000).eligible)

    def test_crypto_model_requires_history_and_calibration(self):
        tracker = CalibrationTracker(min_samples=1)
        model = CryptoStatArbModel(tracker, entry_zscore=1.0)
        timestamp = 1_700_000_000_000
        for index in range(12):
            signal = model.observe(CryptoObservation("m", 0.5, 0.5, timestamp + index), timestamp + index)
        self.assertFalse(signal.eligible)
        self.assertIn("calibration", signal.reason)

    def test_crypto_signal_is_freshness_gated(self):
        tracker = CalibrationTracker(min_samples=1)
        model = CryptoStatArbModel(tracker, entry_zscore=1.0, max_age_seconds=1)
        tracker.record(model.strategy, 0.9, 1)
        history_time = 1_700_000_000_000
        for index in range(12):
            model.observe(CryptoObservation("m", 0.5, 0.5, history_time + index), history_time + index)
        signal = model.observe(CryptoObservation("m", 0.9, 0.5, history_time + 12), history_time + 5_000)
        self.assertFalse(signal.eligible)
        self.assertIn("freshness", signal.reason)

    def test_calibration_tracker_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibration.json")
            tracker = CalibrationTracker(min_samples=1, path=path)
            tracker.record("macro", 0.8, 1)
            restored = CalibrationTracker(min_samples=1, path=path)
            self.assertTrue(restored.is_live_ready("macro"))

    def test_macro_event_is_deduplicated_and_can_require_a_market_map(self):
        tracker = CalibrationTracker(min_samples=2)
        tracker.record("macro-event-v1", 0.9, 1)
        tracker.record("macro-event-v1", 0.1, 0)
        model = MacroEventModel(tracker, min_edge=0.01, surprise_weight=1.0,
                                market_map={"cpi": "m-cpi"})
        release = MacroRelease("e1", "cpi", 5.0, 4.0, 0.5, 1_700_000_000_000)
        first = model.predict(release, 0.5, 1_700_000_001_000)
        second = model.predict(release, 0.5, 1_700_000_001_000)
        self.assertTrue(first.eligible)
        self.assertEqual(first.market_id, "m-cpi")
        self.assertFalse(first.executable)
        self.assertFalse(second.eligible)
        self.assertIn("duplicate", second.reason)
        unmapped = MacroEventModel(tracker, min_edge=0.01, surprise_weight=1.0,
                                   market_map={"other": "m-x"})
        self.assertIn("mapped", unmapped.predict(release, 0.5, 1_700_000_001_000).reason)

    def test_crypto_sell_and_inventory_are_not_executable_on_the_fok_pair_executor(self):
        tracker = CalibrationTracker(min_samples=1)
        tracker.record("crypto-spread-v1", 0.9, 1)
        history_time = 1_700_000_000_000

        def warm(model, market_id="m"):
            for index in range(12):
                market_p = 0.50 + (0.01 if index % 2 == 0 else -0.01)
                model.observe(
                    CryptoObservation(market_id, market_p, 0.5, history_time + index),
                    history_time + index,
                )

        sell_model = CryptoStatArbModel(tracker, entry_zscore=1.0, exit_zscore=0.2, max_inventory=1)
        warm(sell_model)
        sell = sell_model.observe(CryptoObservation("m", 0.9, 0.5, history_time + 12), history_time + 12)
        self.assertEqual(sell.direction, "SELL_MARKET")
        self.assertFalse(sell.eligible)
        self.assertIn("FOK", sell.reason)

        buy_model = CryptoStatArbModel(tracker, entry_zscore=1.0, exit_zscore=0.5)
        warm(buy_model)
        buy = buy_model.observe(CryptoObservation("m", 0.2, 0.5, history_time + 12), history_time + 12)
        self.assertEqual(buy.direction, "BUY_MARKET")
        self.assertTrue(buy.eligible)
        self.assertFalse(buy.executable)
        inventory_model = CryptoStatArbModel(tracker, entry_zscore=1.0, exit_zscore=0.5)
        warm(inventory_model)
        inventory_model.inventory.positions["m"] = "BUY_MARKET"
        held = inventory_model.observe(
            CryptoObservation("m", 0.50, 0.5, history_time + 12),
            history_time + 12,
        )
        self.assertEqual(held.action, "EXIT")
        self.assertFalse(held.executable)

    def test_macro_jsonl_feed_and_execution_flag(self):
        from macro_model import JsonlMacroFeed, MacroEventModel, MacroRelease
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "macro.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"event_id":"e1","indicator":"cpi","actual":5,"consensus":4,'
                             '"historical_std":0.5,"released_at_ms":1700000000000}\n')
            feed = JsonlMacroFeed(path)
            releases = feed.poll()
            self.assertEqual(len(releases), 1)
            self.assertEqual(feed.poll(), [])
        tracker = CalibrationTracker(min_samples=2)
        tracker.record("macro-event-v1", 0.9, 1)
        tracker.record("macro-event-v1", 0.1, 0)
        model = MacroEventModel(tracker, min_edge=0.01, surprise_weight=1.0, market_map={"cpi": "m-cpi"})
        model.allow_execution = True
        signal = model.predict(MacroRelease("e2", "cpi", 5.0, 4.0, 0.5, 1_700_000_000_000), 0.5, 1_700_000_001_000)
        self.assertTrue(signal.executable)
        self.assertEqual(signal.direction, "BUY_YES")

    def test_crypto_reference_adapter_and_directional_sell_entry(self):
        from crypto_model import CryptoReferenceAdapter, CryptoStatArbModel, digital_call_probability
        probability = digital_call_probability(65000, 60000, 0.55, 0.08)
        self.assertIsNotNone(probability)
        self.assertGreater(probability, 0.5)
        quote = CryptoReferenceAdapter().parse({
            "market_id": "m", "spot": 65000, "strike": 60000, "vol": 0.55,
            "t": 0.08, "timestamp_ms": 1700000000000, "source": "fixture",
        })
        self.assertAlmostEqual(quote.implied_probability, probability)
        tracker = CalibrationTracker(min_samples=1)
        tracker.record("crypto-spread-v1", 0.9, 1)
        model = CryptoStatArbModel(tracker, entry_zscore=1.0, allow_execution=True)
        history_time = 1_700_000_000_000
        for index in range(12):
            market_p = 0.50 + (0.01 if index % 2 == 0 else -0.01)
            model.observe(CryptoObservation("m", market_p, 0.5, history_time + index), history_time + index)
        sell = model.observe(CryptoObservation("m", 0.9, 0.5, history_time + 12), history_time + 12)
        self.assertEqual(sell.direction, "BUY_NO")
        self.assertTrue(sell.executable)
        self.assertEqual(sell.action, "ENTER")
        model.mark_open(sell, token_id="no-token", shares=2.0)
        exit_signal = model.observe(CryptoObservation("m", 0.50, 0.5, history_time + 13), history_time + 13)
        self.assertEqual(exit_signal.action, "EXIT")
        self.assertTrue(exit_signal.executable)


if __name__ == "__main__":
    unittest.main()
