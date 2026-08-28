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


if __name__ == "__main__":
    unittest.main()
