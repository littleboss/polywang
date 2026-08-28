import unittest

from polywang.sports_channel import SportsLatencyGate, SportsStateTracker


class SportsChannelTests(unittest.TestCase):
    def test_string_booleans_are_parsed_semantically(self):
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": 12, "status": "FINAL", "live": "false", "ended": "true",
            "score": "1-0", "period": "4",
        })
        self.assertFalse(observation.live)
        self.assertTrue(observation.ended)

    def test_score_change_requires_explicit_source_timestamp(self):
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": 12, "status": "LIVE", "live": True, "ended": False,
            "score": "1-0", "period": "2", "timestamp": None,
        }, received_at_ms=1_700_000_005_000)
        decision = SportsLatencyGate().evaluate(observation, 1_700_000_000_000,
                                                now_ms=1_700_000_005_000)
        self.assertFalse(decision.eligible)
        self.assertIn("timestamp", decision.reason)

    def test_timestamp_proven_gap_is_eligible_only_inside_window(self):
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": 12, "status": "LIVE", "live": True, "ended": False,
            "score": "1-0", "period": "2", "source_timestamp": 1_700_000_004_000,
        }, received_at_ms=1_700_000_005_000)
        gate = SportsLatencyGate(max_age_seconds=5, min_delay_ms=100, max_delay_ms=5_000)
        self.assertTrue(gate.evaluate(observation, 1_700_000_000_000,
                                      now_ms=1_700_000_005_000).eligible)
        self.assertFalse(gate.evaluate(observation, 1_700_000_004_500,
                                       now_ms=1_700_000_005_000).eligible)

    def test_duplicate_score_is_not_a_new_signal(self):
        tracker = SportsStateTracker()
        payload = {"gameId": 12, "status": "LIVE", "live": True, "ended": False,
                   "score": "1-0", "period": "2", "source_timestamp": 1_700_000_004_000}
        tracker.observe(payload, received_at_ms=1_700_000_005_000)
        duplicate = tracker.observe(payload, received_at_ms=1_700_000_005_100)
        self.assertFalse(duplicate.changed)

    def test_unmapped_game_is_observational_only(self):
        from polywang.sports_channel import SportsMarketMap, evaluate_sports_candidate
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": 12, "status": "LIVE", "live": True, "ended": False,
            "score": "1-0", "period": "2", "source_timestamp": 1_700_000_004_000,
        }, received_at_ms=1_700_000_005_000)
        candidate = evaluate_sports_candidate(
            observation, SportsLatencyGate(), SportsMarketMap(), 1_700_000_000_000,
        )
        self.assertFalse(candidate.eligible)
        self.assertFalse(candidate.executable)
        self.assertIn("not mapped", candidate.reason)

    def test_mapped_latency_candidate_is_not_routed_to_the_fok_executor(self):
        from polywang.sports_channel import SportsMarketMap, evaluate_sports_candidate
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": "g1", "status": "LIVE", "live": True, "ended": False,
            "score": "1-0", "period": "2", "source_timestamp": 1_700_000_004_000,
        }, received_at_ms=1_700_000_005_000)
        mapping = SportsMarketMap({"g1": {"market_id": "m-home", "yes_means": "home"}})
        candidate = evaluate_sports_candidate(
            observation, SportsLatencyGate(max_age_seconds=5, min_delay_ms=100, max_delay_ms=5_000),
            mapping, 1_700_000_000_000, market_price=0.42, now_ms=1_700_000_005_000,
        )
        self.assertTrue(candidate.eligible)
        self.assertFalse(candidate.executable)
        self.assertEqual(candidate.market_id, "m-home")
        self.assertEqual(candidate.direction, "BUY_YES")
        self.assertGreater(candidate.fair_probability, 0.5)

    def test_allow_execution_marks_a_mapped_candidate_executable(self):
        from polywang.polymarket_edge import CalibrationTracker
        from polywang.sports_channel import SportsMarketMap, evaluate_sports_candidate
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": "g1", "status": "LIVE", "live": True, "ended": False,
            "score": "2-0", "period": "70'", "source_timestamp": 1_700_000_004_000,
        }, received_at_ms=1_700_000_005_000)
        mapping = SportsMarketMap({"g1": {"market_id": "m-home", "yes_means": "home"}})
        calibration = CalibrationTracker(min_samples=2)
        calibration.record("sports-latency-v1", 0.9, 1)
        calibration.record("sports-latency-v1", 0.1, 0)
        candidate = evaluate_sports_candidate(
            observation, SportsLatencyGate(max_age_seconds=5, min_delay_ms=100, max_delay_ms=5_000),
            mapping, 1_700_000_000_000, market_price=0.42, now_ms=1_700_000_005_000,
            allow_execution=True, min_edge=0.03, yes_token_id="yes-token", no_token_id="no-token",
            calibration=calibration,
        )
        self.assertTrue(candidate.executable)
        self.assertEqual(candidate.direction, "BUY_YES")
        self.assertEqual(candidate.token_id, "yes-token")

    def test_period_clock_minute_is_parsed_and_calibration_gates_execution(self):
        from polywang.polymarket_edge import CalibrationTracker
        from polywang.sports_channel import (
            SportsMarketMap, evaluate_sports_candidate, parse_soccer_minute,
        )
        self.assertEqual(parse_soccer_minute("70'"), 70.0)
        self.assertEqual(parse_soccer_minute("70+2"), 72.0)
        self.assertEqual(parse_soccer_minute("2H 63"), 63.0)
        self.assertEqual(parse_soccer_minute("HT"), 45.0)
        self.assertEqual(parse_soccer_minute("2"), 70.0)
        tracker = SportsStateTracker()
        observation = tracker.observe({
            "gameId": "g1", "status": "LIVE", "live": True, "ended": False,
            "score": "2-0", "period": "70'", "source_timestamp": 1_700_000_004_000,
        }, received_at_ms=1_700_000_005_000)
        mapping = SportsMarketMap({"g1": {"market_id": "m-home", "yes_means": "home"}})
        blocked = evaluate_sports_candidate(
            observation, SportsLatencyGate(max_age_seconds=5, min_delay_ms=100, max_delay_ms=5_000),
            mapping, 1_700_000_000_000, market_price=0.42, now_ms=1_700_000_005_000,
            allow_execution=True, min_edge=0.03, yes_token_id="yes-token",
            calibration=CalibrationTracker(min_samples=20),
        )
        self.assertFalse(blocked.executable)
        self.assertIn("calibration", blocked.reason)


if __name__ == "__main__":
    unittest.main()
