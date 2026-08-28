import unittest

from sports_channel import SportsLatencyGate, SportsStateTracker


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
        from sports_channel import SportsMarketMap, evaluate_sports_candidate
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
        from sports_channel import SportsMarketMap, evaluate_sports_candidate
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


if __name__ == "__main__":
    unittest.main()
