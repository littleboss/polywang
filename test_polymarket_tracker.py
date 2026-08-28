#!/usr/bin/env python3
"""
Regression tests for the v7 safety systems in polymarket-tracker.py.

Every test here pins down a guard that the maintenance spec treats as
load-bearing. They use only the standard library so they can run anywhere,
and they never touch the network.

Run with:  python3 -m unittest -v test_polymarket_tracker
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

# The module filename contains a hyphen, so it cannot be imported by name.
# Load it by path instead, from a scratch directory so the import-time .env
# lookup cannot pick up a developer's real credentials.
_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polymarket-tracker.py")


def _load_tracker_module():
    original_cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="polymarket-tests-")
    try:
        os.chdir(scratch)
        spec = importlib.util.spec_from_file_location("polymarket_tracker", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["polymarket_tracker"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(original_cwd)


tracker_mod = _load_tracker_module()


DEFAULT_CONFIG = {
    "PAPER_TRADING": True,
    "INITIAL_BALANCE": 1000.0,
    "SIMULATED_FEE_PCT": 0.015,
    "SLIPPAGE_PCT": 0.005,
    "SETTLEMENT_FEE_PCT": 0.015,
    "MIN_NET_PROFIT_MARGIN": 0.05,
    "MIN_ARBITRAGE_EDGE_PCT": 0.05,
    "SPORTS_LATENCY_THRESHOLD_SECS": 5,
    "TIME_DECAY_WEIGHT": 1.0,
    "STOP_LOSS_ENABLED": True,
    "WHALE_USD_THRESHOLD": 5000.0,
    "COORDINATION_WINDOW_SECS": 60,
    "COORDINATION_MIN_UNIQUE_WALLETS": 7,
    "MOMENTUM_WINDOW_SECS": 30,
    "MOMENTUM_VOLUME_MULTIPLIER": 3.0,
    "OVERREACTION_PRICE_DELTA": 0.10,
    "PARITY_ARBITRAGE_THRESHOLD": 0.02,
    "POLY_API_KEY": "",
    "POLY_API_SECRET": "",
    "POLY_PASSPHRASE": "",
    "POLY_PRIVATE_KEY": "",
    "HTTP_PROXY": "",
}


class TrackerTestCase(unittest.TestCase):
    """Gives every test a fresh CONFIG so overrides cannot leak between tests."""

    config_overrides = {}

    def setUp(self):
        config = dict(DEFAULT_CONFIG)
        config.update(self.config_overrides)
        patcher = mock.patch.dict(tracker_mod.CONFIG, config, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        # The guards announce themselves through the logger; silence it so test
        # output stays readable.
        log_patcher = mock.patch.object(tracker_mod, "logger", mock.MagicMock())
        log_patcher.start()
        self.addCleanup(log_patcher.stop)

    def make_bot(self):
        return tracker_mod.UnifiedPolymarketBot()


class EnvironmentParsingTests(TrackerTestCase):
    def test_reads_first_name_that_is_set(self):
        with mock.patch.dict(os.environ, {"MIN_ARBITRAGE_EDGE_PCT": "0.09"}, clear=True):
            value = tracker_mod.env_value(["MIN_NET_PROFIT_MARGIN", "MIN_ARBITRAGE_EDGE_PCT"], 0.05, float)
        self.assertAlmostEqual(value, 0.09)

    def test_canonical_name_wins_over_alias(self):
        env = {"MIN_NET_PROFIT_MARGIN": "0.12", "MIN_ARBITRAGE_EDGE_PCT": "0.09"}
        with mock.patch.dict(os.environ, env, clear=True):
            value = tracker_mod.env_value(["MIN_NET_PROFIT_MARGIN", "MIN_ARBITRAGE_EDGE_PCT"], 0.05, float)
        self.assertAlmostEqual(value, 0.12)

    def test_unparseable_value_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"MIN_NET_PROFIT_MARGIN": "not-a-number"}, clear=True):
            value = tracker_mod.env_value("MIN_NET_PROFIT_MARGIN", 0.05, float)
        self.assertAlmostEqual(value, 0.05)

    def test_blank_value_is_treated_as_unset(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "   "}, clear=True):
            self.assertEqual(tracker_mod.env_value("HTTP_PROXY", "fallback"), "fallback")


class FrictionGuardTests(TrackerTestCase):
    """The guard that exists to prevent a high win rate from still losing money [1, 9]."""

    def setUp(self):
        super().setUp()
        self.portfolio = tracker_mod.FrictionAwarePortfolioEngine(1000.0)

    def test_blocks_near_certain_contract_with_no_room_for_fees(self):
        # 0.97 leaves 3 cents of upside; fees and slippage cost more than that.
        eligible, quote = self.portfolio.evaluate_roi_eligibility(0.97, 0.95)
        self.assertFalse(eligible)
        self.assertLess(quote["spec_net_edge"], tracker_mod.CONFIG["MIN_NET_PROFIT_MARGIN"])

    def test_allows_a_genuine_latency_gap(self):
        # Buying at 0.45 something the model prices at 0.76 clears friction easily.
        eligible, quote = self.portfolio.evaluate_roi_eligibility(0.45, 0.76)
        self.assertTrue(eligible)
        self.assertGreater(quote["binding_edge"], tracker_mod.CONFIG["MIN_NET_PROFIT_MARGIN"])

    def test_blocks_wide_spread_when_probability_does_not_support_it(self):
        # A cheap contract looks great on the win-case formula alone, but a 20%
        # chance of paying out makes it negative expected value. The binding edge
        # has to reflect that, otherwise the guard waves through lottery tickets.
        eligible, quote = self.portfolio.evaluate_roi_eligibility(0.30, 0.20)
        self.assertFalse(eligible)
        self.assertGreater(quote["spec_net_edge"], 0.0)
        self.assertLess(quote["expected_net_edge"], 0.0)

    def test_settlement_fee_is_charged_separately_from_the_entry_fee(self):
        quote = self.portfolio.quote_friction(0.50, 1.0)
        self.assertAlmostEqual(quote["settlement_fee"], 0.015)
        self.assertAlmostEqual(quote["market_fee"], quote["entry_price"] * 0.015)
        self.assertAlmostEqual(
            quote["total_friction"],
            quote["slippage_cost"] + quote["market_fee"] + quote["settlement_fee"],
        )

    def test_raising_the_fee_can_flip_an_acceptable_trade_to_blocked(self):
        marginal_price, marginal_prob = 0.45, 0.56
        self.assertTrue(self.portfolio.evaluate_roi_eligibility(marginal_price, marginal_prob)[0])

        with mock.patch.dict(tracker_mod.CONFIG, {"SIMULATED_FEE_PCT": 0.20, "SETTLEMENT_FEE_PCT": 0.20}):
            expensive = tracker_mod.FrictionAwarePortfolioEngine(1000.0)
            self.assertFalse(expensive.evaluate_roi_eligibility(marginal_price, marginal_prob)[0])


class LimitPriceGuardTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.portfolio = tracker_mod.FrictionAwarePortfolioEngine(1000.0)

    def test_allows_a_fill_at_or_below_the_ceiling(self):
        self.assertTrue(self.portfolio.enforce_limit_price(0.45, 0.4522, "Market", "Yes"))
        self.assertTrue(self.portfolio.enforce_limit_price(0.4522, 0.4522, "Market", "Yes"))

    def test_rejects_a_fill_above_the_ceiling(self):
        self.assertFalse(self.portfolio.enforce_limit_price(0.78, 0.4522, "Market", "Yes"))

    def test_book_that_reprices_before_execution_cancels_the_buy(self):
        bot = self.make_bot()
        bot.market_names["m1"] = "Test Market"
        bot.token_prices["m1"]["Yes"] = 0.45

        # The book jumps to the post-goal price the instant before we execute.
        def jump_the_book(*_args, **_kwargs):
            bot.token_prices["m1"]["Yes"] = 0.80
            return {"error": "no key"}

        with mock.patch.object(bot.order_signer, "sign_limit_order", side_effect=jump_the_book):
            with mock.patch.object(bot.portfolio, "execute_buy") as execute_buy:
                bot._dispatch_and_trade(
                    market_id="m1", market_title="Test Market", raw_outcome="Yes",
                    current_price=0.45, target_outcome="Yes",
                    signals={"sports_latency_arbitrage": {"msg": "goal"}},
                    confidence=95, target_probability=0.76,
                )
        execute_buy.assert_not_called()


class SoccerProbabilityTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.engine = tracker_mod.SportsLatencyArbitrageEngine()

    def test_a_late_lead_is_worth_much_more_than_an_early_one(self):
        early = self.engine.calculate_soccer_probability(1, 0, 10, team_focus="home")
        late = self.engine.calculate_soccer_probability(1, 0, 85, team_focus="home")
        self.assertGreater(late, early)
        self.assertGreater(late, 0.9)

    def test_full_time_collapses_to_a_certainty(self):
        self.assertEqual(self.engine.calculate_soccer_probability(2, 1, 90, team_focus="home"), 1.0)
        self.assertEqual(self.engine.calculate_soccer_probability(2, 1, 90, team_focus="away"), 0.0)
        self.assertEqual(self.engine.calculate_soccer_probability(1, 1, 90, team_focus="home"), 0.0)

    def test_probabilities_stay_inside_the_clamp(self):
        for minute in (1, 30, 60, 89):
            for score in ((0, 0), (3, 0), (0, 3)):
                prob = self.engine.calculate_soccer_probability(score[0], score[1], minute)
                self.assertGreaterEqual(prob, 0.01)
                self.assertLessEqual(prob, 0.99)

    def test_time_decay_weight_changes_how_safe_a_lead_looks(self):
        # More expected goals left to come means a one-goal lead is less secure.
        calm = tracker_mod.SportsLatencyArbitrageEngine(time_decay_weight=0.5)
        frantic = tracker_mod.SportsLatencyArbitrageEngine(time_decay_weight=2.0)
        self.assertGreater(
            calm.calculate_soccer_probability(1, 0, 60, team_focus="home"),
            frantic.calculate_soccer_probability(1, 0, 60, team_focus="home"),
        )

    def test_home_and_away_views_are_consistent(self):
        home = self.engine.calculate_soccer_probability(1, 0, 70, team_focus="home")
        away = self.engine.calculate_soccer_probability(1, 0, 70, team_focus="away")
        # Both cannot win, and a draw is possible, so they must sum to under 1.
        self.assertLess(home + away, 1.0)
        self.assertGreater(home, away)


class LatencyWindowTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.bot = self.make_bot()
        self.bot.update_book_quote("m1", "Will Newcastle beat Crystal Palace?", "Yes", 0.45, timestamp=1000.0)

    def _fire_goal(self, event_timestamp, now):
        with mock.patch.object(self.bot, "_dispatch_and_trade") as dispatch:
            self.bot.process_sports_event(
                market_id="m1", game_id="g1", team_home="Newcastle", team_away="Crystal Palace",
                score_home=1, score_away=0, minute=75,
                event_timestamp=event_timestamp, now=now,
            )
        return dispatch

    def test_fresh_goal_opens_a_trade(self):
        dispatch = self._fire_goal(event_timestamp=1010.0, now=1011.0)
        dispatch.assert_called_once()
        self.assertGreaterEqual(dispatch.call_args.kwargs["confidence"], 90)

    def test_goal_acted_on_too_late_is_ignored(self):
        # Past SPORTS_LATENCY_THRESHOLD_SECS the book has had time to reprice, so a
        # gap that still looks open is more likely to be our own stale data.
        dispatch = self._fire_goal(event_timestamp=1010.0, now=1010.0 + 30)
        dispatch.assert_not_called()

    def test_reported_lag_reflects_a_book_that_has_not_moved(self):
        # Regression test: the lag used to be computed against a timestamp assigned
        # one line earlier, so it was always zero no matter how stale the book was.
        dispatch = self._fire_goal(event_timestamp=1042.0, now=1043.0)
        message = dispatch.call_args.kwargs["signals"]["sports_latency_arbitrage"]["msg"]
        self.assertIn("42.0s before the goal", message)

    def test_confidence_decays_as_the_window_closes(self):
        fresh = self._fire_goal(event_timestamp=1010.0, now=1010.0)
        self.bot.sports_engine.match_states.clear()
        stale = self._fire_goal(event_timestamp=1020.0, now=1024.5)
        self.assertGreater(
            fresh.call_args.kwargs["confidence"],
            stale.call_args.kwargs["confidence"],
        )

    def test_no_signal_when_the_book_already_prices_the_goal_in(self):
        self.bot.update_book_quote("m1", "Will Newcastle beat Crystal Palace?", "Yes", 0.95, timestamp=1000.0)
        dispatch = self._fire_goal(event_timestamp=1010.0, now=1011.0)
        dispatch.assert_not_called()


class StateStopLossTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.bot = self.make_bot()
        self.bot.market_names["m1"] = "Will Wolves beat Fulham?"
        self.bot.token_prices["m1"]["Yes"] = 0.40
        self.bot.open_sports_theses["m1"] = {
            "game_id": "g1", "team_focus": "home", "outcome": "Yes",
            "goal_diff_at_entry": 1, "minute_at_entry": 55,
        }
        self.bot.portfolio.positions["m1"] = {"Yes": {"contracts": 200.0, "avg_price": 0.40}}

    def test_exits_when_the_lead_is_lost(self):
        fired = self.bot._check_state_stop_loss("m1", "g1", 1, 2, 78)
        self.assertTrue(fired)
        self.assertNotIn("m1", self.bot.portfolio.positions)
        self.assertNotIn("m1", self.bot.open_sports_theses)

    def test_a_level_score_also_invalidates_a_win_thesis(self):
        # The market is "will they win", not "will they avoid losing", so 1-1 is
        # already a dead thesis even though the team has not gone behind.
        self.assertTrue(self.bot._check_state_stop_loss("m1", "g1", 1, 1, 78))

    def test_holds_while_the_lead_survives(self):
        self.assertFalse(self.bot._check_state_stop_loss("m1", "g1", 2, 1, 78))
        self.assertIn("m1", self.bot.portfolio.positions)

    def test_ignores_score_updates_from_a_different_match(self):
        self.assertFalse(self.bot._check_state_stop_loss("m1", "some-other-game", 0, 3, 78))
        self.assertIn("m1", self.bot.portfolio.positions)

    def test_can_be_switched_off(self):
        with mock.patch.dict(tracker_mod.CONFIG, {"STOP_LOSS_ENABLED": False}):
            self.assertFalse(self.bot._check_state_stop_loss("m1", "g1", 1, 2, 78))
        self.assertIn("m1", self.bot.portfolio.positions)

    def test_away_thesis_is_the_mirror_image(self):
        self.bot.open_sports_theses["m1"]["team_focus"] = "away"
        self.assertFalse(self.bot._check_state_stop_loss("m1", "g1", 0, 1, 78))
        self.assertTrue(self.bot._check_state_stop_loss("m1", "g1", 1, 1, 80))


class PortfolioAccountingTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.portfolio = tracker_mod.FrictionAwarePortfolioEngine(1000.0)

    def test_selling_at_the_entry_price_still_loses_money_to_friction(self):
        self.portfolio.execute_buy("m1", "Market", "Yes", 0.50, 100.0)
        cash_after_buy = self.portfolio.cash
        self.portfolio.execute_sell("m1", "Market", "Yes", 0.50, reason="test")

        self.assertGreater(self.portfolio.cash, cash_after_buy)
        self.assertLess(self.portfolio.cash, 1000.0)
        self.assertEqual(self.portfolio.trade_history[-1]["win"], False)

    def test_selling_a_position_that_does_not_exist_is_a_no_op(self):
        self.assertFalse(self.portfolio.execute_sell("nope", "Market", "Yes", 0.50))

    def test_buy_is_refused_when_cash_is_short(self):
        self.assertFalse(self.portfolio.execute_buy("m1", "Market", "Yes", 0.50, 5000.0))
        self.assertEqual(self.portfolio.cash, 1000.0)

    def test_averaging_into_a_position_weights_the_entry_price(self):
        self.portfolio.execute_buy("m1", "Market", "Yes", 0.40, 100.0)
        self.portfolio.execute_buy("m1", "Market", "Yes", 0.60, 100.0)
        position = self.portfolio.positions["m1"]["Yes"]
        self.assertGreater(position["avg_price"], 0.40)
        self.assertLess(position["avg_price"], 0.60)


class TradeTelemetryTests(TrackerTestCase):
    def test_trade_rate_window_is_populated_and_pruned(self):
        # Regression test: nothing ever wrote to trade_timestamps, so the iceberg
        # strategy and the anti-consensus penalty could never trigger.
        bot = self.make_bot()
        for i in range(5):
            bot.process_live_trade({
                "market_id": "m1", "market_title": "Market", "outcome": "Yes",
                "price": 0.50, "size": 10, "wallet_address": f"0x{i}", "timestamp": 1000.0 + i,
            })
        self.assertEqual(len(bot.trade_timestamps["m1"]), 5)

        # A trade far in the future must evict everything older than the window.
        bot.process_live_trade({
            "market_id": "m1", "market_title": "Market", "outcome": "Yes",
            "price": 0.50, "size": 10, "wallet_address": "0xlate", "timestamp": 9999.0,
        })
        self.assertEqual(len(bot.trade_timestamps["m1"]), 1)

    def test_coordination_needs_enough_distinct_wallets(self):
        bot = self.make_bot()
        for i in range(6):
            self.assertFalse(bot._check_coordination("m1", f"0x{i}", 1000.0 + i))
        self.assertTrue(bot._check_coordination("m1", "0x6", 1006.0))

    def test_one_wallet_trading_repeatedly_is_not_coordination(self):
        bot = self.make_bot()
        for i in range(20):
            self.assertFalse(bot._check_coordination("m1", "0xsame", 1000.0 + i))

    def test_book_quote_refreshes_the_staleness_clock(self):
        bot = self.make_bot()
        bot.update_book_quote("m1", "Market", "Yes", 0.55, timestamp=1234.0)
        self.assertEqual(bot.last_book_update_ts["m1"], 1234.0)
        self.assertEqual(bot.token_prices["m1"]["Yes"], 0.55)


class MidPriceExtractionTests(TrackerTestCase):
    def test_prefers_the_midpoint_of_both_sides(self):
        event = {"bids": [{"price": "0.40"}, {"price": "0.42"}],
                 "asks": [{"price": "0.46"}, {"price": "0.50"}]}
        self.assertAlmostEqual(tracker_mod.extract_mid_price(event), 0.44)

    def test_falls_back_to_a_one_sided_book(self):
        self.assertAlmostEqual(tracker_mod.extract_mid_price({"bids": [{"price": "0.40"}]}), 0.40)
        self.assertAlmostEqual(tracker_mod.extract_mid_price({"asks": [{"price": "0.60"}]}), 0.60)

    def test_falls_back_to_a_flat_price_field(self):
        self.assertAlmostEqual(tracker_mod.extract_mid_price({"price": "0.33"}), 0.33)

    def test_returns_none_when_there_is_nothing_usable(self):
        self.assertIsNone(tracker_mod.extract_mid_price({}))
        self.assertIsNone(tracker_mod.extract_mid_price({"bids": [{"price": "abc"}]}))


class OrderSignerTests(TrackerTestCase):
    def test_refuses_to_sign_with_a_placeholder_key(self):
        signer = tracker_mod.PolymarketOrderSigner(private_key="your_wallet_private_key_here")
        self.assertIn("error", signer.sign_limit_order("t1", 0.45, 100, 0.4522))

    def test_signed_order_carries_the_price_ceiling(self):
        signer = tracker_mod.PolymarketOrderSigner(private_key="0xdeadbeef")
        order = signer.sign_limit_order("t1", 0.45, 100, 0.4522)
        self.assertEqual(order["max_allowed_fill_price"], "0.4522")
        self.assertEqual(order["side"], "BUY")
        self.assertGreater(order["expiration"], order["nonce"] / 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
