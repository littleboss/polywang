#!/usr/bin/env python3
"""
Tests for the edge evaluation module.

The fee tests check against the published fee tables rather than against the
implementation, so if the formula is ever mistranscribed they will catch it.

Run with:  uv run python -m unittest -v tests.test_polymarket_edge
"""

import unittest

from polywang.polymarket_edge import (
    CalibrationTracker,
    EdgeEvaluator,
    NegRiskScanner,
    PolymarketFeeModel,
    StrategyCalibration,
    combo_arb_universe_score,
    merge_gas_clears_at_order_usd,
    merge_gas_startup_warning,
    rank_combo_arb_markets,
    debias_market_price,
    liquidity_adjusted_lambda,
    walk_order_book,
)


class FeeModelTests(unittest.TestCase):
    """Checked against the tables at https://docs.polymarket.com/trading/fees"""

    def test_sports_fees_match_the_published_table(self):
        sports = PolymarketFeeModel("sports")
        published = {0.01: 0.05, 0.10: 0.45, 0.25: 0.94, 0.50: 1.25,
                     0.75: 0.94, 0.90: 0.45, 0.95: 0.24, 0.99: 0.05}
        for price, expected in published.items():
            self.assertAlmostEqual(sports.fee_usd(100, price), expected, places=2,
                                   msg=f"sports fee wrong at price {price}")

    def test_crypto_fees_match_the_published_table(self):
        crypto = PolymarketFeeModel("crypto")
        published = {0.01: 0.07, 0.10: 0.63, 0.50: 1.75, 0.90: 0.63, 0.99: 0.07}
        for price, expected in published.items():
            self.assertAlmostEqual(crypto.fee_usd(100, price), expected, places=2)

    def test_politics_fees_match_the_published_table(self):
        politics = PolymarketFeeModel("politics")
        published = {0.10: 0.36, 0.50: 1.00, 0.90: 0.36, 0.99: 0.04}
        for price, expected in published.items():
            self.assertAlmostEqual(politics.fee_usd(100, price), expected, places=2)

    def test_geopolitics_is_free(self):
        self.assertEqual(PolymarketFeeModel("geopolitics").fee_usd(1000, 0.50), 0.0)

    def test_explicit_market_fee_parameters_override_category_defaults(self):
        model = PolymarketFeeModel("sports", taker_fee_rate=0.02, fee_exponent=2.0)
        self.assertAlmostEqual(model.fee_usd(100, 0.50), 0.125, places=9)

    def test_makers_are_never_charged(self):
        for category in ("sports", "crypto", "politics"):
            model = PolymarketFeeModel(category)
            self.assertEqual(model.fee_usd(1000, 0.50, is_taker=False), 0.0)

    def test_fee_peaks_at_the_midpoint_and_vanishes_at_the_extremes(self):
        sports = PolymarketFeeModel("sports")
        mid = sports.fee_usd(100, 0.50)
        for price in (0.05, 0.20, 0.80, 0.95):
            self.assertLess(sports.fee_usd(100, price), mid)

    def test_fee_is_symmetric_about_the_midpoint(self):
        sports = PolymarketFeeModel("sports")
        for price in (0.10, 0.25, 0.40):
            self.assertAlmostEqual(sports.fee_usd(100, price),
                                   sports.fee_usd(100, 1.0 - price), places=6)

    def test_fraction_of_notional_reduces_to_rate_times_one_minus_price(self):
        # This identity is what makes favourites cheap to trade and the flat
        # percentage model wrong.
        sports = PolymarketFeeModel("sports")
        for price in (0.10, 0.50, 0.90, 0.97):
            self.assertAlmostEqual(sports.fee_as_fraction_of_notional(price),
                                   0.05 * (1.0 - price), places=9)

    def test_flat_model_overstates_favourite_costs_by_an_order_of_magnitude(self):
        sports = PolymarketFeeModel("sports")
        self.assertAlmostEqual(0.015 / sports.fee_as_fraction_of_notional(0.97), 10.0, places=1)
        # And understates the cost of longshots.
        self.assertLess(0.015, sports.fee_as_fraction_of_notional(0.10))

    def test_unknown_category_falls_back_to_the_general_rate(self):
        self.assertEqual(PolymarketFeeModel("nonsense").taker_fee_rate,
                         PolymarketFeeModel("other").taker_fee_rate)

    def test_dust_fees_round_to_zero(self):
        self.assertEqual(PolymarketFeeModel("sports").fee_usd(0.0001, 0.99), 0.0)

    def test_large_fill_does_not_lose_a_small_per_share_fee(self):
        sports = PolymarketFeeModel("sports")
        self.assertGreater(sports.fee_per_share(0.9999), 0.0)
        self.assertGreater(sports.fee_usd(10_000, 0.9999), 0.0)


class BreakevenTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EdgeEvaluator(fee_model=PolymarketFeeModel("sports"))

    def test_breakeven_sits_just_above_the_price_for_a_taker(self):
        breakeven = self.evaluator.breakeven_probability(0.97)
        self.assertGreater(breakeven, 0.97)
        self.assertAlmostEqual(breakeven, 0.97 + 0.05 * 0.97 * 0.03, places=6)

    def test_breakeven_equals_the_price_for_a_maker(self):
        self.assertAlmostEqual(self.evaluator.breakeven_probability(0.97, is_taker=False), 0.97)

    def test_a_favourite_needs_almost_no_extra_accuracy_to_break_even(self):
        # The whole cost of being a favourite is the payoff shape, not the fee.
        self.assertLess(self.evaluator.breakeven_probability(0.97) - 0.97, 0.002)


class TimeHorizonTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EdgeEvaluator(
            fee_model=PolymarketFeeModel("sports"),
            min_ev_per_dollar=0.02, hurdle_apr=0.15, min_edge_over_breakeven=0.02,
        )

    def test_identical_edges_are_judged_differently_by_horizon(self):
        soon = self.evaluator.assess(0.97, 0.995, bankroll=10000, days_to_resolution=2)
        later = self.evaluator.assess(0.97, 0.995, bankroll=10000, days_to_resolution=365)

        self.assertAlmostEqual(soon.ev_per_dollar, later.ev_per_dollar, places=9)
        self.assertTrue(soon.accepted)
        self.assertFalse(later.accepted)

    def test_annualised_return_is_capped_at_something_meaningful(self):
        assessment = self.evaluator.assess(0.97, 0.995, bankroll=10000, days_to_resolution=0.01)
        self.assertLessEqual(assessment.annualised_return, 100.0)

    def test_required_return_grows_with_the_holding_period(self):
        short = self.evaluator.assess(0.50, 0.60, bankroll=10000, days_to_resolution=1)
        long = self.evaluator.assess(0.50, 0.60, bankroll=10000, days_to_resolution=365)
        self.assertLess(short.required_period_return, long.required_period_return)
        self.assertAlmostEqual(long.required_period_return, 0.15, places=6)


class HurdleTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EdgeEvaluator(
            fee_model=PolymarketFeeModel("sports"),
            min_ev_per_dollar=0.02, hurdle_apr=0.15, min_edge_over_breakeven=0.02,
        )

    def test_accepts_a_real_latency_gap(self):
        assessment = self.evaluator.assess(0.45, 0.78, bankroll=10000, days_to_resolution=0.1)
        self.assertTrue(assessment.accepted)
        self.assertGreater(assessment.recommended_stake_usd, 0)

    def test_rejects_an_edge_too_fine_for_the_model_to_resolve(self):
        # Claiming 0.975 against a price of 0.97 is claiming accuracy to a third
        # of a percentage point.
        assessment = self.evaluator.assess(0.97, 0.975, bankroll=10000, days_to_resolution=1)
        self.assertFalse(assessment.accepted)
        self.assertTrue(any("break-even" in reason for reason in assessment.reasons))

    def test_rejects_a_trade_priced_above_fair_value(self):
        assessment = self.evaluator.assess(0.60, 0.55, bankroll=10000, days_to_resolution=1)
        self.assertFalse(assessment.accepted)
        self.assertLess(assessment.ev_per_share, 0)
        self.assertEqual(assessment.recommended_stake_usd, 0.0)

    def test_geopolitics_being_fee_free_can_change_the_verdict(self):
        cheap = EdgeEvaluator(fee_model=PolymarketFeeModel("geopolitics"),
                              min_ev_per_dollar=0.02, hurdle_apr=0.15,
                              min_edge_over_breakeven=0.02)
        expensive = EdgeEvaluator(fee_model=PolymarketFeeModel("crypto"),
                                  min_ev_per_dollar=0.02, hurdle_apr=0.15,
                                  min_edge_over_breakeven=0.02)
        price, fair = 0.50, 0.5320
        self.assertTrue(cheap.assess(price, fair, 10000, days_to_resolution=1).accepted)
        self.assertFalse(expensive.assess(price, fair, 10000, days_to_resolution=1).accepted)

    def test_exiting_early_costs_a_second_fee(self):
        hold = self.evaluator.assess(0.50, 0.60, 10000, days_to_resolution=1)
        exit_early = self.evaluator.assess(0.50, 0.60, 10000, days_to_resolution=1,
                                           exit_before_resolution=True)
        self.assertGreater(exit_early.fee_per_share, hold.fee_per_share)
        self.assertLess(exit_early.ev_per_share, hold.ev_per_share)

    def test_warns_about_asymmetric_payoff_on_favourites(self):
        assessment = self.evaluator.assess(0.97, 0.999, bankroll=10000, days_to_resolution=1)
        self.assertTrue(any("asymmetric" in warning for warning in assessment.warnings))

    def test_warns_when_a_longshot_call_contradicts_the_documented_bias(self):
        assessment = self.evaluator.assess(0.10, 0.16, bankroll=10000, days_to_resolution=10)
        self.assertTrue(any("longshot" in warning for warning in assessment.warnings))


class KellySizingTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EdgeEvaluator(fee_model=PolymarketFeeModel("sports"),
                                       kelly_fraction=0.25, max_position_fraction=0.10)

    def test_bigger_edges_get_more_capital(self):
        small = self.evaluator.assess(0.45, 0.52, bankroll=10000, days_to_resolution=1)
        large = self.evaluator.assess(0.45, 0.78, bankroll=10000, days_to_resolution=1)
        self.assertGreater(large.recommended_stake_usd, small.recommended_stake_usd)

    def test_no_edge_means_no_stake(self):
        assessment = self.evaluator.assess(0.45, 0.45, bankroll=10000, days_to_resolution=1)
        self.assertEqual(assessment.recommended_stake_usd, 0.0)

    def test_position_cap_is_respected(self):
        assessment = self.evaluator.assess(0.10, 0.90, bankroll=10000, days_to_resolution=1)
        self.assertLessEqual(assessment.recommended_stake_usd, 10000 * 0.10 + 1e-9)

    def test_lower_confidence_shrinks_the_stake(self):
        confident = self.evaluator.assess(0.45, 0.78, 10000, days_to_resolution=1, confidence=1.0)
        unsure = self.evaluator.assess(0.45, 0.78, 10000, days_to_resolution=1, confidence=0.2)
        self.assertGreater(confident.recommended_stake_usd, unsure.recommended_stake_usd)

    def test_stake_scales_with_bankroll(self):
        small = self.evaluator.assess(0.45, 0.78, bankroll=1000, days_to_resolution=1)
        large = self.evaluator.assess(0.45, 0.78, bankroll=10000, days_to_resolution=1)
        self.assertAlmostEqual(large.recommended_stake_usd, small.recommended_stake_usd * 10, places=6)


class MakerRoutingTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = EdgeEvaluator(fee_model=PolymarketFeeModel("sports"))

    def test_crosses_the_spread_only_for_a_fast_decaying_edge(self):
        self.assertFalse(self.evaluator.should_post_as_maker(edge_decay_seconds=5))
        self.assertTrue(self.evaluator.should_post_as_maker(edge_decay_seconds=600))

    def test_a_structural_edge_with_no_decay_is_worked_patiently(self):
        self.assertTrue(self.evaluator.should_post_as_maker(edge_decay_seconds=None))


class OrderBookWalkTests(unittest.TestCase):
    def setUp(self):
        self.book = [(0.45, 200), (0.46, 300), (0.48, 500), (0.52, 1000)]

    def test_small_order_fills_at_the_touch(self):
        fill = walk_order_book(self.book, usd_budget=50)
        self.assertAlmostEqual(fill.average_price, 0.45, places=6)
        self.assertTrue(fill.is_complete)

    def test_large_order_pays_progressively_worse_prices(self):
        small = walk_order_book(self.book, usd_budget=50)
        large = walk_order_book(self.book, usd_budget=800)
        self.assertGreater(large.average_price, small.average_price)
        self.assertGreater(large.worst_price, small.worst_price)

    def test_slippage_can_exceed_a_flat_assumption_many_times_over(self):
        fill = walk_order_book(self.book, usd_budget=800)
        realised = fill.average_price / 0.45 - 1.0
        self.assertGreater(realised, 0.005 * 5)

    def test_price_ceiling_stops_the_walk(self):
        fill = walk_order_book(self.book, usd_budget=10000, max_price=0.46)
        self.assertLessEqual(fill.worst_price, 0.46)
        self.assertFalse(fill.is_complete)
        self.assertGreater(fill.budget_unfilled, 0)

    def test_empty_book_fills_nothing(self):
        fill = walk_order_book([], usd_budget=100)
        self.assertEqual(fill.shares, 0.0)
        self.assertEqual(fill.budget_unfilled, 100)

    def test_levels_are_consumed_best_price_first(self):
        shuffled = [(0.52, 1000), (0.45, 200), (0.48, 500), (0.46, 300)]
        self.assertAlmostEqual(walk_order_book(shuffled, 50).average_price, 0.45, places=6)


class NegRiskTests(unittest.TestCase):
    def setUp(self):
        self.scanner = NegRiskScanner(PolymarketFeeModel("politics"), min_net_margin=0.01)

    def test_under_summed_field_is_a_buy_all_yes(self):
        prices = {"a": 0.30, "b": 0.25, "c": 0.20, "d": 0.15}
        opportunity = self.scanner.scan("m", prices)
        self.assertEqual(opportunity.direction, "BUY_ALL_YES")
        self.assertAlmostEqual(opportunity.gross_margin, 0.10, places=6)
        self.assertTrue(opportunity.tradeable)

    def test_over_summed_field_is_a_buy_all_no(self):
        prices = {"a": 0.40, "b": 0.35, "c": 0.30, "d": 0.15}
        self.assertEqual(self.scanner.scan("m", prices).direction, "BUY_ALL_NO")

    def test_fairly_priced_field_offers_nothing(self):
        opportunity = self.scanner.scan("m", {"a": 0.60, "b": 0.40})
        self.assertFalse(opportunity.tradeable)
        self.assertAlmostEqual(opportunity.gross_margin, 0.0, places=9)

    def test_leg_count_can_make_a_real_dislocation_untradeable_for_a_taker(self):
        wide_field = {f"c{i}": 0.09 for i in range(10)}   # sums to 0.90
        as_taker = self.scanner.scan("m", wide_field, is_taker=True)
        as_maker = self.scanner.scan("m", wide_field, is_taker=False)
        self.assertGreater(as_maker.net_margin, as_taker.net_margin)
        self.assertAlmostEqual(as_maker.net_margin, as_maker.gross_margin, places=9)

    def test_single_outcome_is_not_a_negrisk_market(self):
        self.assertIsNone(self.scanner.scan("m", {"a": 0.5}))

    def test_gas_costs_reduce_the_margin(self):
        prices = {"a": 0.30, "b": 0.25, "c": 0.20, "d": 0.15}
        cheap = NegRiskScanner(PolymarketFeeModel("politics"), gas_cost_usd=0.0)
        pricey = NegRiskScanner(PolymarketFeeModel("politics"), gas_cost_usd=50.0)
        self.assertGreater(cheap.scan("m", prices, set_size_usd=100).net_margin,
                           pricey.scan("m", prices, set_size_usd=100).net_margin)


class ComboUniverseRankTests(unittest.TestCase):
    def test_politics_mid_needs_three_ticks_extreme_needs_one(self):
        mid = combo_arb_universe_score("politics", 0.50)
        extreme = combo_arb_universe_score("politics", 0.95)
        geo = combo_arb_universe_score("geopolitics", 0.50)
        self.assertEqual(mid.ticks_to_breakeven, 3)
        self.assertEqual(extreme.ticks_to_breakeven, 1)
        self.assertEqual(geo.ticks_to_breakeven, 1)
        self.assertLess(extreme.fee_per_set, mid.fee_per_set)
        self.assertAlmostEqual(geo.fee_per_set, 0.0, places=9)

    def test_unknown_price_is_scored_at_the_worst_case_mid(self):
        unknown = combo_arb_universe_score("politics", None)
        mid = combo_arb_universe_score("politics", 0.50)
        self.assertFalse(unknown.price_known)
        self.assertEqual(unknown.ticks_to_breakeven, mid.ticks_to_breakeven)

    def test_rank_puts_zero_fee_and_extremes_ahead_of_mid_politics(self):
        from polywang.arbitrage_core import BinaryMarket
        mid = BinaryMarket("mid", "c", "Mid", "y1", "n1", category="politics", implied_yes=0.50)
        extreme = BinaryMarket("ext", "c", "Ext", "y2", "n2", category="politics", implied_yes=0.95)
        geo = BinaryMarket("geo", "c", "Geo", "y3", "n3", category="geopolitics", implied_yes=0.50)
        ranked = rank_combo_arb_markets([mid, extreme, geo])
        self.assertEqual([market.market_id for market in ranked], ["geo", "ext", "mid"])

    def test_merge_gas_warning_flags_small_orders_and_hidden_gas(self):
        self.assertFalse(merge_gas_clears_at_order_usd(5.0, 0.30))
        self.assertTrue(merge_gas_clears_at_order_usd(100.0, 0.30))
        self.assertIn("treat chain merge as free", merge_gas_startup_warning(0.0, 5.0, True))
        self.assertIn("cannot clear gas", merge_gas_startup_warning(0.30, 5.0, False))
        self.assertIsNone(merge_gas_startup_warning(0.0, 5.0, False))


class CalibrationTests(unittest.TestCase):
    def test_perfect_forecasts_score_zero(self):
        record = StrategyCalibration("perfect")
        for _ in range(10):
            record.record(1.0, 1)
            record.record(0.0, 0)
        self.assertAlmostEqual(record.brier_score, 0.0, places=9)

    def test_always_saying_fifty_percent_scores_a_quarter(self):
        record = StrategyCalibration("uninformative")
        for i in range(100):
            record.record(0.5, i % 2)
        self.assertAlmostEqual(record.brier_score, 0.25, places=9)

    def test_confidently_wrong_scores_worse_than_uninformative(self):
        record = StrategyCalibration("backwards")
        for _ in range(50):
            record.record(0.9, 0)
        self.assertGreater(record.brier_score, 0.25)

    def test_bias_detects_systematic_overconfidence(self):
        record = StrategyCalibration("overconfident")
        for i in range(100):
            record.record(0.9, 1 if i % 2 == 0 else 0)
        self.assertAlmostEqual(record.bias, 0.4, places=6)

    def test_new_strategies_are_trusted_until_there_is_evidence(self):
        tracker = CalibrationTracker(min_samples=20)
        for _ in range(5):
            tracker.record("brand_new", 0.9, 0)
        self.assertTrue(tracker.is_trustworthy("brand_new"))
        self.assertTrue(tracker.is_trustworthy("never_seen"))

    def test_a_strategy_is_distrusted_once_the_evidence_is_in(self):
        tracker = CalibrationTracker(min_samples=20)
        for _ in range(30):
            tracker.record("backwards", 0.9, 0)
        self.assertFalse(tracker.is_trustworthy("backwards"))
        self.assertIn("backwards", tracker.underperformers())

    def test_reliability_table_buckets_forecasts(self):
        record = StrategyCalibration("mixed")
        for _ in range(10):
            record.record(0.1, 0)
            record.record(0.9, 1)
        rows = record.reliability_table(buckets=5)
        self.assertEqual(len(rows), 2)
        low, high = rows[0], rows[-1]
        self.assertAlmostEqual(low[3], 0.0, places=6)
        self.assertAlmostEqual(high[3], 1.0, places=6)

    def test_report_names_the_underperformers(self):
        tracker = CalibrationTracker(min_samples=10)
        for _ in range(20):
            tracker.record("bad", 0.95, 0)
        self.assertIn("COIN FLIP", tracker.report())

    def test_walk_forward_and_drift_gate_live_readiness(self):
        tracker = CalibrationTracker(min_samples=4)
        for _ in range(8):
            tracker.record("good", 0.9, 1)
            tracker.record("good", 0.1, 0)
        self.assertTrue(tracker.is_live_ready("good"))
        self.assertIsNotNone(tracker.strategies["good"].walk_forward_brier())
        self.assertLess(tracker.strategies["good"].walk_forward_brier(), 0.25)
        params = tracker.recommended_parameters("good")
        self.assertTrue(params["live_ready"])
        self.assertGreaterEqual(params["min_edge_over_breakeven"], 0.02)

        drifted = CalibrationTracker(min_samples=4)
        for _ in range(20):
            drifted.record("late", 0.9, 1)
        for _ in range(20):
            drifted.record("late", 0.9, 0)
        self.assertTrue(drifted.has_drifted("late", window=20, threshold=0.08))
        self.assertFalse(drifted.is_live_ready("late"))

    def test_wilson_interval_contains_the_hit_rate(self):
        record = StrategyCalibration("mixed")
        for i in range(40):
            record.record(0.7, 1 if i < 28 else 0)
        low, high = record.hit_rate_interval()
        self.assertLessEqual(low, record.hit_rate)
        self.assertGreaterEqual(high, record.hit_rate)


class WangTransformTests(unittest.TestCase):
    def test_debiasing_lowers_the_yes_estimate(self):
        # A positive wedge means the market price sits above physical probability.
        for price in (0.10, 0.30, 0.50, 0.80, 0.95):
            self.assertLess(debias_market_price(price, 0.176), price)

    def test_longshots_are_proportionally_more_inflated_than_favourites(self):
        longshot_ratio = debias_market_price(0.05, 0.176) / 0.05
        favourite_ratio = debias_market_price(0.95, 0.176) / 0.95
        self.assertLess(longshot_ratio, favourite_ratio)

    def test_zero_wedge_is_the_identity(self):
        for price in (0.05, 0.42, 0.91):
            self.assertAlmostEqual(debias_market_price(price, 0.0), price, places=6)

    def test_wedge_disappears_in_liquid_markets(self):
        self.assertAlmostEqual(liquidity_adjusted_lambda(50000), 0.0)
        self.assertGreater(liquidity_adjusted_lambda(100), 0.0)
        self.assertGreater(liquidity_adjusted_lambda(100), liquidity_adjusted_lambda(5000))

    def test_output_stays_a_probability(self):
        for price in (0.001, 0.01, 0.5, 0.99, 0.999):
            value = debias_market_price(price, 0.176)
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
