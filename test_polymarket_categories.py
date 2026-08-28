#!/usr/bin/env python3
"""
Tests for the US macro and crypto market support.

Run with:  python3 -m unittest -v test_polymarket_categories
"""

import math
import unittest

from polymarket_edge import (
    CATEGORY_TAKER_FEE_RATES,
    EdgeEvaluator,
    NegRiskScanner,
    PolymarketFeeModel,
    fee_rate_from_bps,
    resolve_category,
)
from polymarket_crypto import (
    CryptoLatencyEngine,
    RealisedVolatility,
    probability_above,
    probability_band,
    probability_below,
    probability_in_range,
    round_trip_cost_fraction,
)
from polymarket_macro import (
    ScheduledEventGuard,
    SecondaryRepricingWatcher,
    fed_rate_ladder_outcomes,
)


class CategoryResolutionTests(unittest.TestCase):
    def test_us_macro_names_resolve_to_economics(self):
        for name in ("us_macro", "macro", "CPI", "fomc", "NFP", "payrolls", "gdp", "inflation"):
            self.assertEqual(resolve_category(name), "economics", msg=f"{name} misrouted")

    def test_coin_names_resolve_to_crypto(self):
        for name in ("btc", "BTC", "bitcoin", "eth", "Ethereum", "sol"):
            self.assertEqual(resolve_category(name), "crypto", msg=f"{name} misrouted")

    def test_canonical_names_pass_through(self):
        for name in CATEGORY_TAKER_FEE_RATES:
            self.assertEqual(resolve_category(name), name)

    def test_unknown_names_fall_back_rather_than_crash(self):
        self.assertEqual(resolve_category("something novel"), "other")
        self.assertEqual(resolve_category(None), "other")

    def test_macro_and_crypto_get_the_published_rates(self):
        self.assertAlmostEqual(PolymarketFeeModel("us_macro").taker_fee_rate, 0.05)
        self.assertAlmostEqual(PolymarketFeeModel("btc").taker_fee_rate, 0.07)

    def test_crypto_is_the_most_expensive_category(self):
        crypto = PolymarketFeeModel("crypto").taker_fee_rate
        for category, rate in CATEGORY_TAKER_FEE_RATES.items():
            self.assertLessEqual(rate, crypto, msg=f"{category} exceeds crypto")


class PerMarketFeeScheduleTests(unittest.TestCase):
    """The rate is a market parameter, not a constant derived from the category."""

    def test_reads_the_live_schedule_when_present(self):
        market = {
            "category": "crypto",
            "trading": {
                "feesEnabled": True,
                "feeSchedule": {"rate": 0.04, "exponent": 1, "takerOnly": True, "rebateRate": 0.25},
            },
        }
        model = PolymarketFeeModel.from_market_data(market)
        # The live rate wins over the 0.07 the category table would have supplied.
        self.assertAlmostEqual(model.taker_fee_rate, 0.04)
        self.assertAlmostEqual(model.maker_rebate_share, 0.25)

    def test_a_fee_disabled_market_costs_nothing(self):
        market = {"category": "crypto", "trading": {"feesEnabled": False, "feeSchedule": None}}
        model = PolymarketFeeModel.from_market_data(market)
        self.assertEqual(model.fee_usd(1000, 0.50), 0.0)

    def test_falls_back_to_the_category_table_on_partial_data(self):
        model = PolymarketFeeModel.from_market_data({"category": "bitcoin"})
        self.assertAlmostEqual(model.taker_fee_rate, 0.07)
        self.assertGreater(model.fee_usd(100, 0.50), 0.0)

    def test_price_exponent_is_honoured(self):
        linear = PolymarketFeeModel("sports", price_exponent=1.0)
        squared = PolymarketFeeModel("sports", price_exponent=2.0)
        # p(1-p) at 0.5 is 0.25, and squaring it makes the fee smaller.
        self.assertAlmostEqual(squared.fee_usd(100, 0.50), linear.fee_usd(100, 0.50) * 0.25, places=6)

    def test_bps_conversion_matches_the_published_rates(self):
        self.assertAlmostEqual(fee_rate_from_bps(700), 0.07)
        self.assertAlmostEqual(fee_rate_from_bps(500), 0.05)
        self.assertAlmostEqual(fee_rate_from_bps(400), 0.04)

    def test_maker_collects_a_rebate_rather_than_paying(self):
        model = PolymarketFeeModel("crypto")
        self.assertEqual(model.fee_usd(100, 0.50, is_taker=False), 0.0)
        self.assertGreater(model.maker_rebate_usd(100, 0.50), 0.0)
        self.assertAlmostEqual(model.maker_rebate_usd(100, 0.50),
                               model.fee_usd(100, 0.50) * 0.20, places=9)


class CryptoCostTests(unittest.TestCase):
    """The 0.07 rate is what forces the strategy change, so pin the numbers down."""

    def setUp(self):
        self.crypto = PolymarketFeeModel("crypto")

    def test_mid_priced_crypto_costs_three_and_a_half_percent_to_enter(self):
        self.assertAlmostEqual(self.crypto.fee_as_fraction_of_notional(0.50), 0.035, places=6)

    def test_a_round_trip_at_the_midpoint_costs_seven_percent(self):
        self.assertAlmostEqual(round_trip_cost_fraction(0.07, 0.50), 0.07, places=6)

    def test_crypto_favourites_are_still_cheap(self):
        self.assertLess(self.crypto.fee_as_fraction_of_notional(0.95), 0.004)

    def test_crypto_costs_forty_percent_more_than_macro_at_any_price(self):
        macro = PolymarketFeeModel("us_macro")
        for price in (0.10, 0.30, 0.50, 0.70, 0.90):
            ratio = self.crypto.fee_as_fraction_of_notional(price) / macro.fee_as_fraction_of_notional(price)
            self.assertAlmostEqual(ratio, 0.07 / 0.05, places=6)

    def test_the_same_trade_can_clear_in_sports_and_fail_in_crypto(self):
        # Break-even at a price of 0.50 is 0.5125 under the sports rate and
        # 0.5175 under crypto's, so a fair value inside that gap is a trade in one
        # category and not the other. The only thing that changed is the fee.
        price, fair = 0.50, 0.535
        sports = EdgeEvaluator(fee_model=PolymarketFeeModel("sports"),
                               min_ev_per_dollar=0.02, min_edge_over_breakeven=0.02)
        crypto = EdgeEvaluator(fee_model=self.crypto,
                               min_ev_per_dollar=0.02, min_edge_over_breakeven=0.02)

        self.assertAlmostEqual(sports.breakeven_probability(price), 0.5125, places=6)
        self.assertAlmostEqual(crypto.breakeven_probability(price), 0.5175, places=6)
        self.assertTrue(sports.assess(price, fair, 10000, days_to_resolution=1).accepted)
        self.assertFalse(crypto.assess(price, fair, 10000, days_to_resolution=1).accepted)

    def test_scalping_crypto_needs_a_seven_point_move_before_it_earns_anything(self):
        # Entering and exiting both cross the book, so the fee is paid twice.
        hold = EdgeEvaluator(fee_model=self.crypto).assess(
            0.50, 0.60, 10000, days_to_resolution=1)
        scalp = EdgeEvaluator(fee_model=self.crypto).assess(
            0.50, 0.60, 10000, days_to_resolution=1, exit_before_resolution=True)
        self.assertAlmostEqual(scalp.fee_per_share, hold.fee_per_share * 2, places=9)
        self.assertAlmostEqual(scalp.cost_per_share - 0.50, 0.035, places=6)


class DigitalOptionTests(unittest.TestCase):
    def test_deep_in_the_money_approaches_certainty(self):
        self.assertGreater(probability_above(150000, 100000, 0.60, 1.0), 0.99)

    def test_deep_out_of_the_money_approaches_zero(self):
        self.assertLess(probability_above(50000, 100000, 0.60, 1.0), 0.01)

    def test_at_the_money_is_near_a_coin_flip(self):
        probability = probability_above(100000, 100000, 0.60, 7.0)
        self.assertGreater(probability, 0.45)
        self.assertLess(probability, 0.50)

    def test_zero_drift_biases_slightly_below_half_at_the_money(self):
        # With expected price equal to today's price, the median outcome sits
        # below the mean, so an at-the-money digital is worth just under 0.50.
        self.assertLess(probability_above(100000, 100000, 0.60, 30.0), 0.50)

    def test_more_time_pulls_probability_towards_a_coin_flip(self):
        near = probability_above(110000, 100000, 0.60, 1.0)
        far = probability_above(110000, 100000, 0.60, 180.0)
        self.assertGreater(near, far)
        self.assertGreater(far, 0.40)

    def test_higher_volatility_helps_an_underdog_and_hurts_a_favourite(self):
        underdog_calm = probability_above(90000, 100000, 0.30, 30.0)
        underdog_wild = probability_above(90000, 100000, 1.20, 30.0)
        self.assertGreater(underdog_wild, underdog_calm)

        favourite_calm = probability_above(110000, 100000, 0.30, 30.0)
        favourite_wild = probability_above(110000, 100000, 1.20, 30.0)
        self.assertLess(favourite_wild, favourite_calm)

    def test_at_expiry_the_answer_is_settled(self):
        self.assertEqual(probability_above(101000, 100000, 0.60, 0.0), 1.0)
        self.assertEqual(probability_above(99000, 100000, 0.60, 0.0), 0.0)

    def test_above_and_below_are_complementary(self):
        for spot in (80000, 100000, 130000):
            total = (probability_above(spot, 100000, 0.60, 14.0)
                     + probability_below(spot, 100000, 0.60, 14.0))
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_range_probability_is_the_difference_of_two_digitals(self):
        inside = probability_in_range(100000, 95000, 105000, 0.60, 14.0)
        self.assertGreater(inside, 0.0)
        self.assertLess(inside, 1.0)
        self.assertEqual(probability_in_range(100000, 105000, 95000, 0.60, 14.0), 0.0)

    def test_output_is_always_a_probability(self):
        for spot in (1, 50000, 100000, 500000):
            for vol in (0.05, 0.6, 3.0):
                for days in (0.1, 7, 365):
                    value = probability_above(spot, 100000, vol, days)
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)


class VolatilityBandTests(unittest.TestCase):
    def test_band_brackets_the_point_estimate(self):
        band = probability_band(105000, 100000, 0.60, 14.0, volatility_error=0.25)
        self.assertLessEqual(band.low, band.point)
        self.assertLessEqual(band.point, band.high)
        self.assertGreater(band.width, 0.0)

    def test_band_contains_the_point_even_where_the_curve_turns(self):
        # Probability is not monotone in volatility: widening the distribution
        # pulls towards 0.50 while the drift term pushes down, so the extreme can
        # be interior. Sampling only the endpoints used to produce a band that
        # excluded its own point estimate.
        for days in (0.5, 2, 7, 14, 30, 60, 90, 180):
            for spot in (95000, 99000, 100000, 101000, 118000, 125000):
                band = probability_band(spot, 100000, 0.60, days, volatility_error=0.25)
                self.assertLessEqual(band.low, band.point + 1e-12,
                                     msg=f"low above point at {spot}/{days}d")
                self.assertGreaterEqual(band.high, band.point - 1e-12,
                                        msg=f"high below point at {spot}/{days}d")

    def test_no_volatility_uncertainty_means_no_band(self):
        band = probability_band(105000, 100000, 0.60, 14.0, volatility_error=0.0)
        self.assertAlmostEqual(band.width, 0.0, places=9)

    def test_confidence_is_zero_when_the_band_swamps_the_edge(self):
        # Near the money the volatility assumption moves the answer more than the
        # apparent edge does, so there is nothing to size on.
        band = probability_band(100000, 100000, 0.80, 30.0, volatility_error=0.40)
        self.assertEqual(band.confidence_against(band.point - 0.001), 0.0)

    def test_confidence_is_high_when_the_edge_dwarfs_the_band(self):
        band = probability_band(150000, 100000, 0.60, 3.0, volatility_error=0.25)
        self.assertGreater(band.confidence_against(0.50), 0.8)

    def test_a_wider_error_assumption_lowers_confidence(self):
        tight = probability_band(108000, 100000, 0.60, 14.0, volatility_error=0.10)
        loose = probability_band(108000, 100000, 0.60, 14.0, volatility_error=0.50)
        self.assertGreater(tight.confidence_against(0.50), loose.confidence_against(0.50))


class RealisedVolatilityTests(unittest.TestCase):
    def test_reports_nothing_until_there_are_enough_samples(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, minimum_samples=30)
        estimator.extend([100.0, 101.0, 100.5])
        self.assertFalse(estimator.is_ready)
        self.assertIsNone(estimator.annualised())

    def test_recovers_a_known_volatility(self):
        # Build a series with a fixed per-minute log return magnitude, so the
        # annualised figure is analytically predictable.
        import random
        random.seed(7)
        step = 0.001
        prices = [100.0]
        for _ in range(4000):
            prices.append(prices[-1] * math.exp(random.choice([step, -step])))

        estimator = RealisedVolatility(sample_interval_seconds=60, window=4000, minimum_samples=30)
        estimator.extend(prices)

        expected = step * math.sqrt(365 * 24 * 60)
        self.assertAlmostEqual(estimator.annualised(), expected, delta=expected * 0.05)

    def test_a_flat_series_has_no_volatility(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, minimum_samples=10)
        estimator.extend([100.0] * 50)
        self.assertAlmostEqual(estimator.annualised(), 0.0, places=9)

    def test_window_is_bounded(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, window=50)
        estimator.extend([100.0 + i for i in range(500)])
        self.assertLessEqual(estimator.sample_count, 50)

    def test_sampling_interval_changes_the_annualisation(self):
        prices = [100.0 * math.exp(0.001 * ((-1) ** i)) for i in range(200)]
        fast = RealisedVolatility(sample_interval_seconds=1, window=200, minimum_samples=10)
        slow = RealisedVolatility(sample_interval_seconds=3600, window=200, minimum_samples=10)
        fast.extend(prices)
        slow.extend(prices)
        self.assertGreater(fast.annualised(), slow.annualised())

    def test_rejects_a_nonsense_interval(self):
        with self.assertRaises(ValueError):
            RealisedVolatility(sample_interval_seconds=0)


class CryptoLatencyEngineTests(unittest.TestCase):
    def setUp(self):
        self.estimator = RealisedVolatility(sample_interval_seconds=60, window=500, minimum_samples=10)
        self.engine = CryptoLatencyEngine(volatility_estimator=self.estimator,
                                          min_spot_move_pct=0.002, min_edge=0.03)
        import random
        random.seed(11)
        price = 100000.0
        for _ in range(200):
            price *= math.exp(random.choice([0.0008, -0.0008]))
            self.engine.update_spot(price)
        self.engine.reset_reference()

    def test_no_signal_without_a_volatility_estimate(self):
        engine = CryptoLatencyEngine(volatility_estimator=RealisedVolatility(minimum_samples=100))
        engine.update_spot(100000.0)
        self.assertIsNone(engine.evaluate("m", 95000, 0.50, 7.0, seconds_since_book_moved=10))

    def test_a_sharp_spot_move_the_book_missed_produces_a_signal(self):
        self.engine.update_spot(self.engine._last_spot * 1.03)
        dislocation = self.engine.evaluate("btc-above-100k", strike=100000, market_price=0.40,
                                           days_to_expiry=2.0, seconds_since_book_moved=45.0)
        self.assertIsNotNone(dislocation)
        self.assertGreater(dislocation.edge, 0.03)
        self.assertIn("spot", dislocation.describe())

    def test_no_signal_when_the_book_already_agrees(self):
        self.engine.update_spot(self.engine._last_spot * 1.03)
        fair = self.engine.evaluate("m", 100000, 0.40, 2.0, seconds_since_book_moved=45.0).fair_probability
        self.assertIsNone(self.engine.evaluate("m", 100000, fair, 2.0, seconds_since_book_moved=45.0))

    def test_no_signal_when_spot_has_barely_moved(self):
        self.engine.update_spot(self.engine._last_spot * 1.0001)
        self.assertIsNone(self.engine.evaluate("m", 100000, 0.40, 2.0, seconds_since_book_moved=45.0))

    def test_resetting_the_reference_closes_the_signal(self):
        self.engine.update_spot(self.engine._last_spot * 1.03)
        self.assertIsNotNone(self.engine.evaluate("m", 100000, 0.40, 2.0, seconds_since_book_moved=45.0))
        self.engine.reset_reference()
        self.assertIsNone(self.engine.evaluate("m", 100000, 0.40, 2.0, seconds_since_book_moved=45.0))


class ScheduledEventGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = ScheduledEventGuard(blackout_seconds_before=120, blackout_seconds_after=300)
        self.release_at = 1_000_000.0
        self.guard.schedule("cpi", self.release_at)

    def test_trading_is_open_well_before_the_release(self):
        self.assertTrue(self.guard.is_tradeable(self.release_at - 600))

    def test_trading_is_suspended_approaching_the_release(self):
        status = self.guard.status(self.release_at - 60)
        self.assertTrue(status.in_blackout)
        self.assertIn("before", status.reason)

    def test_trading_stays_suspended_through_the_reaction(self):
        self.assertFalse(self.guard.is_tradeable(self.release_at + 1))
        self.assertFalse(self.guard.is_tradeable(self.release_at + 299))

    def test_trading_reopens_once_the_fast_money_is_done(self):
        self.assertTrue(self.guard.is_tradeable(self.release_at + 301))

    def test_the_post_release_window_is_longer_than_the_pre_release_one(self):
        # Being slow after the print is more dangerous than being early before it.
        self.assertGreater(self.guard.blackout_seconds_after, self.guard.blackout_seconds_before)

    def test_reports_the_next_release_when_clear(self):
        status = self.guard.status(self.release_at - 3600)
        self.assertFalse(status.in_blackout)
        self.assertEqual(status.release.name, "cpi")
        self.assertAlmostEqual(status.seconds_to_release, 3600)

    def test_known_release_names_pick_up_a_description(self):
        release = self.guard.schedule("fomc", self.release_at + 86400)
        self.assertIn("FOMC", release.description)

    def test_pruning_drops_finished_releases(self):
        self.guard.prune(self.release_at + 10000)
        self.assertEqual(self.guard.releases, [])

    def test_no_schedule_means_no_blackout(self):
        self.assertTrue(ScheduledEventGuard().is_tradeable(self.release_at))


class SecondaryRepricingTests(unittest.TestCase):
    def setUp(self):
        self.watcher = SecondaryRepricingWatcher(open_after_seconds=30, close_after_seconds=900, min_edge=0.04)
        self.released_at = 500_000.0
        self.watcher.record_release("cpi", self.released_at)

    def test_nothing_before_the_window_opens(self):
        self.assertIsNone(self.watcher.evaluate("fed-march-cut", "cpi", 0.30, 0.45,
                                                now=self.released_at + 5))

    def test_a_lagging_dependent_market_is_flagged(self):
        result = self.watcher.evaluate("fed-march-cut", "cpi", 0.30, 0.45, now=self.released_at + 120)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.edge, 0.15, places=9)
        self.assertIn("fed-march-cut", result.describe())

    def test_nothing_once_the_inference_is_common_knowledge(self):
        self.assertIsNone(self.watcher.evaluate("fed-march-cut", "cpi", 0.30, 0.45,
                                                now=self.released_at + 1200))

    def test_a_small_disagreement_is_not_a_signal(self):
        self.assertIsNone(self.watcher.evaluate("fed-march-cut", "cpi", 0.30, 0.32,
                                                now=self.released_at + 120))

    def test_an_unreleased_driver_produces_nothing(self):
        self.assertIsNone(self.watcher.evaluate("m", "nfp", 0.30, 0.60, now=self.released_at + 120))


class FedLadderTests(unittest.TestCase):
    def test_a_balanced_ladder_sums_to_one(self):
        total, deviation = fed_rate_ladder_outcomes({"3.75-4.00": 0.60, "4.00-4.25": 0.30, "4.25-4.50": 0.10})
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(deviation, 0.0)

    def test_an_over_summed_ladder_is_a_negrisk_opportunity(self):
        prices = {"3.50-3.75": 0.08, "3.75-4.00": 0.55, "4.00-4.25": 0.34, "4.25-4.50": 0.09}
        total, deviation = fed_rate_ladder_outcomes(prices)
        self.assertGreater(deviation, 0.0)

        scanner = NegRiskScanner(PolymarketFeeModel("us_macro"), min_net_margin=0.01)
        opportunity = scanner.scan("fed-ladder", prices, is_taker=False)
        self.assertEqual(opportunity.direction, "BUY_ALL_NO")
        self.assertTrue(opportunity.tradeable)

    def test_per_leg_fees_can_kill_the_ladder_for_a_taker(self):
        prices = {f"range{i}": 0.09 for i in range(10)}
        scanner = NegRiskScanner(PolymarketFeeModel("us_macro"), min_net_margin=0.01)
        as_taker = scanner.scan("fed-ladder", prices, is_taker=True)
        as_maker = scanner.scan("fed-ladder", prices, is_taker=False)
        self.assertGreater(as_maker.net_margin, as_taker.net_margin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
