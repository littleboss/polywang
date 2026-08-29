#!/usr/bin/env python3
"""Category resolution, per-market fee flags, and volatility uncertainty."""

from __future__ import annotations

import math
import random

import pytest

from polywang.arbitrage_core import BinaryMarket, _category
from polywang.crypto_model import (
    ProbabilityBand,
    RealisedVolatility,
    digital_call_probability,
    digital_call_probability_band,
)
from polywang.negrisk import NegRiskMarket
from polywang.polymarket_edge import (
    CATEGORY_MAKER_REBATE_SHARES,
    CATEGORY_TAKER_FEE_RATES,
    PolymarketFeeModel,
    fee_rate_from_bps,
    resolve_category,
)


class TestCategoryResolution:
    """Gamma tags are free-form; the rate table keys are not."""

    @pytest.mark.parametrize("tag", [
        "btc", "BTC", "bitcoin", "Bitcoin", "eth", "Ethereum",
        "Crypto Prices", "crypto-prices", "sol", "defi", "altcoins",
    ])
    def test_coin_tags_resolve_to_crypto(self, tag):
        assert resolve_category(tag) == "crypto"

    @pytest.mark.parametrize("tag", [
        "us_macro", "US Macro", "macro", "CPI", "core_cpi", "PCE",
        "FOMC", "fed", "NFP", "payrolls", "unemployment", "GDP", "inflation",
    ])
    def test_us_macro_tags_resolve_to_economics(self, tag):
        assert resolve_category(tag) == "economics"

    @pytest.mark.parametrize("canonical", sorted(CATEGORY_TAKER_FEE_RATES))
    def test_canonical_names_pass_through(self, canonical):
        assert resolve_category(canonical) == canonical

    @pytest.mark.parametrize("tag", [None, "", "   ", "something nobody has seen"])
    def test_unrecognised_tags_fall_back_rather_than_raise(self, tag):
        assert resolve_category(tag) == "other"

    def test_separators_are_normalised(self):
        assert resolve_category("us-macro") == resolve_category("us macro") == "economics"


class TestCryptoFeeMispricing:
    """
    A crypto market tagged anything other than the literal word "crypto" used to
    miss the rate table and fall through to the 0.05 default while the venue
    charges 0.07. Understating a fee is the direction that admits losing trades.
    """

    @pytest.mark.parametrize("tag", ["crypto", "Bitcoin", "BTC", "eth", "Crypto Prices"])
    def test_every_crypto_spelling_is_charged_the_crypto_rate(self, tag):
        assert PolymarketFeeModel(_category(tag)).taker_fee_rate == pytest.approx(0.07)

    @pytest.mark.parametrize("tag", ["economics", "US Macro", "CPI", "FOMC"])
    def test_every_macro_spelling_is_charged_the_economics_rate(self, tag):
        assert PolymarketFeeModel(_category(tag)).taker_fee_rate == pytest.approx(0.05)

    def test_the_mispricing_was_material(self):
        # The gap the bug produced, on the category where it mattered most.
        modelled = PolymarketFeeModel("other").fee_as_fraction_of_notional(0.50)
        actual = PolymarketFeeModel("bitcoin").fee_as_fraction_of_notional(0.50)
        assert actual == pytest.approx(0.035)
        assert modelled == pytest.approx(0.025)
        assert actual / modelled == pytest.approx(1.4)

    def test_gamma_payload_tagged_bitcoin_prices_the_fee_correctly(self):
        market = BinaryMarket.from_gamma({
            "id": "1", "conditionId": "0xabc",
            "question": "Will BTC close above $120k?",
            "clobTokenIds": '["tok-yes", "tok-no"]',
            "outcomes": '["Yes", "No"]',
            "category": "Bitcoin",
            "minimum_order_size": 5, "order_price_min_tick_size": 0.01,
        })
        assert market is not None
        assert market.category == "crypto"
        assert PolymarketFeeModel(market.category).taker_fee_rate == pytest.approx(0.07)


class TestFeesEnabled:
    """Markets can run fee-free whatever their category."""

    def test_a_fee_disabled_model_charges_nothing(self):
        model = PolymarketFeeModel("crypto", fees_enabled=False)
        assert model.fee_usd(1000, 0.50) == 0.0
        assert model.fee_as_fraction_of_notional(0.50) == 0.0

    def test_fees_default_to_enabled(self):
        assert PolymarketFeeModel("crypto").fee_usd(1000, 0.50) > 0.0

    def test_gamma_payload_carries_the_flag_through(self):
        payload = {
            "id": "1", "conditionId": "0xabc", "question": "q",
            "clobTokenIds": '["tok-yes", "tok-no"]', "outcomes": '["Yes", "No"]',
            "category": "crypto", "minimum_order_size": 5,
            "order_price_min_tick_size": 0.01, "feesEnabled": False,
        }
        market = BinaryMarket.from_gamma(payload)
        assert market is not None and market.fees_enabled is False

        payload["feesEnabled"] = True
        assert BinaryMarket.from_gamma(payload).fees_enabled is True

    def test_negrisk_market_defaults_to_fees_enabled(self):
        assert NegRiskMarket("m", "c", "t", tuple()).fees_enabled is True


class TestMakerRebate:
    def test_makers_are_paid_rather_than_merely_exempt(self):
        model = PolymarketFeeModel("crypto")
        assert model.fee_usd(100, 0.50, is_taker=False) == 0.0
        assert model.maker_rebate_usd(100, 0.50) == pytest.approx(
            model.fee_usd(100, 0.50) * 0.20)

    @pytest.mark.parametrize("category,share", sorted(CATEGORY_MAKER_REBATE_SHARES.items()))
    def test_published_rebate_shares(self, category, share):
        assert PolymarketFeeModel(category).maker_rebate_share == pytest.approx(share)

    def test_an_explicit_share_overrides_the_table(self):
        assert PolymarketFeeModel("crypto", maker_rebate_share=0.0).maker_rebate_usd(100, 0.5) == 0.0

    def test_a_fee_free_market_pays_no_rebate(self):
        assert PolymarketFeeModel("geopolitics").maker_rebate_usd(100, 0.50) == 0.0


class TestBasisPointConversion:
    @pytest.mark.parametrize("bps,rate", [(700, 0.07), (500, 0.05), (400, 0.04), (0, 0.0)])
    def test_conversion_matches_the_published_rates(self, bps, rate):
        assert fee_rate_from_bps(bps) == pytest.approx(rate)


class TestProbabilityBand:
    def test_band_contains_the_point_estimate(self):
        band = digital_call_probability_band(105_000, 100_000, 0.60, 14 / 365)
        assert band.low <= band.point <= band.high
        assert band.width > 0.0

    @pytest.mark.parametrize("days", [0.5, 2, 7, 14, 30, 60, 90, 180, 365])
    @pytest.mark.parametrize("spot", [95_000, 99_000, 100_000, 101_000, 118_000, 140_000])
    def test_band_contains_the_point_wherever_the_curve_turns(self, days, spot):
        # Probability is not monotone in volatility: widening the distribution
        # pulls towards 0.50 while the drift term pushes down, so the extreme can
        # be interior. Checking only the endpoints produced bands that excluded
        # their own point estimate.
        band = digital_call_probability_band(spot, 100_000, 0.60, days / 365)
        assert band.low <= band.point + 1e-12
        assert band.high >= band.point - 1e-12

    def test_no_uncertainty_means_no_band(self):
        band = digital_call_probability_band(105_000, 100_000, 0.60, 14 / 365, vol_error=0.0)
        assert band.width == pytest.approx(0.0, abs=1e-12)

    def test_a_wider_error_assumption_widens_the_band(self):
        tight = digital_call_probability_band(108_000, 100_000, 0.60, 14 / 365, vol_error=0.10)
        loose = digital_call_probability_band(108_000, 100_000, 0.60, 14 / 365, vol_error=0.50)
        assert loose.width > tight.width

    def test_invalid_inputs_return_none_rather_than_a_number(self):
        assert digital_call_probability_band(0, 100_000, 0.60, 0.1) is None
        assert digital_call_probability_band(100_000, 100_000, 0.0, 0.1) is None
        assert digital_call_probability_band(100_000, 100_000, 0.60, 0.0) is None

    def test_confidence_collapses_when_the_band_swamps_the_edge(self):
        band = digital_call_probability_band(100_000, 100_000, 0.80, 30 / 365, vol_error=0.40)
        assert band.confidence_against(band.point - 0.001) == 0.0

    def test_confidence_is_high_when_the_edge_dwarfs_the_band(self):
        band = digital_call_probability_band(150_000, 100_000, 0.60, 3 / 365)
        assert band.confidence_against(0.50) > 0.8

    def test_confidence_is_bounded(self):
        band = ProbabilityBand(point=0.6, low=0.55, high=0.65, vol=0.6, vol_error=0.25)
        for price in (0.0, 0.3, 0.6, 0.9, 1.0):
            assert 0.0 <= band.confidence_against(price) <= 1.0

    def test_the_point_matches_the_plain_digital_call(self):
        band = digital_call_probability_band(105_000, 100_000, 0.60, 14 / 365)
        assert band.point == pytest.approx(
            digital_call_probability(105_000, 100_000, 0.60, 14 / 365))


class TestRealisedVolatility:
    def test_reports_nothing_until_there_is_enough_evidence(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, minimum_samples=30)
        estimator.extend([100.0, 101.0, 100.5])
        assert not estimator.is_ready
        assert estimator.annualised() is None

    def test_recovers_a_known_volatility(self):
        random.seed(7)
        step = 0.001
        prices = [100.0]
        for _ in range(4000):
            prices.append(prices[-1] * math.exp(random.choice([step, -step])))

        estimator = RealisedVolatility(sample_interval_seconds=60, window=4000, minimum_samples=30)
        estimator.extend(prices)

        expected = step * math.sqrt(365 * 24 * 60)
        assert estimator.annualised() == pytest.approx(expected, rel=0.05)

    def test_a_flat_series_has_no_volatility(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, minimum_samples=10)
        estimator.extend([100.0] * 50)
        assert estimator.annualised() == pytest.approx(0.0, abs=1e-12)

    def test_the_window_is_bounded(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, window=50)
        estimator.extend([100.0 + i for i in range(500)])
        assert estimator.sample_count <= 50

    def test_the_sampling_interval_changes_the_annualisation(self):
        prices = [100.0 * math.exp(0.001 * ((-1) ** i)) for i in range(200)]
        fast = RealisedVolatility(sample_interval_seconds=1, window=200, minimum_samples=10)
        slow = RealisedVolatility(sample_interval_seconds=3600, window=200, minimum_samples=10)
        fast.extend(prices)
        slow.extend(prices)
        assert fast.annualised() > slow.annualised()

    def test_junk_observations_are_ignored(self):
        estimator = RealisedVolatility(sample_interval_seconds=60, minimum_samples=2)
        estimator.extend([100.0, None, "abc", -5.0, 0.0, float("nan"), 101.0, 102.0])
        assert estimator.sample_count == 2
        assert estimator.annualised() is not None

    def test_a_nonsense_interval_is_rejected(self):
        with pytest.raises(ValueError):
            RealisedVolatility(sample_interval_seconds=0)
