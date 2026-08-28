#!/usr/bin/env python3
"""
Fair value for Polymarket crypto markets (BTC, ETH and friends).

A market like "Will BTC be above $120,000 on Friday?" is a digital call option.
That changes what the bot should be doing, because unlike a football match the
underlying is observable to the cent, continuously, for free.

Why this replaces the Poisson engine rather than extending it
-------------------------------------------------------------
The sports engine exists because you cannot observe "how likely Newcastle are to
win" — you infer it from a scoreline and a goal-arrival model. For a coin there
is nothing to infer: the spot price is the state variable, and the probability of
finishing above a strike follows from it and a volatility estimate. Substituting
a goal model here would be throwing away the one input that is actually known.

What does carry over is the latency structure. When spot jumps, the Polymarket
book often lags by seconds, which is the same trade as a goal the book has not
priced. The difference is that the trigger is continuous rather than discrete, so
there is no natural exploit window to time out on. `CryptoLatencyEngine` uses the
size of the gap and its persistence instead.

What does not carry over is cost tolerance. Crypto is the most expensive category
on the venue at a 0.07 taker rate: 3.5% of notional at a price of 0.50, so 7% for
a round trip. Scalping in and out of a mid-priced crypto contract as a taker is
close to unwinnable. These markets have to be held to resolution, or worked with
resting orders, or left alone.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import statistics

SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def probability_above(spot: float, strike: float, annual_volatility: float,
                      days_to_expiry: float, annual_drift: float = 0.0) -> float:
    """
    Probability that the underlying finishes above `strike`, under a lognormal
    random walk. This is the N(d2) term of a digital call:

        d2 = [ln(S / K) + (mu - sigma^2 / 2) * T] / (sigma * sqrt(T))

    `annual_drift` defaults to zero, which is the assumption that the expected
    price is today's price. That is the neutral stance and the right default: any
    other drift is a directional forecast smuggled into what should be a pricing
    function, and it will quietly bias every market the bot looks at.

    Note the model's shape, because it is where the errors come from. Real crypto
    returns have fatter tails than a lognormal, so this understates the odds of
    reaching a distant strike and overstates the safety of a near-certain one.
    Trust it least at the extremes, which is exactly where its output looks most
    confident.
    """
    if spot <= 0 or strike <= 0:
        return 0.0

    time_years = max(0.0, days_to_expiry) / 365.0

    # At expiry, or with no volatility, the question is settled by inspection.
    if time_years <= 0.0 or annual_volatility <= 0.0:
        return 1.0 if spot > strike else 0.0

    sigma_sqrt_t = annual_volatility * math.sqrt(time_years)
    d2 = (math.log(spot / strike) + (annual_drift - 0.5 * annual_volatility ** 2) * time_years) / sigma_sqrt_t
    return _clamp(_standard_normal_cdf(d2), 0.0, 1.0)


def probability_below(spot: float, strike: float, annual_volatility: float,
                      days_to_expiry: float, annual_drift: float = 0.0) -> float:
    return 1.0 - probability_above(spot, strike, annual_volatility, days_to_expiry, annual_drift)


def probability_in_range(spot: float, lower: float, upper: float, annual_volatility: float,
                         days_to_expiry: float, annual_drift: float = 0.0) -> float:
    """Probability of finishing inside a band, used for bracketed price markets."""
    if lower >= upper:
        return 0.0
    above_lower = probability_above(spot, lower, annual_volatility, days_to_expiry, annual_drift)
    above_upper = probability_above(spot, upper, annual_volatility, days_to_expiry, annual_drift)
    return max(0.0, above_lower - above_upper)


@dataclass
class ProbabilityBand:
    """
    Fair value together with how far it moves when the volatility input is wrong.

    Volatility is an estimate, not an observation, and the whole answer hinges on
    it. Carrying the band alongside the point estimate is what lets the bot tell
    a real edge from the width of its own uncertainty.
    """
    point: float
    low: float
    high: float
    volatility: float
    volatility_error: float

    @property
    def width(self) -> float:
        return abs(self.high - self.low)

    def confidence_against(self, market_price: float) -> float:
        """
        How much of the apparent edge survives the volatility uncertainty, on a
        0-1 scale suitable for shrinking a Kelly stake.

        When the band is wider than the edge itself the model cannot distinguish
        a mispriced contract from its own error, and this returns zero. That is
        the honest answer, and it is the guard that stops the bot from sizing up
        on a near-the-money contract where a small change in the volatility
        assumption flips the sign of the trade.
        """
        edge = abs(self.point - market_price)
        if edge <= 1e-9:
            return 0.0
        return _clamp(1.0 - (self.width / (2.0 * edge)), 0.0, 1.0)


def probability_band(spot: float, strike: float, annual_volatility: float, days_to_expiry: float,
                     volatility_error: float = 0.25, annual_drift: float = 0.0,
                     samples: int = 21) -> ProbabilityBand:
    """
    Reprices the contract across volatility plus and minus `volatility_error`.

    A 25% default is not pessimism. Realised volatility over a short window is a
    noisy estimator of the volatility that will actually be realised, and crypto
    regimes shift fast enough that being a quarter out is ordinary.

    The range is swept rather than evaluated at its two ends, because the
    probability is not monotone in volatility. More volatility widens the
    distribution, pulling the answer towards 0.50, but it also drags the median
    down through the -sigma^2 T / 2 term. Those pull in opposite directions above
    the strike, so the extreme can sit in the middle of the range and checking
    only the endpoints would report a band that does not contain its own point
    estimate.
    """
    error = max(0.0, volatility_error)
    low_vol = max(0.0, annual_volatility * (1.0 - error))
    high_vol = annual_volatility * (1.0 + error)

    point = probability_above(spot, strike, annual_volatility, days_to_expiry, annual_drift)

    steps = max(2, samples)
    values = [point]
    for i in range(steps):
        vol = low_vol + (high_vol - low_vol) * (i / (steps - 1))
        values.append(probability_above(spot, strike, vol, days_to_expiry, annual_drift))

    return ProbabilityBand(
        point=point,
        low=min(values),
        high=max(values),
        volatility=annual_volatility,
        volatility_error=error,
    )


class RealisedVolatility:
    """
    Rolling annualised volatility from a stream of spot prices.

    Uses log returns, because prices compound. The sampling interval has to be
    passed in rather than inferred: annualising assumes every observation covers
    the same span, and a feed that skips ticks will otherwise report a volatility
    that is quietly wrong by the square root of however much it skipped.
    """

    def __init__(self, sample_interval_seconds: float = 60.0, window: int = 240,
                 minimum_samples: int = 30):
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.sample_interval_seconds = sample_interval_seconds
        self.window = window
        self.minimum_samples = minimum_samples
        self._prices: List[float] = []

    def add(self, price: float) -> None:
        if price <= 0:
            return
        self._prices.append(price)
        # Keep one extra price so `window` returns remain computable.
        if len(self._prices) > self.window + 1:
            self._prices = self._prices[-(self.window + 1):]

    def extend(self, prices: Sequence[float]) -> None:
        for price in prices:
            self.add(price)

    @property
    def sample_count(self) -> int:
        return max(0, len(self._prices) - 1)

    @property
    def is_ready(self) -> bool:
        return self.sample_count >= self.minimum_samples

    def log_returns(self) -> List[float]:
        return [math.log(b / a) for a, b in zip(self._prices, self._prices[1:]) if a > 0 and b > 0]

    def annualised(self) -> Optional[float]:
        """Annualised volatility, or None until there are enough samples to mean it."""
        returns = self.log_returns()
        if len(returns) < max(2, self.minimum_samples):
            return None
        periods_per_year = SECONDS_PER_YEAR / self.sample_interval_seconds
        return statistics.stdev(returns) * math.sqrt(periods_per_year)


@dataclass
class CryptoDislocation:
    """A crypto contract whose book has not caught up with spot."""
    market_id: str
    spot: float
    strike: float
    market_price: float
    fair_probability: float
    band: ProbabilityBand
    edge: float
    confidence: float
    spot_move_pct: float
    seconds_since_book_moved: float

    def describe(self) -> str:
        return (
            f"spot ${self.spot:,.2f} vs strike ${self.strike:,.2f}: fair {self.fair_probability:.3f} "
            f"(band {self.band.low:.3f}-{self.band.high:.3f}), book at {self.market_price:.3f}, "
            f"edge {self.edge * 100:+.1f}pp at {self.confidence * 100:.0f}% confidence. "
            f"Spot moved {self.spot_move_pct * 100:+.2f}% and the book has been still for "
            f"{self.seconds_since_book_moved:.1f}s."
        )


class CryptoLatencyEngine:
    """
    Watches spot against the book and reports contracts the book has not repriced.

    The sports engine waits for a discrete event and then opens a fixed window.
    Spot has no events, so the trigger here is a move large enough to matter
    relative to what the contract is worth, combined with a book that has not
    moved since. Requiring both is what separates a genuine lag from a market
    that has simply already agreed with us.
    """

    def __init__(self,
                 volatility_estimator: Optional[RealisedVolatility] = None,
                 min_spot_move_pct: float = 0.002,
                 min_edge: float = 0.03,
                 volatility_error: float = 0.25):
        self.volatility = volatility_estimator or RealisedVolatility()
        self.min_spot_move_pct = min_spot_move_pct
        self.min_edge = min_edge
        self.volatility_error = volatility_error

        self._last_spot: Optional[float] = None
        self._reference_spot: Optional[float] = None

    def update_spot(self, price: float) -> float:
        """Feeds a spot observation and returns the move since the last reference."""
        self.volatility.add(price)
        if self._reference_spot is None:
            self._reference_spot = price
        move = (price / self._reference_spot - 1.0) if self._reference_spot else 0.0
        self._last_spot = price
        return move

    def reset_reference(self) -> None:
        """Marks the current spot as the new baseline, after acting on a move."""
        self._reference_spot = self._last_spot

    def evaluate(self, market_id: str, strike: float, market_price: float, days_to_expiry: float,
                 seconds_since_book_moved: float, spot: Optional[float] = None,
                 annual_volatility: Optional[float] = None) -> Optional[CryptoDislocation]:
        spot = self._last_spot if spot is None else spot
        if spot is None or spot <= 0:
            return None

        volatility = annual_volatility if annual_volatility is not None else self.volatility.annualised()
        if volatility is None or volatility <= 0:
            # Without a volatility estimate there is no fair value, and guessing
            # one produces a number that looks authoritative and is not.
            return None

        band = probability_band(spot, strike, volatility, days_to_expiry,
                                volatility_error=self.volatility_error)
        edge = band.point - market_price
        if abs(edge) < self.min_edge:
            return None

        spot_move = (spot / self._reference_spot - 1.0) if self._reference_spot else 0.0
        if abs(spot_move) < self.min_spot_move_pct:
            return None

        return CryptoDislocation(
            market_id=market_id,
            spot=spot,
            strike=strike,
            market_price=market_price,
            fair_probability=band.point,
            band=band,
            edge=edge,
            confidence=band.confidence_against(market_price),
            spot_move_pct=spot_move,
            seconds_since_book_moved=seconds_since_book_moved,
        )


def round_trip_cost_fraction(taker_fee_rate: float, price: float) -> float:
    """
    Cost of entering and exiting at the same price, as a fraction of notional.

    Crypto's 0.07 rate makes this the number that decides whether a strategy is
    viable at all: 7% at a price of 0.50 means a scalp has to be right about a
    seven-point probability move before it earns anything. Holding to resolution
    pays this once; a resting order pays none of it.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    per_side = taker_fee_rate * (1.0 - price)
    return 2.0 * per_side
