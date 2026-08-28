#!/usr/bin/env python3
"""
Edge evaluation and position sizing for Polymarket.

This module replaces the flat-percentage friction model with the fee formula
Polymarket actually uses, and adds the two dimensions that decide whether a
high-probability contract is worth owning: how long your capital is locked up,
and how much of the bankroll the edge justifies risking.

Sources
-------
Fee formula and per-category rates: https://docs.polymarket.com/trading/fees
    fee = C x feeRate x p x (1 - p),  charged to takers only, makers pay zero.

The consequence that drives most of this module: expressed as a fraction of
notional, that fee is `feeRate x (1 - p)`. It falls towards zero as the price
approaches 1.00, so a flat percentage model overstates the cost of favourites
by an order of magnitude and understates the cost of longshots.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import json
import os
import tempfile

# Taker fee rate per market category, from the Polymarket fee schedule.
# Makers are never charged, in any category.
CATEGORY_TAKER_FEE_RATES: Dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "general": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "tech": 0.04,
    "mentions": 0.04,
    "geopolitics": 0.0,
}

DEFAULT_CATEGORY = "other"

# Fees below this round to zero on the platform.
MIN_CHARGEABLE_FEE_USDC = 0.00001


class PolymarketFeeModel:
    """
    The platform's real fee schedule.

    Two properties matter for strategy design and neither survives a flat
    percentage approximation:

    1. With the current default exponent of one, the fee is quadratic in price,
       peaking at p = 0.50 and vanishing at both extremes. Market-specific fee
       schedules may override the rate and exponent.
    2. Makers pay nothing. Crossing the spread is a choice with a price tag, so
       any edge that is not time-critical should be captured with a resting
       limit order instead.
    """

    def __init__(self, category: str = DEFAULT_CATEGORY, maker_rebate_share: float = 0.0,
                 taker_fee_rate: Optional[float] = None, fee_exponent: float = 1.0):
        self.category = (category or DEFAULT_CATEGORY).strip().lower()
        default_rate = CATEGORY_TAKER_FEE_RATES.get(self.category, CATEGORY_TAKER_FEE_RATES[DEFAULT_CATEGORY])
        self.taker_fee_rate = max(0.0, float(taker_fee_rate)) if taker_fee_rate is not None else default_rate
        self.fee_exponent = max(0.0, float(fee_exponent))
        self.maker_rebate_share = maker_rebate_share

    def fee_usd(self, shares: float, price: float, is_taker: bool = True) -> float:
        """Total fee in USDC for a fill of `shares` at `price`."""
        fee = abs(shares) * self.fee_per_share(price, is_taker=is_taker)
        return 0.0 if fee < MIN_CHARGEABLE_FEE_USDC else fee

    def fee_per_share(self, price: float, is_taker: bool = True) -> float:
        if not is_taker or self.taker_fee_rate <= 0.0:
            return 0.0
        price = _clamp(price, 0.0, 1.0)
        return self.taker_fee_rate * (price * (1.0 - price)) ** self.fee_exponent

    def fee_as_fraction_of_notional(self, price: float, is_taker: bool = True) -> float:
        """
        Fee expressed against the dollars deployed, which is the number worth
        comparing against an expected return. For the default exponent this is
        algebraically `feeRate x (1 - p)`.
        """
        if not is_taker or price <= 0.0:
            return 0.0
        return self.fee_per_share(price, is_taker=True) / price


@dataclass
class BookFill:
    """Result of consuming resting liquidity up to a budget."""
    shares: float
    average_price: float
    worst_price: float
    budget_filled: float
    budget_unfilled: float

    @property
    def is_complete(self) -> bool:
        return self.budget_unfilled <= 1e-9


def walk_order_book(levels: Sequence[Tuple[float, float]], usd_budget: float,
                    max_price: Optional[float] = None) -> BookFill:
    """
    Walks resting asks to work out what a given budget actually buys.

    A flat slippage percentage assumes depth that thin prediction markets do not
    have. Two dollars and two thousand dollars do not receive the same fill, and
    the difference is often larger than the entire modelled edge, so slippage has
    to be read off the book rather than assumed.

    `levels` is an iterable of (price, size_in_shares), best price first.
    `max_price` stops the walk once the book is worse than the ceiling.
    """
    remaining = max(0.0, usd_budget)
    shares = 0.0
    spent = 0.0
    worst = 0.0

    for price, size in sorted(levels, key=lambda level: level[0]):
        if remaining <= 1e-9:
            break
        if price <= 0.0 or size <= 0.0:
            continue
        if max_price is not None and price > max_price:
            break

        affordable_shares = min(size, remaining / price)
        if affordable_shares <= 0.0:
            continue

        shares += affordable_shares
        cost = affordable_shares * price
        spent += cost
        remaining -= cost
        worst = price

    average = (spent / shares) if shares > 0 else 0.0
    return BookFill(
        shares=shares,
        average_price=average,
        worst_price=worst,
        budget_filled=spent,
        budget_unfilled=max(0.0, remaining),
    )


@dataclass
class EdgeAssessment:
    """Everything needed to accept or reject a candidate trade, and why."""
    price: float
    fair_probability: float
    entry_price: float
    fee_per_share: float
    cost_per_share: float
    breakeven_probability: float
    ev_per_share: float
    ev_per_dollar: float
    days_to_resolution: float
    required_period_return: float
    annualised_return: float
    kelly_fraction: float
    recommended_stake_usd: float
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def explain(self) -> str:
        verdict = "ACCEPT" if self.accepted else "REJECT"
        return (
            f"[{verdict}] price ${self.price:.4f} vs fair {self.fair_probability:.4f}\n"
            f"  break-even probability : {self.breakeven_probability:.4f}\n"
            f"  cost per share         : ${self.cost_per_share:.4f} (fee ${self.fee_per_share:.5f})\n"
            f"  expected value         : ${self.ev_per_share:.4f}/share, {self.ev_per_dollar * 100:.2f}% on capital\n"
            f"  horizon                : {self.days_to_resolution:.1f}d, needs {self.required_period_return * 100:.2f}%, "
            f"annualised {self.annualised_return * 100:.1f}%\n"
            f"  Kelly                  : {self.kelly_fraction * 100:.2f}% of bankroll -> ${self.recommended_stake_usd:.2f}\n"
            + "".join(f"  - {reason}\n" for reason in self.reasons)
            + "".join(f"  ! {warning}\n" for warning in self.warnings)
        )


class EdgeEvaluator:
    """
    Decides whether a candidate trade is worth the capital it consumes.

    Three hurdles, each rejecting a different way to lose money:

    `min_ev_per_dollar` rejects trades whose edge does not survive fees. This is
    the classic failure where a high hit rate still loses money because each win
    is too small to pay for itself.

    `hurdle_apr` rejects trades that are profitable but slow. Three cents of
    edge on a 0.97 contract is excellent over two days and poor over eight
    months, and nothing in a raw edge number distinguishes them. Capital sitting
    in a resolved-in-June contract cannot take the next opportunity.

    `min_edge_over_breakeven` rejects trades that need the model to be more
    precise than it can be. At a price of 0.97 the break-even probability is
    around 0.9715, so anyone claiming an edge there is claiming accuracy to a
    fraction of a percentage point. Demanding a visible margin above break-even
    keeps the bot away from bets it cannot actually evaluate.
    """

    def __init__(self,
                 fee_model: Optional[PolymarketFeeModel] = None,
                 min_ev_per_dollar: float = 0.02,
                 hurdle_apr: float = 0.15,
                 min_edge_over_breakeven: float = 0.02,
                 kelly_fraction: float = 0.25,
                 max_position_fraction: float = 0.10,
                 estimate_confidence: float = 0.5,
                 longshot_price_threshold: float = 0.20):
        self.fee_model = fee_model or PolymarketFeeModel()
        self.min_ev_per_dollar = min_ev_per_dollar
        self.hurdle_apr = hurdle_apr
        self.min_edge_over_breakeven = min_edge_over_breakeven
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.estimate_confidence = estimate_confidence
        self.longshot_price_threshold = longshot_price_threshold

    def breakeven_probability(self, price: float, is_taker: bool = True) -> float:
        """
        The true probability at which the trade merely returns its cost.

        Cost per share is the price plus the fee, so break-even is
        `p x (1 + feeRate x (1 - p))` for a taker and simply `p` for a maker.
        """
        price = _clamp(price, 0.0, 1.0)
        return min(1.0, price + self.fee_model.fee_per_share(price, is_taker=is_taker))

    def assess(self,
               price: float,
               fair_probability: float,
               bankroll: float,
               days_to_resolution: float = 1.0,
               is_taker: bool = True,
               confidence: Optional[float] = None,
               exit_before_resolution: bool = False) -> EdgeAssessment:
        price = _clamp(price, 1e-6, 0.999999)
        fair = _clamp(fair_probability, 0.0, 1.0)
        days = max(days_to_resolution, 1.0 / 24.0)
        confidence = self.estimate_confidence if confidence is None else _clamp(confidence, 0.0, 1.0)

        fee_per_share = self.fee_model.fee_per_share(price, is_taker=is_taker)
        if exit_before_resolution:
            # Leaving early means crossing the book a second time. Resolution
            # itself is free, so holding on is strictly cheaper in fee terms.
            fee_per_share += self.fee_model.fee_per_share(price, is_taker=True)

        cost_per_share = price + fee_per_share
        breakeven = min(1.0, cost_per_share)

        ev_per_share = fair - cost_per_share
        ev_per_dollar = ev_per_share / cost_per_share if cost_per_share > 0 else 0.0

        # Convert the annual hurdle into what this specific holding period must
        # earn, rather than annualising a short holding period into a number too
        # large to reason about.
        required_period_return = (1.0 + self.hurdle_apr) ** (days / 365.0) - 1.0
        annualised = _annualise(ev_per_dollar, days)

        # Shrink the estimate towards the market before sizing. The market price
        # is itself a forecast produced by everyone else's information, so a
        # disagreement is only partly ours to bet on.
        shrunk_fair = price + confidence * (fair - price)
        kelly = _kelly_fraction(price, shrunk_fair)
        sized_fraction = min(self.kelly_fraction * kelly, self.max_position_fraction)
        sized_fraction = max(0.0, sized_fraction)
        stake = bankroll * sized_fraction

        reasons: List[str] = []
        if ev_per_dollar < self.min_ev_per_dollar:
            reasons.append(
                f"edge {ev_per_dollar * 100:.2f}% is under the {self.min_ev_per_dollar * 100:.2f}% minimum after fees"
            )
        if ev_per_dollar < required_period_return:
            reasons.append(
                f"{ev_per_dollar * 100:.2f}% over {days:.1f} days is below the "
                f"{required_period_return * 100:.2f}% this capital must earn to beat a {self.hurdle_apr * 100:.0f}% APR alternative"
            )
        if fair - breakeven < self.min_edge_over_breakeven:
            reasons.append(
                f"fair value {fair:.4f} is only {(fair - breakeven) * 100:.2f}pp above break-even {breakeven:.4f}; "
                f"the model is not accurate enough to bet on a margin that thin"
            )
        if stake <= 0.0:
            reasons.append("Kelly sizing returns zero, so there is no edge worth staking")

        warnings: List[str] = []
        if price <= self.longshot_price_threshold and fair > price:
            # Resolved-contract studies find longshots trade above their physical
            # probability, not below it. Claiming a cheap contract is still too
            # cheap runs against that, so the estimate needs outside support.
            warnings.append(
                f"at ${price:.2f} this is longshot territory, where market prices are documented to sit "
                f"above true probability; a model claiming further upside here is arguing against the bias"
            )
        if price >= 0.95:
            warnings.append(
                f"payoff is asymmetric: risking ${cost_per_share:.4f} to win "
                f"${1.0 - cost_per_share:.4f}, so one loss undoes roughly "
                f"{(cost_per_share / max(1e-9, 1.0 - cost_per_share)):.0f} wins"
            )

        accepted = not reasons
        if accepted:
            reasons.append(
                f"clears all hurdles: {ev_per_dollar * 100:.2f}% over {days:.1f} days "
                f"({annualised * 100:.0f}% annualised)"
            )

        return EdgeAssessment(
            price=price,
            fair_probability=fair,
            entry_price=price,
            fee_per_share=fee_per_share,
            cost_per_share=cost_per_share,
            breakeven_probability=breakeven,
            ev_per_share=ev_per_share,
            ev_per_dollar=ev_per_dollar,
            days_to_resolution=days,
            required_period_return=required_period_return,
            annualised_return=annualised,
            kelly_fraction=sized_fraction,
            recommended_stake_usd=stake,
            accepted=accepted,
            reasons=reasons,
            warnings=warnings,
        )

    def should_post_as_maker(self, edge_decay_seconds: Optional[float],
                             patience_seconds: float = 30.0) -> bool:
        """
        Whether to rest a limit order instead of crossing the spread.

        Makers pay no fee at all, so the taker fee is really the price of
        immediacy. Pay it only when the edge disappears while you wait: a goal
        that the book has not yet priced is worth crossing for, a structural
        mispricing that will still be there in a minute is not.
        """
        if edge_decay_seconds is None:
            return True
        return edge_decay_seconds > patience_seconds


def _kelly_fraction(price: float, probability: float) -> float:
    """
    Growth-optimal share of bankroll for a binary contract, which reduces to
    (q - p) / (1 - p) for a payout of 1.00 against a cost of p.

    Note how unstable this is near the top of the range: at a price of 0.97 the
    denominator is 0.03, so a one-point error in q moves the recommended stake
    by 33 percentage points. That sensitivity, not squeamishness, is why full
    Kelly is never used directly here.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    edge = probability - price
    if edge <= 0.0:
        return 0.0
    return edge / (1.0 - price)


# Annualising a holding period of hours produces figures in the billions, which
# are arithmetically correct and completely uninformative. Report anything above
# this as "off the scale" rather than pretending the number means something.
MAX_MEANINGFUL_APR = 100.0  # 10,000%


def _annualise(period_return: float, days: float) -> float:
    if days <= 0:
        return 0.0
    base = 1.0 + period_return
    if base <= 0.0:
        return -1.0
    try:
        annualised = base ** (365.0 / days) - 1.0
    except OverflowError:
        return MAX_MEANINGFUL_APR
    return min(annualised, MAX_MEANINGFUL_APR)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class NegRiskOpportunity:
    """
    A multi-outcome market whose prices do not sum to 1.00.

    Margins are quoted per complete set: buying one share of every outcome costs
    `price_sum` and pays exactly 1.00. `return_on_capital` restates the net
    margin against the money actually tied up, which is the comparable number.
    """
    market_id: str
    outcome_prices: Dict[str, float]
    price_sum: float
    gross_margin: float
    fee_per_set: float
    net_margin: float
    return_on_capital: float
    direction: str          # "BUY_ALL_YES" or "BUY_ALL_NO"
    tradeable: bool
    note: str


class NegRiskScanner:
    """
    Looks for structural mispricing in mutually exclusive multi-outcome markets.

    Exactly one outcome pays 1.00, so the YES prices must sum to 1.00. When they
    sum to less, buying the full set locks in the difference no matter which
    outcome wins; when they sum to more, the same logic runs on the NO side.

    The reason this persists is liquidity fragmentation: retail flow concentrates
    on the two favourites while the tail of the field trades thin, so the implied
    distribution stops adding up. It is the one edge here that does not depend on
    forecasting anything, which is also why the margins are small and get taken
    quickly.
    """

    def __init__(self, fee_model: Optional[PolymarketFeeModel] = None,
                 min_net_margin: float = 0.01,
                 gas_cost_usd: float = 0.0):
        self.fee_model = fee_model or PolymarketFeeModel()
        self.min_net_margin = min_net_margin
        self.gas_cost_usd = gas_cost_usd

    def scan(self, market_id: str, outcome_prices: Dict[str, float],
             is_taker: bool = True, set_size_usd: float = 100.0) -> Optional[NegRiskOpportunity]:
        prices = {k: _clamp(v, 0.0, 1.0) for k, v in outcome_prices.items() if v is not None}
        if len(prices) < 2:
            return None

        price_sum = sum(prices.values())
        deviation = price_sum - 1.0

        if deviation < 0:
            direction = "BUY_ALL_YES"
            gross_margin = -deviation
            legs = list(prices.values())
        else:
            direction = "BUY_ALL_NO"
            gross_margin = deviation
            # The NO side of each outcome trades at 1 - p, and the fee formula is
            # symmetric about 0.50, so each leg costs the same fee either way.
            legs = list(prices.values())

        # One fee per leg, and the leg count is what usually kills this trade as a
        # taker. The same basket costs nothing to assemble with resting orders.
        fee_per_set = sum(self.fee_model.fee_per_share(p, is_taker=is_taker) for p in legs)
        gas_per_set = (self.gas_cost_usd / set_size_usd) if set_size_usd > 0 else 0.0
        net_margin = gross_margin - fee_per_set - gas_per_set

        # A full NO basket is collateralised at the true maximum loss of 1.00 per
        # set rather than the naive sum of the legs, which is what makes the trade
        # fit a retail bankroll at all.
        capital_per_set = price_sum if direction == "BUY_ALL_YES" else 1.0
        return_on_capital = net_margin / capital_per_set if capital_per_set > 0 else 0.0

        tradeable = net_margin >= self.min_net_margin
        if gross_margin <= 0:
            note = "prices sum to 1.00; no structural edge"
        elif tradeable:
            note = (f"{len(prices)} legs, {gross_margin * 100:.2f}% gross - {fee_per_set * 100:.2f}% fees "
                    f"= {net_margin * 100:.2f}% per set ({return_on_capital * 100:.2f}% on capital)")
        else:
            note = (f"{gross_margin * 100:.2f}% gross is eaten by {fee_per_set * 100:.2f}% of taker fees "
                    f"across {len(prices)} legs; makers pay none of this")

        return NegRiskOpportunity(
            market_id=market_id,
            outcome_prices=prices,
            price_sum=price_sum,
            gross_margin=gross_margin,
            fee_per_set=fee_per_set,
            net_margin=net_margin,
            return_on_capital=return_on_capital,
            direction=direction,
            tradeable=tradeable,
            note=note,
        )


@dataclass
class StrategyCalibration:
    """Accuracy record for one signal source."""
    name: str
    forecasts: List[Tuple[float, int]] = field(default_factory=list)

    def record(self, forecast_probability: float, outcome: int) -> None:
        self.forecasts.append((_clamp(forecast_probability, 0.0, 1.0), 1 if outcome else 0))

    @property
    def count(self) -> int:
        return len(self.forecasts)

    @property
    def brier_score(self) -> Optional[float]:
        """
        Mean squared error of the probability forecasts. Lower is better; 0.25 is
        what you get by answering 0.50 to everything, so anything above that is
        worse than refusing to forecast.
        """
        if not self.forecasts:
            return None
        return sum((p - o) ** 2 for p, o in self.forecasts) / len(self.forecasts)

    @property
    def mean_forecast(self) -> Optional[float]:
        if not self.forecasts:
            return None
        return sum(p for p, _ in self.forecasts) / len(self.forecasts)

    @property
    def hit_rate(self) -> Optional[float]:
        if not self.forecasts:
            return None
        return sum(o for _, o in self.forecasts) / len(self.forecasts)

    @property
    def bias(self) -> Optional[float]:
        """
        Mean forecast minus realised frequency. Positive means the strategy talks
        itself into outcomes that do not happen as often as it claims.
        """
        if not self.forecasts:
            return None
        return self.mean_forecast - self.hit_rate

    def rolling_brier(self, window: int) -> Optional[float]:
        if not self.forecasts:
            return None
        sample = self.forecasts[-max(1, int(window)):]
        return sum((p - o) ** 2 for p, o in sample) / len(sample)

    def hit_rate_interval(self, z: float = 1.96) -> Optional[Tuple[float, float]]:
        """Wilson score interval for realised hit rate."""
        n = self.count
        if n <= 0:
            return None
        hits = sum(o for _, o in self.forecasts)
        p = hits / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def walk_forward_brier(self, train_fraction: float = 0.7) -> Optional[float]:
        if self.count < 4:
            return None
        split = max(1, min(self.count - 1, int(self.count * train_fraction)))
        held_out = self.forecasts[split:]
        if not held_out:
            return None
        return sum((p - o) ** 2 for p, o in held_out) / len(held_out)

    def reliability_table(self, buckets: int = 5) -> List[Tuple[str, int, float, float]]:
        """Forecast bucket, sample count, mean forecast, realised frequency."""
        grouped: Dict[int, List[Tuple[float, int]]] = {}
        for probability, outcome in self.forecasts:
            index = min(buckets - 1, int(probability * buckets))
            grouped.setdefault(index, []).append((probability, outcome))

        rows = []
        for index in sorted(grouped):
            entries = grouped[index]
            low = index / buckets
            high = (index + 1) / buckets
            mean_forecast = sum(p for p, _ in entries) / len(entries)
            realised = sum(o for _, o in entries) / len(entries)
            rows.append((f"{low:.1f}-{high:.1f}", len(entries), mean_forecast, realised))
        return rows


class CalibrationTracker:
    """
    Scores every strategy against what actually happened.

    Without this the bot cannot tell a working strategy from a lucky one, and
    "fair value" stays an assertion. A strategy whose Brier score is worse than
    0.25 is contributing less than a coin flip and should stop sizing positions
    until it is fixed.
    """

    WORSE_THAN_UNINFORMATIVE = 0.25

    def __init__(self, min_samples: int = 20, path: str = ""):
        self.min_samples = max(1, int(min_samples))
        self.path = path
        self.strategies: Dict[str, StrategyCalibration] = {}
        self.load()

    def load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("calibration state must be an object")
            for name, values in loaded.items():
                if not isinstance(name, str) or not isinstance(values, list):
                    continue
                record = StrategyCalibration(name)
                for entry in values:
                    if isinstance(entry, list) and len(entry) == 2:
                        record.record(entry[0], entry[1])
                self.strategies[name] = record
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Calibration state is unreadable: {self.path}") from error

    def save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".calibration-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({name: record.forecasts for name, record in self.strategies.items()},
                          handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def record(self, strategy: str, forecast_probability: float, outcome: int) -> None:
        self.strategies.setdefault(strategy, StrategyCalibration(strategy)).record(forecast_probability, outcome)
        self.save()

    def is_trustworthy(self, strategy: str) -> bool:
        """
        A strategy is trusted until it has enough evidence to convict it. Blocking
        a new strategy before it has a track record would prevent it from ever
        acquiring one.
        """
        record = self.strategies.get(strategy)
        if record is None or record.count < self.min_samples:
            return True
        return record.brier_score < self.WORSE_THAN_UNINFORMATIVE

    def is_live_ready(self, strategy: str) -> bool:
        """Require evidence before a predictive signal can reach live trading."""
        record = self.strategies.get(strategy)
        if record is None or record.count < self.min_samples or not self.is_trustworthy(strategy):
            return False
        if self.has_drifted(strategy):
            return False
        oos = record.walk_forward_brier()
        if oos is not None and oos >= self.WORSE_THAN_UNINFORMATIVE:
            return False
        return True

    def has_drifted(self, strategy: str, window: int = 20, threshold: float = 0.08) -> bool:
        """Recent Brier worse than the prior window by more than `threshold`."""
        record = self.strategies.get(strategy)
        if record is None or record.count < max(self.min_samples, window * 2):
            return False
        recent = record.rolling_brier(window)
        older = record.forecasts[:-window]
        if recent is None or not older:
            return False
        prior = sum((p - o) ** 2 for p, o in older[-window:]) / min(window, len(older))
        return recent - prior > threshold

    def recommended_parameters(self, strategy: str) -> dict:
        """Shrink confidence and raise the edge hurdle when the record is noisy."""
        record = self.strategies.get(strategy)
        defaults = {
            "estimate_confidence": 0.5,
            "min_edge_over_breakeven": 0.02,
            "live_ready": False,
            "reason": "insufficient samples",
        }
        if record is None or record.count < self.min_samples:
            return defaults
        interval = record.hit_rate_interval()
        width = (interval[1] - interval[0]) if interval else 1.0
        bias = abs(record.bias or 0.0)
        confidence = max(0.05, min(0.5, 0.5 - bias - 0.5 * max(0.0, width - 0.2)))
        min_edge = max(0.02, 0.02 + bias + 0.5 * width)
        ready = self.is_live_ready(strategy)
        if self.has_drifted(strategy):
            reason = "recent forecasts drifted versus the prior window"
        elif not ready:
            reason = "out-of-sample Brier is not yet informative"
        else:
            reason = "rolling calibration updated edge and confidence"
        return {
            "estimate_confidence": confidence,
            "min_edge_over_breakeven": min_edge,
            "live_ready": ready,
            "brier": record.brier_score,
            "walk_forward_brier": record.walk_forward_brier(),
            "bias": record.bias,
            "hit_rate_interval": interval,
            "reason": reason,
        }

    def apply_recommended_edge(self, evaluator, strategy: str):
        """Optionally retune an EdgeEvaluator from persisted calibration."""
        params = self.recommended_parameters(strategy)
        evaluator.estimate_confidence = float(params["estimate_confidence"])
        evaluator.min_edge_over_breakeven = float(params["min_edge_over_breakeven"])
        return params

    def underperformers(self) -> List[str]:
        return [name for name in self.strategies if not self.is_trustworthy(name)]

    def report(self) -> str:
        if not self.strategies:
            return "No forecasts recorded yet."

        lines = [f"{'strategy':<28}{'n':>6}{'brier':>9}{'bias':>9}  verdict"]
        for name in sorted(self.strategies):
            record = self.strategies[name]
            brier = record.brier_score
            bias = record.bias
            if record.count < self.min_samples:
                verdict = f"needs {self.min_samples - record.count} more samples"
            elif brier < self.WORSE_THAN_UNINFORMATIVE:
                verdict = "informative"
            else:
                verdict = "WORSE THAN A COIN FLIP — stop sizing on this"
            lines.append(f"{name:<28}{record.count:>6}{brier:>9.4f}{bias:>+9.4f}  {verdict}")
        return "\n".join(lines)


def debias_market_price(market_price: float, wedge_lambda: float = 0.176) -> float:
    """
    Estimates physical probability from a market price using the Wang transform,
    p_market = Phi(Phi^-1(p*) + lambda), inverted for p*.

    Real-money prediction markets price in a positive wedge: the YES side sits
    above the physical probability, and proportionally much further above it for
    longshots than for favourites. Calibrated at lambda ~= 0.176 on resolved
    Polymarket contracts.

    Treat this as a prior, not a signal. The published estimates put the wedge at
    roughly zero in the highest-volume markets and show it decaying to nothing as
    a contract approaches resolution, so applying a static lambda to a liquid,
    near-expiry market invents an edge that is not there. Pass wedge_lambda=0.0
    to switch it off.
    """
    price = _clamp(market_price, 1e-6, 1.0 - 1e-6)
    return _standard_normal_cdf(_standard_normal_ppf(price) - wedge_lambda)


def liquidity_adjusted_lambda(volume_usd: float, base_lambda: float = 0.176,
                              saturation_volume_usd: float = 10000.0) -> float:
    """
    Scales the pricing wedge down as a market gets more liquid, reaching zero at
    the volume tier where competitive trading has been measured to remove it.
    """
    if volume_usd <= 0:
        return base_lambda
    if volume_usd >= saturation_volume_usd:
        return 0.0
    return base_lambda * (1.0 - volume_usd / saturation_volume_usd)


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _standard_normal_ppf(p: float) -> float:
    """
    Inverse standard normal CDF via Acklam's rational approximation, accurate to
    about 1.15e-9 across the range, which is far finer than any input we have.
    """
    p = _clamp(p, 1e-12, 1.0 - 1e-12)

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
