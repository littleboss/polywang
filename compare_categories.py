#!/usr/bin/env python3
"""
What changes when the bot trades US macro and crypto instead of sports.

Run with:  python3 compare_categories.py

Every number here comes from the published fee schedule or from the pricing
functions under test, so the strategy claims can be checked rather than taken
on trust.
"""

import math
import random

from polymarket_edge import EdgeEvaluator, NegRiskScanner, PolymarketFeeModel
from polymarket_crypto import (
    RealisedVolatility,
    probability_above,
    probability_band,
    round_trip_cost_fraction,
)
from polymarket_macro import ScheduledEventGuard, SecondaryRepricingWatcher

CATEGORIES = [("sports", "Sports"), ("us_macro", "US macro"), ("btc", "Crypto (BTC/ETH)"),
              ("politics", "Politics"), ("geopolitics", "Geopolitics")]


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def costs_by_category():
    rule("1. What each category costs, as a share of the money deployed")
    print("Fee = rate x (1 - price). US macro settles under Economics at 0.05, the")
    print("same as sports. Crypto is the most expensive category on the venue.\n")

    models = {label: PolymarketFeeModel(key) for key, label in CATEGORIES}
    print(f"{'price':>7}" + "".join(f"{label:>18}" for _, label in CATEGORIES))
    for price in (0.10, 0.30, 0.50, 0.70, 0.90, 0.97):
        row = f"{price:>7.2f}"
        for _, label in CATEGORIES:
            row += f"{models[label].fee_as_fraction_of_notional(price) * 100:>17.3f}%"
        print(row)

    print("\nRound trip at a price of 0.50 (enter and exit, both crossing the book):")
    for key, label in CATEGORIES:
        rate = PolymarketFeeModel(key).taker_fee_rate
        print(f"  {label:<20} {round_trip_cost_fraction(rate, 0.50) * 100:>6.2f}%  "
              f"= the probability move needed before a scalp earns anything")


def crypto_strategy_shift():
    rule("2. Crypto: the fair value comes from spot, not from a goal model")
    print("A market asking 'BTC above $120,000 on Friday' is a digital call. The")
    print("underlying is observable to the cent, so the probability follows from it")
    print("and a volatility estimate rather than from anything that has to be")
    print("inferred the way a football scoreline does.\n")

    spot, strike = 118_000.0, 120_000.0
    print(f"Spot ${spot:,.0f}, strike ${strike:,.0f}, 60% annualised volatility:\n")
    print(f"{'days left':>11}{'fair value':>13}{'vol band (+/-25%)':>22}{'usable?':>10}")
    for days in (0.5, 2, 7, 30, 90):
        band = probability_band(spot, strike, 0.60, days)
        confidence = band.confidence_against(band.point - 0.05)
        print(f"{days:>11.1f}{band.point:>13.3f}"
              f"{f'{band.low:.3f} - {band.high:.3f}':>22}"
              f"{'yes' if confidence > 0.3 else 'too noisy':>10}")

    print("\nThe band is the point. Volatility is estimated, not observed, and near")
    print("the money a 25% error in it moves the answer further than the edge does.")
    print("When that happens the model cannot tell a mispriced contract from its own")
    print("noise, so the position size it justifies is zero.")

    rule("3. Crypto: what the 0.07 rate rules out")
    crypto = EdgeEvaluator(fee_model=PolymarketFeeModel("btc"))
    macro = EdgeEvaluator(fee_model=PolymarketFeeModel("us_macro"))

    print("Break-even probability, the accuracy needed just to cover the fee:\n")
    print(f"{'price':>7}{'US macro':>12}{'crypto':>10}{'extra accuracy needed':>25}")
    for price in (0.20, 0.50, 0.80, 0.95):
        m = macro.breakeven_probability(price)
        c = crypto.breakeven_probability(price)
        print(f"{price:>7.2f}{m:>12.4f}{c:>10.4f}{(c - m) * 100:>23.2f}pp")

    print("\nSame trade, both categories, buying at 0.50 something worth 0.535:")
    for label, evaluator in (("US macro", macro), ("Crypto  ", crypto)):
        assessment = evaluator.assess(0.50, 0.535, bankroll=10000, days_to_resolution=3)
        verdict = "accept" if assessment.accepted else "reject"
        print(f"  {label}: {verdict:<7} ({assessment.ev_per_dollar * 100:+.2f}% on capital)")

    print("\nWorking the same edge with a resting order instead of crossing:")
    for label, key in (("US macro", "us_macro"), ("Crypto  ", "btc")):
        model = PolymarketFeeModel(key)
        rebate = model.maker_rebate_usd(100, 0.50)
        print(f"  {label}: fee $0.00 on 100 shares, plus a ${rebate:.2f} rebate "
              f"({model.maker_rebate_share * 100:.0f}% of what the taker paid)")


def macro_strategy_shift():
    rule("4. US macro: the latency trade does not transfer")
    print("A goal is a surprise in time, so a fast feed has seconds of advantage. A")
    print("CPI print is scheduled to the second and everyone is watching the same")
    print("clock, so there is no interval where we know something the market does")
    print("not. There is only a race decided in microseconds, and a Python process")
    print("polling an endpoint is structurally the slow side of it.\n")

    guard = ScheduledEventGuard(blackout_seconds_before=120, blackout_seconds_after=300)
    release = 1_000_000.0
    guard.schedule("cpi", release)

    for offset, label in ((-600, "10 min before"), (-60, "1 min before"),
                          (5, "5 s after"), (120, "2 min after"), (400, "7 min after")):
        status = guard.status(release + offset)
        state = "SUSPENDED" if status.in_blackout else "trading"
        print(f"  {label:<16} {state:<11} {status.reason[:78]}")

    print("\nThe window after the print is longer than the window before it. Being")
    print("slow after the number lands is the expensive direction: any fill we get")
    print("there is one a faster participant looked at and declined.")

    rule("5. US macro: what is left once the race is conceded")
    print("The market asking the question directly resolves instantly and is")
    print("unplayable. The markets that merely depend on it get re-reasoned at human")
    print("speed, which is a window measured in minutes.\n")

    watcher = SecondaryRepricingWatcher(open_after_seconds=30, close_after_seconds=900, min_edge=0.04)
    watcher.record_release("cpi", release)
    for offset in (5, 120, 600, 1200):
        result = watcher.evaluate("fed-cuts-in-march", "cpi", 0.30, 0.45, now=release + offset)
        if result:
            print(f"  +{offset:>4}s  signal: {result.edge * 100:+.0f}pp on '{result.market_id}'")
        else:
            reason = "still in the microsecond race" if offset < 30 else "edge is common knowledge by now"
            print(f"  +{offset:>4}s  no signal ({reason})")

    print("\nFed target-range ladders are mutually exclusive, so their prices must sum")
    print("to 1.00. That is a structural edge needing no forecast at all:\n")

    ladder = {"3.50-3.75": 0.08, "3.75-4.00": 0.55, "4.00-4.25": 0.34, "4.25-4.50": 0.09}
    scanner = NegRiskScanner(PolymarketFeeModel("us_macro"), min_net_margin=0.01)
    for is_taker, label in ((True, "crossing the book"), (False, "resting orders")):
        opportunity = scanner.scan("fed-ladder", ladder, is_taker=is_taker)
        print(f"  sum={opportunity.price_sum:.3f}  {label:<20} "
              f"{'TRADE' if opportunity.tradeable else 'skip ':<6} {opportunity.note}")


def volatility_demo():
    rule("6. Volatility has to be measured, and the measurement decides the size")
    print("The estimator recovers the regime from a price stream. Spot and strike are")
    print("held fixed below so the only thing changing is the measured volatility.\n")

    random.seed(3)
    spot, strike = 105_000.0, 100_000.0

    print(f"Spot ${spot:,.0f}, strike ${strike:,.0f}, 7 days left. A market price of 0.60:\n")
    print(f"{'regime':<14}{'measured vol':>14}{'fair':>8}{'band':>16}{'width':>9}{'confidence':>13}")
    for label, step in (("calm", 0.0004), ("normal", 0.0010), ("stressed", 0.0030)):
        prices = [spot]
        for _ in range(2000):
            prices.append(prices[-1] * math.exp(random.choice([step, -step])))

        estimator = RealisedVolatility(sample_interval_seconds=60, window=2000, minimum_samples=30)
        estimator.extend(prices)
        vol = estimator.annualised()

        band = probability_band(spot, strike, vol, days_to_expiry=7.0)
        print(f"  {label:<12}{vol * 100:>13.1f}%{band.point:>8.3f}"
              f"{f'{band.low:.3f}-{band.high:.3f}':>16}"
              f"{band.width * 100:>8.1f}pp{band.confidence_against(0.60) * 100:>12.0f}%")

    print("\nA wider band is not a worse forecast, it is an honest one. Confidence")
    print("feeds straight into the Kelly stake, so the same nominal edge buys a")
    print("smaller position in a market the model cannot pin down.")


if __name__ == "__main__":
    costs_by_category()
    crypto_strategy_shift()
    macro_strategy_shift()
    volatility_demo()
    print()
