#!/usr/bin/env python3
"""
Side-by-side comparison of the flat-percentage friction model and the exact one.

Run with:  python3 compare_models.py

The point of this script is to show where the two models disagree and which one
is right, rather than to assert that the new one is better. Every number here is
derived from the published fee formula, fee = C x feeRate x p x (1 - p).
"""

from polymarket_edge import (
    CalibrationTracker,
    EdgeEvaluator,
    NegRiskScanner,
    PolymarketFeeModel,
    walk_order_book,
)

FLAT_FEE_PCT = 0.015      # what the tracker assumed
FLAT_SLIPPAGE_PCT = 0.005


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def compare_fee_models():
    rule("1. Fee as a share of notional: flat assumption vs published formula")
    print("Polymarket charges fee = C x feeRate x p x (1-p). Against the dollars")
    print("deployed that simplifies to feeRate x (1-p), so it collapses towards")
    print("zero as the price approaches 1.00.\n")

    sports = PolymarketFeeModel("sports")
    politics = PolymarketFeeModel("politics")
    geo = PolymarketFeeModel("geopolitics")

    print(f"{'price':>7}{'flat 1.5%':>12}{'sports':>10}{'politics':>10}{'geopolitics':>13}{'error at sports':>18}")
    for price in (0.10, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.99):
        real = sports.fee_as_fraction_of_notional(price)
        ratio = FLAT_FEE_PCT / real if real > 0 else float("inf")
        print(f"{price:>7.2f}{FLAT_FEE_PCT * 100:>11.2f}%{real * 100:>9.3f}%"
              f"{politics.fee_as_fraction_of_notional(price) * 100:>9.3f}%"
              f"{geo.fee_as_fraction_of_notional(price) * 100:>12.3f}%"
              f"{ratio:>16.1f}x")

    print("\nMakers are never charged, in any category:")
    print(f"  taking at 0.50 (sports): {sports.fee_as_fraction_of_notional(0.50) * 100:.3f}% of notional")
    print(f"  making at 0.50 (sports): {sports.fee_as_fraction_of_notional(0.50, is_taker=False) * 100:.3f}% of notional")


def compare_decisions():
    rule("2. Where the two models disagree about the same trade")
    print("Old rule: accept when (fair - price(1+slippage)) - flat fees > 5%.")
    print("New rule: accept when the edge survives the real fee, beats the return")
    print("this capital could earn elsewhere, and leaves a margin the model can")
    print("actually resolve.\n")

    evaluator = EdgeEvaluator(
        fee_model=PolymarketFeeModel("sports"),
        min_ev_per_dollar=0.02,
        hurdle_apr=0.15,
        min_edge_over_breakeven=0.02,
    )

    cases = [
        ("stadium goal, book has not moved", 0.45, 0.78, 0.2),
        ("near-certain favourite, resolves tomorrow", 0.97, 0.995, 1.0),
        ("near-certain favourite, resolves in 6 months", 0.97, 0.995, 180.0),
        ("high probability, no real edge", 0.97, 0.975, 30.0),
        ("coin flip with a genuine read", 0.50, 0.58, 7.0),
        ("cheap longshot that looks mispriced", 0.10, 0.14, 60.0),
    ]

    print(f"{'scenario':<42}{'old':>7}{'new':>7}  {'why they differ'}")
    for label, price, fair, days in cases:
        old_entry = price * (1 + FLAT_SLIPPAGE_PCT)
        old_fees = old_entry * FLAT_FEE_PCT + 1.0 * FLAT_FEE_PCT
        old_accept = (fair - old_entry) - old_fees > 0.05

        new = evaluator.assess(price, fair, bankroll=10000.0, days_to_resolution=days)

        if old_accept == new.accepted:
            note = "agree"
        elif new.accepted:
            note = "old model over-charged fees at this price"
        else:
            note = new.reasons[0]

        print(f"{label:<42}{_yn(old_accept):>7}{_yn(new.accepted):>7}  {note}")

    rule("3. The same contract at different horizons")
    print("Buying at 0.97 something worth 0.995. The edge never changes; only how")
    print("long the capital is trapped does. A flat edge number cannot tell these")
    print("apart, which is how a 'safe' position quietly underperforms cash.\n")
    for days in (0.5, 2, 7, 30, 90, 365):
        assessment = evaluator.assess(0.97, 0.995, bankroll=10000.0, days_to_resolution=days)
        annualised = assessment.annualised_return
        shown = ">10000" if annualised >= 100.0 else f"{annualised * 100:.1f}"
        print(f"  {days:>6.1f} days  ->  {assessment.ev_per_dollar * 100:>5.2f}% on capital, "
              f"{shown:>10}% annualised   "
              f"{'accept' if assessment.accepted else 'reject'}")


def compare_position_sizing():
    rule("4. Position sizing: fixed 10% cap vs fractional Kelly")
    print("The tracker stakes min(cash x 10%, $100) on every accepted signal, so a")
    print("2% edge and a 40% edge get identical capital. Kelly scales with the edge")
    print("and with how much room the price leaves, f* = (q - p) / (1 - p).\n")

    evaluator = EdgeEvaluator(fee_model=PolymarketFeeModel("sports"), kelly_fraction=0.25)
    bankroll = 10000.0

    print(f"{'price':>7}{'fair':>7}{'edge':>8}{'old stake':>12}{'Kelly stake':>13}{'verdict':>10}")
    for price, fair in ((0.45, 0.78), (0.45, 0.50), (0.90, 0.95), (0.97, 0.99), (0.97, 0.975)):
        assessment = evaluator.assess(price, fair, bankroll=bankroll, days_to_resolution=1.0)
        old_stake = min(bankroll * 0.10, 100.0)
        stake = assessment.recommended_stake_usd if assessment.accepted else 0.0
        print(f"{price:>7.2f}{fair:>7.3f}{(fair - price) * 100:>7.1f}pp"
              f"${old_stake:>10.2f}${stake:>12.2f}"
              f"{'accept' if assessment.accepted else 'reject':>10}")

    print("\nKelly is unstable exactly where the price is highest: at 0.97 the")
    print("denominator is 0.03, so a one-point error in the estimate swings the")
    print("recommendation by a third of the bankroll. That is why the estimate is")
    print("shrunk towards the market price and the result is quarter-Kelly capped.")


def compare_slippage():
    rule("5. Slippage: flat 0.5% vs walking the actual book")
    print("Prediction market books are thin. A percentage assumption prices every")
    print("order size the same, which is only true for the smallest of them.\n")

    book = [(0.45, 200), (0.46, 300), (0.48, 500), (0.52, 1000), (0.60, 5000)]
    print(f"{'budget':>9}{'flat 0.5% price':>18}{'real avg fill':>16}{'real slippage':>16}")
    for budget in (50, 200, 500, 2000):
        fill = walk_order_book(book, budget)
        flat = 0.45 * (1 + FLAT_SLIPPAGE_PCT)
        real_slippage = (fill.average_price / 0.45 - 1.0) if fill.shares else 0.0
        print(f"${budget:>8.0f}{flat:>17.4f}{fill.average_price:>16.4f}{real_slippage * 100:>15.2f}%")

    print("\nA $2000 order pays roughly ten times the assumed slippage. Any edge")
    print("smaller than that gap is imaginary at that size.")


def demo_negrisk():
    rule("6. Structural edge that needs no forecast at all")
    print("In a mutually exclusive multi-outcome market exactly one outcome pays")
    print("1.00, so the YES prices must sum to 1.00. When they do not, the gap is")
    print("collectable regardless of who wins.\n")

    scanner = NegRiskScanner(PolymarketFeeModel("politics"), min_net_margin=0.01)

    markets = {
        "10-way field, under-summed": {f"c{i}": p for i, p in enumerate(
            [0.31, 0.24, 0.13, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])},
        "6-way field, over-summed": {f"c{i}": p for i, p in enumerate(
            [0.40, 0.28, 0.15, 0.10, 0.06, 0.05])},
        "fairly priced": {"a": 0.60, "b": 0.40},
    }

    for label, prices in markets.items():
        as_taker = scanner.scan(label, prices, is_taker=True)
        as_maker = scanner.scan(label, prices, is_taker=False)
        print(f"  {label:<30} sum={as_taker.price_sum:.3f}")
        print(f"  {'  as taker:':<30} {'TRADE' if as_taker.tradeable else 'skip':<6} {as_taker.note}")
        print(f"  {'  as maker:':<30} {'TRADE' if as_maker.tradeable else 'skip':<6} {as_maker.note}")

    print("\nEvery leg carries its own taker fee, so a wide field can be structurally")
    print("mispriced and still untradeable if you cross the spread on all of it. The")
    print("same basket assembled with resting limit orders pays nothing.")


def demo_calibration():
    rule("7. The feedback loop the tracker does not have")
    print("Six strategies produce confidence scores and nothing ever checks them")
    print("against outcomes. Brier score makes a bad strategy visible: 0.25 is what")
    print("answering 0.50 to everything scores, so anything above that is noise.\n")

    tracker = CalibrationTracker(min_samples=20)

    # A strategy that is genuinely informative: confident and usually right.
    for i in range(40):
        tracker.record("sports_latency_arbitrage", 0.90, 1 if i % 10 != 0 else 0)
    # A strategy that is confident and wrong about as often as not.
    for i in range(40):
        tracker.record("whale_overreaction", 0.85, 1 if i % 2 == 0 else 0)
    # Too new to judge.
    for i in range(5):
        tracker.record("sentiment_divergence", 0.70, 1)

    print(tracker.report())
    print(f"\nStrategies that should stop sizing positions: {tracker.underperformers()}")


def _yn(value):
    return "buy" if value else "skip"


if __name__ == "__main__":
    compare_fee_models()
    compare_decisions()
    compare_position_sizing()
    compare_slippage()
    demo_negrisk()
    demo_calibration()
    print()
