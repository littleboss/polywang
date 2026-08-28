#!/usr/bin/env python3
"""
US macro markets: CPI, payrolls, FOMC, GDP.

These settle under the Economics fee schedule (0.05 taker rate, same as sports).
The fee arithmetic therefore carries over unchanged. The strategy does not.

Why the latency trade does not transfer
---------------------------------------
The sports engine works because a goal is a surprise in time. Nobody knows the
minute it will arrive, so the feed that sees it first has seconds of advantage
over a book that has not been told yet.

A CPI print is the opposite. It lands at 08:30:00 Eastern, to the second, on a
date published months ahead, and every participant is watching the same clock.
There is no interval during which we know something the market does not; there is
only a race, measured in microseconds and won by whoever is closest to the
matching engine. A Python process polling an HTTP endpoint is structurally the
slow side of that race.

Being slow in a race is worse than not entering it. A resting order during a
release gets filled precisely when the number went against it, and a marketable
order fills only when someone faster is happy to take the other side. Both are
adverse selection, and the expected value is negative before fees.

So the correct behaviour around a scheduled release is to stand aside, and the
module's main export is a guard that enforces that.

What is left once the race is conceded
--------------------------------------
Two things, both slower and both real:

`SecondaryRepricingWatcher` covers the derived markets. When CPI prints, the
market asking about CPI resolves almost instantly, but the markets that merely
depend on it — "Fed cuts in March", "recession this year" — reprice over minutes
as people work out the implication. That is a human-speed inference, not a
latency race, and it is the same shape as the trade the sports engine makes.

Fed rate ladders are multi-outcome NegRisk markets: mutually exclusive target
ranges whose prices must sum to 1.00. `NegRiskScanner` in polymarket_edge already
handles those, and they need no forecast at all.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import time

# Recurring US releases and the volatility they inject. The window is how long
# the market takes to settle down afterwards, not how long the number takes to
# print.
KNOWN_RELEASES: Dict[str, str] = {
    "cpi": "Consumer Price Index, 08:30 ET",
    "core_cpi": "Core CPI, 08:30 ET",
    "pce": "PCE price index, 08:30 ET",
    "nfp": "Non-farm payrolls, 08:30 ET",
    "unemployment": "Unemployment rate, 08:30 ET",
    "gdp": "GDP advance estimate, 08:30 ET",
    "fomc": "FOMC rate decision, 14:00 ET",
    "fomc_minutes": "FOMC minutes, 14:00 ET",
    "jolts": "JOLTS job openings, 10:00 ET",
    "retail_sales": "Retail sales, 08:30 ET",
}


@dataclass
class ScheduledRelease:
    """One scheduled data release."""
    name: str
    timestamp: float
    description: str = ""

    def seconds_until(self, now: Optional[float] = None) -> float:
        return self.timestamp - (time.time() if now is None else now)


@dataclass
class BlackoutStatus:
    """Whether trading should be suspended, and why."""
    in_blackout: bool
    release: Optional[ScheduledRelease] = None
    seconds_to_release: float = 0.0
    reason: str = ""


class ScheduledEventGuard:
    """
    Suspends trading around scheduled macro releases.

    The window is deliberately asymmetric. Beforehand it only needs to be long
    enough that a resting order cannot still be sitting there when the number
    lands, since positioning into a release is a forecast rather than an edge.
    Afterwards it needs to be long enough for the fast money to finish, because
    that is the period in which our fills are the ones nobody faster wanted.

    Setting `after` too short is the expensive mistake: it re-enables trading
    while the book is still being cleaned out, which is the single worst moment
    to be the slowest participant in a market.
    """

    def __init__(self,
                 blackout_seconds_before: float = 120.0,
                 blackout_seconds_after: float = 300.0,
                 releases: Optional[Sequence[ScheduledRelease]] = None):
        self.blackout_seconds_before = blackout_seconds_before
        self.blackout_seconds_after = blackout_seconds_after
        self.releases: List[ScheduledRelease] = list(releases or [])

    def schedule(self, name: str, timestamp: float, description: str = "") -> ScheduledRelease:
        release = ScheduledRelease(
            name=name,
            timestamp=timestamp,
            description=description or KNOWN_RELEASES.get(name.strip().lower(), ""),
        )
        self.releases.append(release)
        return release

    def status(self, now: Optional[float] = None) -> BlackoutStatus:
        now = time.time() if now is None else now

        for release in self.releases:
            delta = now - release.timestamp
            if -self.blackout_seconds_before <= delta <= self.blackout_seconds_after:
                if delta < 0:
                    reason = (
                        f"{abs(delta):.0f}s before {release.name}. Holding a position into a scheduled "
                        f"release is a forecast, not an edge, and a resting order here will be filled "
                        f"by whoever reads the number first."
                    )
                else:
                    reason = (
                        f"{delta:.0f}s after {release.name}. The repricing race is decided in microseconds "
                        f"by colocated systems; any fill we get in this window is one a faster participant "
                        f"declined."
                    )
                return BlackoutStatus(
                    in_blackout=True,
                    release=release,
                    seconds_to_release=-delta,
                    reason=reason,
                )

        upcoming = [r for r in self.releases if r.timestamp > now]
        if upcoming:
            nearest = min(upcoming, key=lambda r: r.timestamp)
            return BlackoutStatus(
                in_blackout=False,
                release=nearest,
                seconds_to_release=nearest.timestamp - now,
                reason=f"clear; next release {nearest.name} in {(nearest.timestamp - now) / 60.0:.0f} min",
            )

        return BlackoutStatus(in_blackout=False, reason="clear; no releases scheduled")

    def is_tradeable(self, now: Optional[float] = None) -> bool:
        return not self.status(now).in_blackout

    def prune(self, now: Optional[float] = None) -> None:
        """Drops releases whose blackout has fully passed."""
        now = time.time() if now is None else now
        self.releases = [r for r in self.releases
                         if now - r.timestamp <= self.blackout_seconds_after]


@dataclass
class SecondaryRepricing:
    """A dependent market that has not yet absorbed a released number."""
    market_id: str
    driver: str
    market_price: float
    implied_probability: float
    edge: float
    seconds_since_release: float

    def describe(self) -> str:
        return (
            f"{self.driver} has printed and '{self.market_id}' still trades at {self.market_price:.3f} "
            f"against an implied {self.implied_probability:.3f} ({self.edge * 100:+.1f}pp), "
            f"{self.seconds_since_release:.0f}s after the release."
        )


class SecondaryRepricingWatcher:
    """
    Finds markets that depend on a release but have not yet absorbed it.

    The market that asks the question directly resolves the instant the number
    prints and is unplayable. The market that merely depends on it has to be
    re-reasoned, and people do that at human speed. "CPI above 3%" is settled;
    "Fed cuts in March" takes minutes to agree on.

    The window is bounded on both sides. Too early and we are back in the
    microsecond race; too late and the inference is common knowledge. The default
    opens once the immediate reaction is over and closes when the edge should
    reasonably have been competed away.
    """

    def __init__(self,
                 open_after_seconds: float = 30.0,
                 close_after_seconds: float = 900.0,
                 min_edge: float = 0.04):
        self.open_after_seconds = open_after_seconds
        self.close_after_seconds = close_after_seconds
        self.min_edge = min_edge
        self._releases: Dict[str, float] = {}

    def record_release(self, driver: str, timestamp: Optional[float] = None) -> None:
        self._releases[driver] = time.time() if timestamp is None else timestamp

    def evaluate(self, market_id: str, driver: str, market_price: float,
                 implied_probability: float, now: Optional[float] = None) -> Optional[SecondaryRepricing]:
        released_at = self._releases.get(driver)
        if released_at is None:
            return None

        now = time.time() if now is None else now
        elapsed = now - released_at
        if elapsed < self.open_after_seconds or elapsed > self.close_after_seconds:
            return None

        edge = implied_probability - market_price
        if abs(edge) < self.min_edge:
            return None

        return SecondaryRepricing(
            market_id=market_id,
            driver=driver,
            market_price=market_price,
            implied_probability=implied_probability,
            edge=edge,
            seconds_since_release=elapsed,
        )


def fed_rate_ladder_outcomes(prices: Dict[str, float]) -> Tuple[float, float]:
    """
    Sum of a Fed target-range ladder and its deviation from 1.00.

    The ranges are mutually exclusive and exhaustive, so the prices must sum to
    one. Retail flow piles into the one or two ranges everyone expects and leaves
    the tails thin, which is what pushes the sum off. Feed the deviation to
    NegRiskScanner to find out whether it survives per-leg fees, since a ladder
    with many rungs usually only clears them for a maker.
    """
    total = sum(prices.values())
    return total, total - 1.0
