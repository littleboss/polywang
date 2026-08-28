"""Wallet-aware whale flow telemetry for Polymarket.

This module is deliberately signal-only.  It does not place orders and it
never treats an anonymous or malformed address as a distinct wallet.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import os
import re
import tempfile
import time
from typing import Deque, Dict, Optional


ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
ANONYMOUS_ADDRESSES = {"", "0x0", "0x0000000000000000000000000000000000000000", "unknown", "anonymous"}


def normalize_wallet(value: object) -> str:
    wallet = str(value or "").strip()
    if wallet.lower() in ANONYMOUS_ADDRESSES or not ADDRESS_RE.fullmatch(wallet):
        return ""
    return wallet.lower()


@dataclass(frozen=True)
class WhaleTrade:
    trade_id: str
    market_id: str
    outcome: str
    wallet: str
    side: str
    price: float
    size: float
    timestamp: float
    tx_hash: str = ""

    @property
    def notional_usd(self) -> float:
        return max(0.0, self.price * self.size)

    @classmethod
    def from_payload(cls, payload: dict) -> Optional["WhaleTrade"]:
        wallet = normalize_wallet(payload.get("wallet_address", payload.get("wallet", payload.get("user"))))
        if not wallet:
            return None
        try:
            market_id = str(payload.get("market_id", payload.get("market", "")))
            outcome = str(payload.get("outcome", ""))
            side = str(payload.get("side", "")).upper()
            price = float(payload.get("price", 0.0))
            size = float(payload.get("size", 0.0))
            timestamp = float(payload.get("timestamp", time.time()))
        except (TypeError, ValueError):
            return None
        if not market_id or not (0.0 < price < 1.0) or size <= 0.0 or side not in {"BUY", "SELL"}:
            return None
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        trade_id = str(payload.get("trade_id", payload.get("tradeId", payload.get("id", ""))))
        tx_hash = str(payload.get("tx_hash", payload.get("transaction_hash", "")))
        if not trade_id:
            raw = f"{market_id}|{outcome}|{wallet}|{side}|{price:.12f}|{size:.12f}|{timestamp:.6f}|{tx_hash}"
            trade_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return cls(trade_id, market_id, outcome, wallet, side, price, size, timestamp, tx_hash)


@dataclass
class WhaleSignal:
    trade_id: str
    market_id: str
    wallet: str
    side: str
    notional_usd: float
    wallet_quality: float
    settled_markets: int
    market_pressure: float
    unique_wallets: int
    confidence: float
    eligible: bool
    reason: str
    coordination: Optional["CoordinationSignal"] = None


@dataclass
class CoordinationSignal:
    market_id: str
    outcome: str
    side: str
    unique_wallets: int
    qualified_wallets: int
    total_notional_usd: float
    pressure: float
    max_wallet_share: float
    confidence: float
    eligible: bool
    reason: str


class WhaleIntelligenceEngine:
    """Persist wallet outcomes and score quality-weighted recent flow."""

    def __init__(self, path: str = "", threshold_usd: float = 5000.0,
                 window_seconds: float = 60.0, min_unique_wallets: int = 3,
                 min_settled_markets: int = 20, min_quality: float = 0.58,
                 min_pressure: float = 0.60, min_qualified_wallets: int = 1,
                 max_concentration: float = 0.75,
                 min_coordination_trade_usd: Optional[float] = None):
        self.path = path
        self.threshold_usd = max(0.0, float(threshold_usd))
        self.window_seconds = max(1.0, float(window_seconds))
        self.min_unique_wallets = max(1, int(min_unique_wallets))
        self.min_settled_markets = max(1, int(min_settled_markets))
        self.min_quality = min(1.0, max(0.5, float(min_quality)))
        self.min_pressure = min(1.0, max(0.0, float(min_pressure)))
        self.min_qualified_wallets = max(1, int(min_qualified_wallets))
        self.max_concentration = min(1.0, max(0.0, float(max_concentration)))
        self.min_coordination_trade_usd = max(
            0.0, float(min_coordination_trade_usd)
            if min_coordination_trade_usd is not None else self.threshold_usd * 0.10
        )
        self.state = {"wallets": {}, "seen_trade_ids": [], "anonymous_events": 0}
        self.market_flows: Dict[str, Deque[WhaleTrade]] = defaultdict(deque)
        self.load()

    def load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("wallets"), dict):
                raise ValueError("missing wallets")
            self.state.update(loaded)
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Whale state is unreadable: {self.path}") from error

    def save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".whale-state-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
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

    def _wallet(self, wallet: str) -> dict:
        return self.state["wallets"].setdefault(wallet, {
            "trades": 0, "volume_usd": 0.0, "settled_markets": 0,
            "wins": 0, "losses": 0, "realized_pnl": 0.0,
            "positions": {},
        })

    def wallet_quality(self, wallet: str) -> float:
        stats = self._wallet(wallet)
        settled = int(stats["settled_markets"])
        posterior = (float(stats["wins"]) + 2.0) / (settled + 4.0)
        volume = max(1.0, float(stats["volume_usd"]))
        roi_component = min(1.0, max(0.0, 0.5 + float(stats["realized_pnl"]) / (2.0 * volume)))
        return min(1.0, max(0.0, 0.7 * posterior + 0.3 * roi_component))

    def _prune(self, market_id: str, now: float) -> None:
        flow = self.market_flows[market_id]
        cutoff = now - self.window_seconds
        while flow and flow[0].timestamp < cutoff:
            flow.popleft()

    def coordination_signal(self, market_id: str, now: Optional[float] = None) -> Optional[CoordinationSignal]:
        """Evaluate same-market, same-outcome, same-direction wallet flow."""
        now = time.time() if now is None else float(now)
        self._prune(market_id, now)
        groups: Dict[tuple[str, str], List[WhaleTrade]] = defaultdict(list)
        for trade in self.market_flows[market_id]:
            if trade.notional_usd >= self.min_coordination_trade_usd:
                groups[(trade.outcome.strip().lower(), trade.side)].append(trade)
        if not groups:
            return None
        group = max(groups.values(), key=lambda items: sum(item.notional_usd for item in items))
        wallets: Dict[str, float] = defaultdict(float)
        for trade in group:
            wallets[trade.wallet] += trade.notional_usd
        unique = len(wallets)
        total = sum(wallets.values())
        top_share = max(wallets.values()) / total if total else 1.0
        qualified = sum(
            1 for wallet in wallets
            if self._wallet(wallet)["settled_markets"] >= self.min_settled_markets
            and self.wallet_quality(wallet) >= self.min_quality
        )
        eligible = (
            unique >= self.min_unique_wallets
            and total >= self.threshold_usd
            and qualified >= self.min_qualified_wallets
            and top_share <= self.max_concentration
        )
        confidence = min(0.99, max(0.0,
            0.35 * min(1.0, unique / max(1, self.min_unique_wallets * 2))
            + 0.25 * min(1.0, total / max(1.0, self.threshold_usd * 2))
            + 0.20 * (1.0 - top_share)
            + 0.20 * min(1.0, qualified / max(1, self.min_qualified_wallets * 2))
        ))
        reason = "independent same-direction flow passed size, concentration, and history checks" if eligible else (
            "observation only: coordination needs same-direction size, independent wallets, low concentration, "
            "and at least one historically qualified wallet"
        )
        return CoordinationSignal(
            market_id=market_id, outcome=group[0].outcome, side=group[0].side,
            unique_wallets=unique, qualified_wallets=qualified,
            total_notional_usd=total, pressure=1.0 if group[0].side == "BUY" else -1.0,
            max_wallet_share=top_share, confidence=confidence,
            eligible=eligible, reason=reason,
        )

    def _update_position(self, trade: WhaleTrade) -> None:
        wallet = self._wallet(trade.wallet)
        market = wallet["positions"].setdefault(trade.market_id, {})
        position = market.setdefault(trade.outcome, {"shares": 0.0, "cost": 0.0, "proceeds": 0.0})
        notional = trade.notional_usd
        if trade.side == "BUY":
            position["shares"] += trade.size
            position["cost"] += notional
        else:
            position["shares"] -= trade.size
            position["proceeds"] += notional

    def record_trade(self, payload: dict | WhaleTrade) -> Optional[WhaleSignal]:
        trade = payload if isinstance(payload, WhaleTrade) else WhaleTrade.from_payload(payload)
        if trade is None:
            self.state["anonymous_events"] += 1
            self.save()
            return None
        seen = self.state["seen_trade_ids"]
        if trade.trade_id in seen:
            return None
        seen.append(trade.trade_id)
        del seen[:-10000]
        stats = self._wallet(trade.wallet)
        stats["trades"] += 1
        stats["volume_usd"] += trade.notional_usd
        self._update_position(trade)
        flow = self.market_flows[trade.market_id]
        flow.append(trade)
        self._prune(trade.market_id, trade.timestamp)
        self.save()

        coordination = self.coordination_signal(trade.market_id, trade.timestamp)

        total = 0.0
        net = 0.0
        wallets = set()
        for item in flow:
            weight = 0.5 + self.wallet_quality(item.wallet)
            signed = item.notional_usd if item.side == "BUY" else -item.notional_usd
            total += item.notional_usd * weight
            net += signed * weight
            wallets.add(item.wallet)
        pressure = net / total if total else 0.0
        quality = self.wallet_quality(trade.wallet)
        settled = int(stats["settled_markets"])
        aligned = (trade.side == "BUY" and pressure >= self.min_pressure) or (
            trade.side == "SELL" and pressure <= -self.min_pressure
        )
        consensus_flow = coordination is not None and coordination.eligible
        solo_proven = settled >= self.min_settled_markets * 2 and quality >= min(1.0, self.min_quality + 0.08)
        eligible = (
            trade.notional_usd >= self.threshold_usd
            and settled >= self.min_settled_markets
            and quality >= self.min_quality
            and (consensus_flow or solo_proven)
            and abs(pressure) >= self.min_pressure
            and aligned
        )
        confidence = min(0.99, max(0.0,
            0.45 * quality + 0.35 * abs(pressure) +
            0.20 * min(1.0, len(wallets) / self.min_unique_wallets)))
        reason = "qualified wallet and aligned quality-weighted flow" if eligible else "observation only: quality, sample, flow, or size threshold not met"
        return WhaleSignal(
            trade_id=trade.trade_id, market_id=trade.market_id, wallet=trade.wallet,
            side=trade.side, notional_usd=trade.notional_usd,
            wallet_quality=quality, settled_markets=settled,
            market_pressure=pressure, unique_wallets=len(wallets),
            confidence=confidence, eligible=eligible, reason=reason,
            coordination=coordination,
        )

    def settle_market(self, market_id: str, winning_outcome: str) -> int:
        settled = 0
        winning = str(winning_outcome).strip().lower()
        for wallet, stats in self.state["wallets"].items():
            market = stats["positions"].pop(market_id, None)
            if not market:
                continue
            pnl = 0.0
            for outcome, position in market.items():
                payout = position["shares"] if str(outcome).strip().lower() == winning else 0.0
                pnl += position["proceeds"] + payout - position["cost"]
            stats["settled_markets"] += 1
            stats["realized_pnl"] += pnl
            stats["wins" if pnl > 0.0 else "losses"] += 1
            settled += 1
        if settled:
            self.save()
        return settled

    def snapshot(self, wallet: str) -> Optional[dict]:
        normalized = normalize_wallet(wallet)
        stats = self.state["wallets"].get(normalized)
        if not stats:
            return None
        result = dict(stats)
        result["quality"] = self.wallet_quality(normalized)
        return result
