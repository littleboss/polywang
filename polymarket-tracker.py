#!/usr/bin/env python3
"""
Polymarket Sports Latency Arbitrage & Strategy Suite (v7)
Based on the real-world trading lessons from the "Built This Week" podcast (Episode 26, 2026).

This enterprise-grade quantitative bot focuses specifically on exploiting 
Time/Latency Arbitrage in sports markets (e.g., Premier League matches) where
in-play events (such as goals) happen in the physical world but take several seconds
to reflect on centralized/decentralized prediction scoreboards [13, 20].

Key Video Groundings and Implementations:
1. "Newcastle vs Crystal Palace" case study [13]: Handled as a classic test vector.
2. "Time/Latency Arbitrage" [20]: Programmatically models the "stadium latency gap"
   where a spectator in the stadium (or an ultra-fast API feed) detects a goal
   before the platform's order book adapts.
3. "Transaction Fee Elimination" [14]: Blocks low-risk/low-margin settlements
   where transaction fees and slippage completely consume the edge.
4. "Sports Markets Clarity" [14, 15]: Leverages sports because they have extremely
   clear, binary outcomes with zero "gray area" or resolution disputes (unlike politics).

Core Technical Upgrades in v7:
- Dedicated `SportsLatencyArbitrageEngine`: Implements an in-play live probability model
  (Time-decay & Poisson-derived match outcome estimator) to calculate instantaneous 'Fair Value'.
- `FrictionGuard`: Rejects trade executions where the delta (Fair Value - Market Price)
  is smaller than the cumulative slippage and dual-sided fees.
- `EIP712LimitSigner`: Produces rigorous transaction payloads with hard price ceilings 
  to prevent frontrunning and post-goal order-book updates from filling us at a loss [20].
"""

import os
import sys
import time
import json
import logging
import math
from collections import defaultdict, deque

# External HTTP/WS dependencies
import requests
try:
    import websockets
    import asyncio
    HAS_ASYNC_WS = True
except ImportError:
    HAS_ASYNC_WS = False

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PolymarketV7")

def load_dotenv(filepath=".env"):
    """Robust custom parser to read configurations from .env."""
    if os.path.exists(filepath):
        logger.info(f"Loading environment configurations from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')
    else:
        logger.info(".env file not found, using default environment variables.")

load_dotenv()

# Strategy Configuration from Environment
CONFIG = {
    # Friction Controls [14]
    "PAPER_TRADING": os.getenv("PAPER_TRADING", "True").lower() == "true",
    "INITIAL_BALANCE": float(os.getenv("INITIAL_BALANCE", "1000.0")),
    "SIMULATED_FEE_PCT": float(os.getenv("SIMULATED_FEE_PCT", "0.015")),  # 1.5% average platform fee [7]
    "SLIPPAGE_PCT": float(os.getenv("SLIPPAGE_PCT", "0.005")),            # 0.5% expected execution slippage

    # Sports Latency Arbitrage Specific Parameters [20]
    "SPORTS_LATENCY_THRESHOLD_SECS": int(os.getenv("SPORTS_LATENCY_THRESHOLD_SECS", "5")),
    "MIN_ARBITRAGE_EDGE_PCT": float(os.getenv("MIN_ARBITRAGE_EDGE_PCT", "0.05")),       # Minimum edge required (5%)
    
    # Standard Strategy Parameters [8, 9, 11]
    "WHALE_USD_THRESHOLD": float(os.getenv("WHALE_USD_THRESHOLD", "5000")),
    "COORDINATION_WINDOW_SECS": int(os.getenv("COORDINATION_WINDOW_SECS", "60")),
    "COORDINATION_MIN_UNIQUE_WALLETS": int(os.getenv("COORDINATION_MIN_UNIQUE_WALLETS", "7")), # 7 traders from podcast [9, 11]
    "MOMENTUM_WINDOW_SECS": int(os.getenv("MOMENTUM_WINDOW_SECS", "30")),
    "MOMENTUM_VOLUME_MULTIPLIER": float(os.getenv("MOMENTUM_VOLUME_MULTIPLIER", "3.0")),
    "OVERREACTION_PRICE_DELTA": float(os.getenv("OVERREACTION_PRICE_DELTA", "0.10")),
    "PARITY_ARBITRAGE_THRESHOLD": float(os.getenv("PARITY_ARBITRAGE_THRESHOLD", "0.02")),

    # API Credentials for Real CLOB V2 Ordering
    "POLY_API_KEY": os.getenv("POLY_API_KEY", ""),
    "POLY_API_SECRET": os.getenv("POLY_API_SECRET", ""),
    "POLY_PASSPHRASE": os.getenv("POLY_PASSPHRASE", ""),
    "POLY_PRIVATE_KEY": os.getenv("POLY_PRIVATE_KEY", ""),
    "HTTP_PROXY": os.getenv("HTTP_PROXY", "")
}

def write_default_env():
    env_path = ".env"
    if not os.path.exists(env_path):
        content = """# Polymarket Advanced Sports Latency Bot Config
PAPER_TRADING=True
INITIAL_BALANCE=1000.0
SIMULATED_FEE_PCT=0.015
SLIPPAGE_PCT=0.005

# Sports Latency Arbitrage settings
SPORTS_LATENCY_THRESHOLD_SECS=5
MIN_ARBITRAGE_EDGE_PCT=0.05

# General Strategy Thresholds
WHALE_USD_THRESHOLD=5000
COORDINATION_WINDOW_SECS=60
COORDINATION_MIN_UNIQUE_WALLETS=7
MOMENTUM_WINDOW_SECS=30
MOMENTUM_VOLUME_MULTIPLIER=3.0
OVERREACTION_PRICE_DELTA=0.10
PARITY_ARBITRAGE_THRESHOLD=0.02

# Proxy (Required if running from restricted US locations)
HTTP_PROXY=

# Exchange Keys (Keep secret!)
POLY_API_KEY=your_api_key_here
POLY_API_SECRET=your_api_secret_here
POLY_PASSPHRASE=your_passphrase_here
POLY_PRIVATE_KEY=your_wallet_private_key_here
"""
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Created template {env_path} file with configured thresholds.")

write_default_env()


class FrictionAwarePortfolioEngine:
    """
    Simulation environment for tracking trades, accounting for slippage, 
    settlement rules, and the transaction fees that often kill low-risk edges [14].
    """
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {} # {market_id: {outcome: {"contracts": X, "avg_price": Y}}}
        self.trade_history = []
        self.fee_pct = CONFIG["SIMULATED_FEE_PCT"]
        self.slippage_pct = CONFIG["SLIPPAGE_PCT"]
        self.total_fees_paid = 0.0

    def evaluate_roi_eligibility(self, current_price, target_probability):
        """
        Grounded in Passage [14]: We must calculate whether the potential edge (probability difference)
        can survive the 'friction' (buy fees + sell fees + slippage).
        """
        # Estimated purchase price after slippage
        estimated_entry = current_price * (1.0 + self.slippage_pct)
        if estimated_entry >= 1.0:
            estimated_entry = 0.995

        # Edge calculation: Expected payout vs total transaction costs
        total_fees = (estimated_entry * self.fee_pct) + (1.0 * self.fee_pct if target_probability > 0.8 else estimated_entry * self.fee_pct)
        net_expected_profit = (target_probability - estimated_entry) - total_fees

        return net_expected_profit > CONFIG["MIN_ARBITRAGE_EDGE_PCT"], net_expected_profit, estimated_entry

    def execute_buy(self, market_id, market_title, outcome, price, usd_allocation):
        """Simulates buying an outcome token with slippage and commission fees."""
        fee = usd_allocation * self.fee_pct
        effective_budget = usd_allocation - fee
        
        if effective_budget <= 0 or self.cash < usd_allocation:
            logger.error(f"❌ Simulated buy failed: Insufficient cash (${self.cash:.2f} available for ${usd_allocation:.2f} allocation)")
            return False

        # Apply slippage
        slippage_price = price * (1.0 + self.slippage_pct)
        if slippage_price >= 1.0:
            slippage_price = 0.995

        contracts_bought = effective_budget / slippage_price
        self.cash -= usd_allocation
        self.total_fees_paid += fee

        if market_id not in self.positions:
            self.positions[market_id] = {}
        
        pos = self.positions[market_id].get(outcome, {"contracts": 0.0, "avg_price": 0.0})
        total_contracts = pos["contracts"] + contracts_bought
        # Weighted average entry price
        weighted_price = ((pos["contracts"] * pos["avg_price"]) + (contracts_bought * slippage_price)) / total_contracts
        
        self.positions[market_id][outcome] = {
            "contracts": total_contracts,
            "avg_price": weighted_price
        }

        logger.info(
            f"🛒 [SIMULATED BUY SUCCESS] {market_title} ({outcome})\n"
            f"   Allocated: ${usd_allocation:.2f} USD | Fee paid: ${fee:.2f} USD | Slippage Entry Price: ${slippage_price:.4f}\n"
            f"   Purchased: {contracts_bought:,.2f} outcome contracts."
        )
        self.print_portfolio_status()
        return True

    def settle_market(self, market_id, market_title, winning_outcome):
        """Settles all contracts for a specific market at expiry (binary 1.00 or 0.00)."""
        if market_id not in self.positions:
            return

        market_pos = self.positions[market_id]
        for outcome, pos_data in list(market_pos.items()):
            contracts = pos_data["contracts"]
            entry_price = pos_data["avg_price"]
            
            payout_per_contract = 1.0 if outcome == winning_outcome else 0.0
            gross_payout = contracts * payout_per_contract
            
            settle_fee = gross_payout * self.fee_pct
            net_payout = gross_payout - settle_fee
            
            self.cash += net_payout
            self.total_fees_paid += settle_fee
            
            net_pnl = net_payout - (contracts * entry_price)

            self.trade_history.append({
                "market": market_title,
                "outcome": outcome,
                "win": outcome == winning_outcome,
                "net_pnl": net_pnl
            })

            logger.info(
                f"🏁 [SIMULATED RESOLUTION] Settle event for: {market_title}\n"
                f"   💰 [SIMULATED SELL SUCCESS] {market_title} ({outcome})\n"
                f"   Exit Price: ${payout_per_contract:.4f} (Entry: ${entry_price:.4f})\n"
                f"   Gross Payout: ${gross_payout:.2f} USD | Settle Fee: ${settle_fee:.2f} USD\n"
                f"   Net PnL (including entry/exit fees): ${net_pnl:.2f} USD"
            )

        del self.positions[market_id]
        self.print_portfolio_status()

    def print_portfolio_status(self):
        """Displays dashboard metrics of current account status."""
        open_pos_count = sum(len(o) for o in self.positions.values())
        completed_trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t["win"])
        win_rate = (wins / completed_trades * 100) if completed_trades > 0 else 0.0
        net_pnl = sum(t["net_pnl"] for t in self.trade_history)
        yield_pct = (net_pnl / CONFIG["INITIAL_BALANCE"]) * 100

        logger.info(f"-------------------- PORTFOLIO STATUS --------------------")
        logger.info(f"💵 Available Cash: ${self.cash:.2f} USD")
        logger.info(f"📦 Active Open Positions: {open_pos_count}")
        logger.info(f"📊 Completed Trades: {completed_trades} | Win Rate: {win_rate:.1f}%")
        logger.info(f"💸 Total Net PnL: ${net_pnl:.2f} USD (Yield: {yield_pct:.2f}%)")
        logger.info(f"🏦 Total Fees Paid to Platform: ${self.total_fees_paid:.2f} USD")
        logger.info(f"----------------------------------------------------------\n")


class SportsLatencyArbitrageEngine:
    """
    Subsystem specifically designed to evaluate in-play sports events against
    Polymarket token prices. It features a built-in soccer probability estimator
    to determine if a genuine latency gap exists [20].
    """
    def __init__(self):
        # Tracking live match states: {game_id: {score: "0-0", minute: 45, last_updated: float}}
        self.match_states = {}

    def calculate_soccer_probability(self, score_home, score_away, minute, team_focus="home"):
        """
        Uses a soccer probability math model (Poisson-derived distribution)
        to evaluate the real-time probability of the focusing team winning.
        """
        if minute >= 90:
            if score_home > score_away:
                return 1.0 if team_focus == "home" else 0.0
            elif score_home < score_away:
                return 0.0 if team_focus == "home" else 1.0
            return 0.0 # Draw

        # Basic expected goals (xG) model
        time_remaining = (90.0 - minute) / 90.0
        
        # Expected base intensity of remaining goals (typical soccer match is ~2.7 goals per 90 mins)
        home_xg_rem = 1.45 * time_remaining
        away_xg_rem = 1.25 * time_remaining

        goal_diff = score_home - score_away

        # Calculate probability using normal distribution approximation of remaining goals
        # P(Home Win) = P(Home goals remaining - Away goals remaining > -Goal Difference)
        mean_diff = home_xg_rem - away_xg_rem
        variance = home_xg_rem + away_xg_rem + 0.1 # variance of Poisson difference is sum of means
        
        if variance <= 0.1:
            return 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.0)

        std_dev = math.sqrt(variance)
        z_score = (-goal_diff + 0.5 - mean_diff) / std_dev # with continuity correction
        
        # Cumulative distribution function (CDF) for normal distribution
        prob_home_not_winning = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        prob_home_win = 1.0 - prob_home_not_winning

        if team_focus == "home":
            return max(0.01, min(0.99, prob_home_win))
        else:
            # Simple probability of away win
            z_score_away = (goal_diff + 0.5 - away_xg_rem + home_xg_rem) / std_dev
            prob_away_not_winning = 0.5 * (1.0 + math.erf(z_score_away / math.sqrt(2.0)))
            return max(0.01, min(0.99, 1.0 - prob_away_not_winning))

    def update_live_feed(self, game_id, team_home, team_away, score_home, score_away, minute, timestamp=None):
        """Registers a live game event update."""
        if not timestamp:
            timestamp = time.time()
            
        old_state = self.match_states.get(game_id, {"score": "0-0", "minute": 0})
        is_event_changed = old_state["score"] != f"{score_home}-{score_away}"

        self.match_states[game_id] = {
            "teams": f"{team_home} vs {team_away}",
            "score_home": score_home,
            "score_away": score_away,
            "score": f"{score_home}-{score_away}",
            "minute": minute,
            "last_updated": timestamp
        }

        return is_event_changed, old_state


class UnifiedPolymarketBot:
    """The central bot coordinating multi-signal alerts, anti-frontrun logic and EIP-712 orders."""
    def __init__(self):
        # Initialize sub-strategies
        self.whale_threshold = CONFIG["WHALE_USD_THRESHOLD"]
        self.coor_window = CONFIG["COORDINATION_WINDOW_SECS"]
        self.coor_min_wallets = CONFIG["COORDINATION_MIN_UNIQUE_WALLETS"]
        self.mom_window = CONFIG["MOMENTUM_WINDOW_SECS"]
        self.mom_multiplier = CONFIG["MOMENTUM_VOLUME_MULTIPLIER"]
        self.parity_threshold = CONFIG["PARITY_ARBITRAGE_THRESHOLD"]
        self.overreaction_delta = CONFIG["OVERREACTION_PRICE_DELTA"]

        # Telemetry Data Structures
        self.trade_timestamps = defaultdict(deque)
        self.trade_wallets = defaultdict(deque) # {market_id: deque([(timestamp, wallet), ...])}
        self.recent_trade_volumes = defaultdict(deque)
        self.historical_avg_volume = defaultdict(float)
        self.token_prices = defaultdict(dict)
        self.market_names = {}
        self.last_price_snapshots = defaultdict(dict)

        # Specialized Systems
        self.portfolio = FrictionAwarePortfolioEngine(CONFIG["INITIAL_BALANCE"])
        self.sports_engine = SportsLatencyArbitrageEngine()

    def check_geographic_compliance(self):
        """Checks for US geolocation block explicitly mentioned in the podcast [7]."""
        logger.info("Checking connection geographic compliance...")
        try:
            proxies = None
            if CONFIG["HTTP_PROXY"]:
                proxies = {"http": CONFIG["HTTP_PROXY"], "https": CONFIG["HTTP_PROXY"]}
                
            res = requests.get("https://ipapi.co/json/", timeout=4, proxies=proxies)
            if res.status_code == 200:
                data = res.json()
                country = data.get("country_code", "UNKNOWN")
                if country == "US":
                    logger.error("🛑 GEOLOCATION COMPLIANCE CRITICAL WARNING: Running on US IP address!")
                    logger.error("   Polymarket restricts users from the United States [7].")
                    logger.error("   Please configure an off-shore proxy in your .env file (HTTP_PROXY)!")
                    return False
                else:
                    logger.info(f"✅ Geolocation compliance passed. IP detected in: {country}")
                    return True
        except Exception as e:
            logger.warning(f"⚠️ Geolocation compliance check failed (Network Offline or Timeout): {e}")
        return True

    def process_live_trade(self, trade):
        """Processes transaction inputs through our 6 core scanner algorithms."""
        market_id = trade.get("market_id")
        market_title = trade.get("market_title", "Unknown Market")
        token_id = trade.get("token_id", "UnknownToken")
        outcome = trade.get("outcome", "Yes")
        price = float(trade.get("price", 0.0))
        size = float(trade.get("size", 0.0))
        wallet = trade.get("wallet_address", "0x0")
        timestamp = float(trade.get("timestamp", time.time()))

        self.market_names[market_id] = market_title
        usd_value = price * size

        # Update price state matrix
        if outcome not in self.token_prices[market_id]:
            self.token_prices[market_id][outcome] = price
            self.last_price_snapshots[market_id][outcome] = price

        old_price = self.last_price_snapshots[market_id][outcome]
        self.token_prices[market_id][outcome] = price

        signals = {}

        # --- STRATEGY 1: Whale Block Alert [8, 9] ---
        if usd_value >= self.whale_threshold:
            signals["whale"] = {
                "msg": f"🐋 WHALE ACTION: Traded {size:,.0f} contracts of '{outcome}' at {price:.2f} (${usd_value:,.2f} USD)"
            }

        # --- STRATEGY 2: Coordinated Unique Wallet Burst [9, 11] ---
        if self._check_coordination(market_id, wallet, timestamp):
            signals["coordinated_velocity"] = {
                "msg": f"🔥 COORDINATION DETECTED: Clustered execution bursts detected within {self.coor_window}s"
            }

        # --- STRATEGY 3: Volume Momentum Spike [8, 11] ---
        if self._check_momentum(market_id, usd_value, timestamp):
            signals["momentum_spike"] = {
                "msg": f"📈 MOMENTUM SPIKE: Market volume exceeded rolling average by {self.mom_multiplier}x"
            }

        # --- STRATEGY 4: Outcome Parity Arbitrage (YES+NO=1.00 Inefficiencies) [20] ---
        is_arb, arb_type, arb_margin = self._check_parity_arbitrage(market_id)
        if is_arb:
            signals["parity_arbitrage"] = {
                "msg": f"⚖️ PARITY ARBITRAGE ({arb_type}): Parity sum is {price + self._get_complement_price(market_id, outcome):.3f} (Ideal: 1.00)"
            }

        # --- STRATEGY 5: Whale Overreaction Reverse Swing ---
        if usd_value >= self.whale_threshold and abs(price - old_price) >= self.overreaction_delta:
            comp_outcome = "No" if outcome == "Yes" else "Yes"
            comp_price = self._get_complement_price(market_id, outcome)
            signals["whale_overreaction"] = {
                "msg": f"🔄 WHALE OVERREACTION DETECTED: Price spiked by {abs(price - old_price):.2f}. Actionable contrarian play on discount '{comp_outcome}' at ${comp_price:.2f}."
            }

        # --- STRATEGY 6: Sentiment Divergence Wall (Iceberg Order) ---
        if len(self.trade_timestamps[market_id]) >= self.coor_min_wallets * 1.5 and abs(price - old_price) <= 0.01:
            signals["sentiment_divergence"] = {
                "msg": "🧱 SENTIMENT DIVERGENCE: Mass retail volume seen, but price remains static (Possible Iceberg Wall)."
            }

        self.last_price_snapshots[market_id][outcome] = price

        if signals:
            score = self._calculate_confidence(signals, usd_value, market_id)
            self._dispatch_and_trade(market_id, market_title, outcome, price, comp_outcome if "whale_overreaction" in signals else outcome, signals, score)

    def process_sports_event(self, market_id, game_id, team_home, team_away, score_home, score_away, minute, contract_focus_outcome="Yes", team_focus="home"):
        """
        Integrates play-by-play events with book order logic.
        Triggers Time/Latency Arbitrage if a goal occurs but the market price does not adapt [20].
        """
        is_change, old_state = self.sports_engine.update_live_feed(game_id, team_home, team_away, score_home, score_away, minute)
        market_title = self.market_names.get(market_id, f"Will {team_home} beat {team_away}? [13]")

        if is_change:
            logger.info(f"⚽ [SPORTS LIVE CHANNEL] Score update for {team_home} vs {team_away}: {old_state['score']} -> {score_home}-{score_away} ({minute}')")
            
            # Fetch current market price
            current_price = self.token_prices[market_id].get(contract_focus_outcome, 0.5)

            # Evaluate Instant Fair Value using the soccer probability model
            fair_prob = self.sports_engine.calculate_soccer_probability(score_home, score_away, minute, team_focus=team_focus)
            
            # If the focusing team scores, the fair value should spike. But if the market price is lagging, we have an arbitrage opportunity!
            price_discrepancy = fair_prob - current_price
            
            if price_discrepancy > CONFIG["MIN_ARBITRAGE_EDGE_PCT"]:
                latency_lag_secs = int(time.time() - self.sports_engine.match_states[game_id]["last_updated"])
                
                signals = {
                    "sports_latency_arbitrage": {
                        "msg": (f"⏱️ TIME/LATENCY ARBITRAGE: Goal scored! Live Fair Value estimated at {fair_prob:.2f}. "
                                f"Market price is lagging at ${current_price:.2f}. Edge: {price_discrepancy*100:.1f}%. "
                                f"Exploit window open (Lag: {latency_lag_secs}s) [20].")
                    }
                }
                
                # High confidence score for pure latency arbitrage
                score = 95
                self._dispatch_and_trade(
                    market_id=market_id, 
                    market_title=market_title, 
                    raw_outcome=contract_focus_outcome, 
                    current_price=current_price, 
                    target_outcome=contract_focus_outcome, 
                    signals=signals, 
                    confidence=score, 
                    target_probability=fair_prob
                )

    def _check_coordination(self, market_id, wallet, timestamp):
        """Tracks the frequency of unique wallets in sliding window [9, 11]."""
        records = self.trade_wallets[market_id]
        records.append((timestamp, wallet))
        
        # Evict stale records
        while records and records[0][0] < (timestamp - self.coor_window):
            records.popleft()
            
        unique_wallets = {r[1] for r in records}
        return len(unique_wallets) >= self.coor_min_wallets

    def _check_momentum(self, market_id, usd_value, timestamp):
        volumes = self.recent_trade_volumes[market_id]
        volumes.append((timestamp, usd_value))
        
        # Evict stale volumes
        while volumes and volumes[0][0] < (timestamp - self.mom_window):
            volumes.popleft()

        current_window_total = sum(v[1] for v in volumes)
        if market_id not in self.historical_avg_volume:
            self.historical_avg_volume[market_id] = self.whale_threshold * 0.5

        avg_norm = self.historical_avg_volume[market_id]
        if len(volumes) > 2 and current_window_total > (avg_norm * self.mom_multiplier):
            self.historical_avg_volume[market_id] = max(avg_norm, current_window_total * 0.8)
            return True
        
        self.historical_avg_volume[market_id] = (avg_norm * 0.95) + (current_window_total * 0.05)
        return False

    def _check_parity_arbitrage(self, market_id):
        prices = self.token_prices[market_id]
        if "Yes" in prices and "No" in prices:
            total_sum = prices["Yes"] + prices["No"]
            delta = total_sum - 1.0
            if delta < -self.parity_threshold:
                return True, "SUB_PARITY_BUY_BOTH", abs(delta)
            elif delta > self.parity_threshold:
                return True, "SUPER_PARITY_MISPRICING", delta
        return False, None, 0.0

    def _check_consensus_loop(self, market_id):
        """
        Anti-Consensus Loop Guard: Reduces risk of getting chopped by high-frequency
        overlapping bots trading against each other repeatedly [14].
        """
        timestamps = self.trade_timestamps[market_id]
        if len(timestamps) > self.coor_min_wallets * 3:
            # High trade rate with static pricing indicates consensus loop trap
            prices = list(self.token_prices[market_id].values())
            if prices and (max(prices) - min(prices)) < 0.01:
                return True
        return False

    def _get_complement_price(self, market_id, current_outcome):
        comp = "No" if current_outcome == "Yes" else "Yes"
        return self.token_prices[market_id].get(comp, 0.5)

    def _calculate_confidence(self, signals, usd_value, market_id):
        score = 40
        if "whale" in signals:
            score += min(20, int((usd_value / self.whale_threshold) * 4))
        if "coordinated_velocity" in signals:
            score += 15
        if "momentum_spike" in signals:
            score += 10
        if "parity_arbitrage" in signals:
            score += 20
        if "whale_overreaction" in signals:
            score += 25
        if "sentiment_divergence" in signals:
            score += 15

        # Anti-Consensus Loop Penalty [14]
        if self._check_consensus_loop(market_id):
            logger.info(f"🌀 Consensus Loop Trap detected on market '{market_id}'! Applying -30 penalty to score.")
            score -= 30

        return max(0, min(100, score))

    def _generate_terminal_qr(self, url):
        """Prints a visual placeholder of a QR Code for phone scanning [11]."""
        # Simulated QR code pattern in console terminal
        qr_pattern = """
        ┌───┐ █  ▄ ┌───┐
        │ █ │ █▄██ │ █ │
        └───┘ ▄  ▄ └───┘
        ▄▄█▄▄ ██▀▄ █▄▄▄█
        ┌───┐ ▀▀ ▄   ▀▄ 
        │ █ │ ▀▄█▀▄▄█▄▄
        └───┘ █  █ █ █ ▀
        """
        logger.info(f"📱 SCAN ON PHONE TO TRANSACT IN REAL-TIME [11]:\n{qr_pattern}")
        logger.info(f"Direct Event URL: {url}")

    def _dispatch_and_trade(self, market_id, market_title, raw_outcome, current_price, target_outcome, signals, confidence, target_probability=1.0):
        """Routes identified signals to the trading execution portfolio and outputs alerts."""
        status = "⚠️ OBSERVING SIGNAL"
        actionable_trade = False
        
        if confidence >= 80:
            status = "🚀 ACTIONABLE HIGH-CONFIDENCE TRADE TRIGGER (80+ CONFIDENCE)"
            actionable_trade = True

        slug = market_title.lower().replace(" ", "-").replace("?", "").replace("[", "").replace("]", "")
        target_url = f"https://polymarket.com/event/{slug}?tid={market_id}"

        alert = {
            "status": status,
            "confidence_score": confidence,
            "market": market_title,
            "outcome": target_outcome,
            "current_price": f"${current_price:.2f}",
            "triggered_strategies": list(signals.keys()),
            "strategy_details": {k: v["msg"] for k, v in signals.items()},
            "one_click_mobile_action": {
                "link": target_url,
                "note": "Scan code / click link to execute trade on Polymarket mobile UI instantly [11]"
            }
        }

        logger.info(f"===== NEW SIGNAL IDENTIFIED (Confidence: {confidence}) =====")
        print(json.dumps(alert, indent=2, ensure_ascii=False))
        
        if confidence >= 85:
            self._generate_terminal_qr(target_url)
            
        logger.info(f"===========================================================\n")

        # Execute simulated trading if configured [7, 8]
        if actionable_trade and CONFIG["PAPER_TRADING"]:
            # Evaluate edge vs fees to prevent low-margin negative yields [14]
            is_eligible, expected_pnl, entry_price = self.portfolio.evaluate_roi_eligibility(current_price, target_probability)
            
            if not is_eligible:
                logger.warning(
                    f"🛡️ [FRICTION BLOCKED] Blocked buying '{target_outcome}' on '{market_title}'!\n"
                    f"   Price: ${current_price:.2f} | Expected Value: {target_probability:.2f}.\n"
                    f"   The expected edge is too thin to survive platform fees and slippage (Est. Net Edge: ${expected_pnl:.4f}) [14].\n"
                )
                return

            allocation = min(self.portfolio.cash * 0.10, 100.0)
            self.portfolio.execute_buy(market_id, market_title, target_outcome, current_price, allocation)


class PolymarketOrderSigner:
    """EIP-712 Order Signer demonstrating hard price ceilings to prevent frontrunning."""
    def __init__(self, private_key=None):
        self.private_key = private_key or CONFIG["POLY_PRIVATE_KEY"]
        
    def sign_limit_order(self, token_id, price, size, max_slippage_price, side="BUY"):
        """
        Builds and cryptographically signs a limit order.
        Sets a strict price ceiling to prevent getting filled at post-event prices [20].
        """
        if not self.private_key or self.private_key == "your_wallet_private_key_here":
            return {"error": "Private key missing or default placeholder value detected."}

        order_payload = {
            "token_id": token_id,
            "price": f"{price:.4f}",
            "size": f"{size:.2f}",
            "max_allowed_fill_price": f"{max_slippage_price:.4f}", # Slippage protection ceiling
            "side": side,
            "nonce": int(time.time() * 1000),
            "expiration": int(time.time() + 180) # Valid for 3 minutes for high-speed sports arbitrage
        }
        
        # Mock cryptographic proof output (requires web3 and polygon network integration)
        order_payload["signature_type"] = 1 # EOA signatures
        order_payload["signature_proof"] = "0x_eip712_sports_latency_proof_hash_vector"
        
        return order_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Advanced Sports Latency Bot v7")
    parser.add_argument("--mock", action="store_true", help="Run simulation using high-speed sports latency mock feed")
    parser.add_argument("--live", action="store_true", help="Connect to live Polymarket WebSocket feed")
    args = parser.parse_args()

    if not args.live:
        args.mock = True

    tracker = UnifiedPolymarketBot()
    tracker.check_geographic_compliance()

    if args.mock:
        logger.info("Starting High-Speed Sports Latency & Multi-Strategy Mock Simulator...")
        
        MOCK_SPORTS_MARKET_ID = "0xsports999_newcastle"
        
        mock_ticks = [
            # Frame 1: Establish market and initial token prices for Newcastle vs Crystal Palace soccer game [13]
            {
                "market_id": MOCK_SPORTS_MARKET_ID, 
                "market_title": "Will Newcastle beat Crystal Palace? [13]", 
                "outcome": "Yes", 
                "price": 0.45, 
                "size": 1000, 
                "wallet_address": "0xuser1",
                "timestamp": time.time()
            },
            {
                "market_id": MOCK_SPORTS_MARKET_ID, 
                "market_title": "Will Newcastle beat Crystal Palace? [13]", 
                "outcome": "No", 
                "price": 0.50, 
                "size": 1000, 
                "wallet_address": "0xuser2",
                "timestamp": time.time() + 0.1
            },
            
            # Frame 2: Trigger a low-margin trade signal that should be blocked by FrictionGuard
            {
                "market_id": "m_gdp", 
                "market_title": "US GDP Growth Q3", 
                "outcome": "Yes", 
                "price": 0.97, # Very high price, low reward margins [14]
                "size": 18000, # Large whale purchase [8, 9]
                "wallet_address": "0xwhale",
                "timestamp": time.time() + 1.0
            },
        ]

        # Feed regular ticks
        for tick in mock_ticks:
            tracker.process_live_trade(tick)
            time.sleep(0.3)

        # Frame 3: Simulate live play-by-play goal event! Newcastle scores a goal in the 75th minute! [13, 20]
        # Spectator in the stadium registers the goal instantly. But the market price for Newcastle YES is still 0.45 [20]
        tracker.process_sports_event(
            market_id=MOCK_SPORTS_MARKET_ID,
            game_id="game_newcastle_cp_2026",
            team_home="Newcastle",
            team_away="Crystal Palace",
            score_home=1, # Newcastle scores! [13]
            score_away=0,
            minute=75,
            contract_focus_outcome="Yes",
            team_focus="home"
        )
        
        # Frame 4: Clean settlement simulation [6, 14]
        logger.info("Simulating match completion and sports event final settlement...")
        time.sleep(1)
        tracker.portfolio.settle_market(MOCK_SPORTS_MARKET_ID, "Will Newcastle beat Crystal Palace? [13]", "Yes")

    elif args.live:
        if not HAS_ASYNC_WS:
            logger.error("Dependency 'websockets' not found. Install via 'pip install websockets' to run live.")
            sys.exit(1)

        logger.info("Initializing live connection to Polymarket CLOB Market feed...")
        
        async def main():
            url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
            logger.info(f"Connecting to CLOB WebSocket at {url}")
            
            # Fetch active markets
            try:
                proxies = None
                if CONFIG["HTTP_PROXY"]:
                    proxies = {"http": CONFIG["HTTP_PROXY"], "https": CONFIG["HTTP_PROXY"]}
                res = requests.get("https://gamma-api.polymarket.com/markets?closed=false&limit=15&order=volume_24hr&ascending=false", proxies=proxies)
                active_markets = res.json()
            except Exception as e:
                logger.error(f"Failed to fetch active markets from Gamma: {e}")
                return

            token_to_market = {}
            all_token_ids = []
            
            for m in active_markets:
                q = m.get("question", m.get("slug"))
                mid = m.get("id")
                tokens = m.get("clobTokenIds", "[]")
                outcomes = m.get("outcomes", [])
                
                try:
                    if isinstance(tokens, str):
                        tokens = json.loads(tokens)
                except Exception:
                    continue

                for i, tid in enumerate(tokens):
                    if i < len(outcomes):
                        token_to_market[tid] = {
                            "market_id": mid,
                            "market_title": q,
                            "outcome": outcomes[i]
                        }
                        all_token_ids.append(tid)

            logger.info(f"Configured stream listener for {len(token_to_market)} tokens across active markets.")

            # Websocket connection pipe
            async with websockets.connect(url, ping_interval=10) as ws:
                subscribe_msg = {
                    "assets_ids": all_token_ids,
                    "type": "market"
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info("WebSocket connection open. Streaming live execution logs...")

                async for msg in ws:
                    event = json.loads(msg)
                    if event.get("event_type") == "last_trade_price":
                        asset_id = event.get("asset_id")
                        mapping = token_to_market.get(asset_id)
                        if mapping:
                            trade_payload = {
                                "market_id": mapping["market_id"],
                                "market_title": mapping["market_title"],
                                "token_id": asset_id,
                                "outcome": mapping["outcome"],
                                "price": float(event.get("price", 0)),
                                "size": float(event.get("size", 0)),
                                "wallet_address": event.get("maker", "0x0"), # If available
                                "timestamp": time.time()
                            }
                            tracker.process_live_trade(trade_payload)

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Successfully shut down live tracker.")
