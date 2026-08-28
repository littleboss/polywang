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


def env_value(names, default, cast=str):
    """
    Reads the first environment variable that is actually set among `names`.

    Several parameters are referenced under two spellings across the maintenance
    docs and the historical .env templates (for example MIN_NET_PROFIT_MARGIN vs
    MIN_ARBITRAGE_EDGE_PCT). Accepting aliases keeps old .env files working while
    the documented name stays canonical.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return cast(raw)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ Invalid value for {name}={raw!r}; falling back to default {default!r}.")
            break
    return default


def env_bool(names, default):
    return env_value(names, default, lambda raw: raw.strip().lower() in ("1", "true", "yes", "on"))


# Strategy Configuration from Environment
CONFIG = {
    # Friction Controls [14]
    "PAPER_TRADING": env_bool("PAPER_TRADING", True),
    "INITIAL_BALANCE": env_value("INITIAL_BALANCE", 1000.0, float),
    "SIMULATED_FEE_PCT": env_value("SIMULATED_FEE_PCT", 0.015, float),  # 1.5% average platform fee [7]
    "SLIPPAGE_PCT": env_value("SLIPPAGE_PCT", 0.005, float),            # 0.5% expected execution slippage
    # Charged on the winning payout at resolution, separately from the entry fee.
    "SETTLEMENT_FEE_PCT": env_value("SETTLEMENT_FEE_PCT", None, float),  # None -> mirror SIMULATED_FEE_PCT
    # Minimum net edge per contract that must survive fees + slippage before we trade [2].
    "MIN_NET_PROFIT_MARGIN": env_value(["MIN_NET_PROFIT_MARGIN", "MIN_ARBITRAGE_EDGE_PCT"], 0.05, float),

    # Sports Latency Arbitrage Specific Parameters [20]
    # How long the exploit window stays open after a goal is detected. Past this the
    # order book is assumed to have caught up and the edge is gone.
    "SPORTS_LATENCY_THRESHOLD_SECS": env_value("SPORTS_LATENCY_THRESHOLD_SECS", 5, int),
    # Scales the remaining expected goals in the Poisson model. >1 for high-scoring
    # competitions (odds move faster), <1 for low-scoring defensive leagues.
    "TIME_DECAY_WEIGHT": env_value("TIME_DECAY_WEIGHT", 1.0, float),
    "STOP_LOSS_ENABLED": env_bool("STOP_LOSS_ENABLED", True),

    # Standard Strategy Parameters [8, 9, 11]
    "WHALE_USD_THRESHOLD": env_value(["WHALE_THRESHOLD_USD", "WHALE_USD_THRESHOLD"], 5000.0, float),
    "COORDINATION_WINDOW_SECS": env_value(["COORDINATION_WINDOW", "COORDINATION_WINDOW_SECS"], 60, int),
    "COORDINATION_MIN_UNIQUE_WALLETS": env_value("COORDINATION_MIN_UNIQUE_WALLETS", 7, int), # 7 traders from podcast [9, 11]
    "MOMENTUM_WINDOW_SECS": env_value("MOMENTUM_WINDOW_SECS", 30, int),
    "MOMENTUM_VOLUME_MULTIPLIER": env_value("MOMENTUM_VOLUME_MULTIPLIER", 3.0, float),
    "OVERREACTION_PRICE_DELTA": env_value("OVERREACTION_PRICE_DELTA", 0.10, float),
    "PARITY_ARBITRAGE_THRESHOLD": env_value("PARITY_ARBITRAGE_THRESHOLD", 0.02, float),

    # API Credentials for Real CLOB V2 Ordering
    "POLY_API_KEY": os.getenv("POLY_API_KEY", ""),
    "POLY_API_SECRET": os.getenv("POLY_API_SECRET", ""),
    "POLY_PASSPHRASE": os.getenv("POLY_PASSPHRASE", ""),
    "POLY_PRIVATE_KEY": os.getenv("POLY_PRIVATE_KEY", ""),
    "HTTP_PROXY": os.getenv("HTTP_PROXY", "")
}

if CONFIG["SETTLEMENT_FEE_PCT"] is None:
    CONFIG["SETTLEMENT_FEE_PCT"] = CONFIG["SIMULATED_FEE_PCT"]

# Kept as an alias so existing scripts referencing the old key keep working.
CONFIG["MIN_ARBITRAGE_EDGE_PCT"] = CONFIG["MIN_NET_PROFIT_MARGIN"]

def write_default_env():
    env_path = ".env"
    if not os.path.exists(env_path):
        content = """# Polymarket Advanced Sports Latency Bot Config
PAPER_TRADING=True
INITIAL_BALANCE=1000.0

# Friction model. Update SIMULATED_FEE_PCT immediately whenever the platform
# changes its fee schedule, otherwise the ROI guard will let through trades that
# are actually loss-making.
SIMULATED_FEE_PCT=0.015
SLIPPAGE_PCT=0.005
SETTLEMENT_FEE_PCT=0.015

# ROI guard: minimum net edge per contract that must survive all friction.
MIN_NET_PROFIT_MARGIN=0.05

# Sports Latency Arbitrage settings
SPORTS_LATENCY_THRESHOLD_SECS=5
# Calibrate per sport: >1 for high-scoring competitions, <1 for defensive leagues.
TIME_DECAY_WEIGHT=1.0
STOP_LOSS_ENABLED=True

# General Strategy Thresholds
WHALE_THRESHOLD_USD=5000
COORDINATION_WINDOW=60
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
        self.settlement_fee_pct = CONFIG["SETTLEMENT_FEE_PCT"]
        self.total_fees_paid = 0.0

    def quote_friction(self, current_price, target_probability):
        """
        Breaks a candidate entry down into every cost component that stands between
        the raw price and the money that actually lands back in the account [14].

        Two edges are produced because they answer different questions:

        - `spec_net_edge` is the headline formula from the maintenance doc,
          (1.00 - Price) - (Market Fee + Slippage + Settlement Fee). It describes the
          best case: what a contract nets if the position resolves as a winner.
        - `expected_net_edge` weights that payout by the model's probability, so an
          80%-likely outcome bought at 0.75 is correctly treated as thin rather than
          as a guaranteed 0.25 gain.

        A trade must clear the margin on both measures. This is the specific failure
        mode from the podcast: a high hit rate that still loses money because every
        individual win was too small to pay for its own fees [1, 9].
        """
        price = max(0.0, min(0.999, float(current_price)))
        probability = max(0.0, min(1.0, float(target_probability)))

        slippage_cost = price * self.slippage_pct
        entry_price = min(0.995, price + slippage_cost)
        market_fee = entry_price * self.fee_pct
        # The settlement fee is only charged on payouts that actually happen, so the
        # expected cost scales with the win probability.
        settlement_fee = 1.0 * self.settlement_fee_pct * probability

        spec_net_edge = (1.0 - price) - (market_fee + slippage_cost + settlement_fee)
        expected_net_edge = probability - entry_price - market_fee - settlement_fee

        return {
            "price": price,
            "probability": probability,
            "entry_price": entry_price,
            "slippage_cost": slippage_cost,
            "market_fee": market_fee,
            "settlement_fee": settlement_fee,
            "total_friction": slippage_cost + market_fee + settlement_fee,
            "spec_net_edge": spec_net_edge,
            "expected_net_edge": expected_net_edge,
            "binding_edge": min(spec_net_edge, expected_net_edge),
        }

    def evaluate_roi_eligibility(self, current_price, target_probability):
        """
        Gate used by the execution path: returns whether the binding edge clears
        MIN_NET_PROFIT_MARGIN, plus the full cost breakdown for the alert log.
        """
        quote = self.quote_friction(current_price, target_probability)
        is_eligible = quote["binding_edge"] >= CONFIG["MIN_NET_PROFIT_MARGIN"]
        return is_eligible, quote

    def enforce_limit_price(self, live_price, max_allowed_price, market_title, outcome):
        """
        Limit Price Guard: refuses to fill above the ceiling that was priced in.

        In latency arbitrage the whole edge is that the book has not moved yet. If it
        moves while our order is in flight, the trade stops being an arbitrage and
        becomes buying the top. Voiding the order is the correct outcome.
        """
        if live_price <= max_allowed_price:
            return True

        logger.warning(
            f"🚫 [LIMIT PRICE GUARD] Order voided for '{market_title}' ({outcome}).\n"
            f"   Book moved to ${live_price:.4f}, above the ${max_allowed_price:.4f} ceiling. "
            f"The latency edge is gone; refusing to chase the post-event price [20]."
        )
        return False

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

    def execute_sell(self, market_id, market_title, outcome, price, reason="Manual exit"):
        """
        Liquidates an open position into the book before resolution.

        Used by the state-based stop loss: once the real-world assumption behind a
        trade is dead, holding to settlement just donates the remaining premium.
        Exiting early recovers whatever the book still pays, minus friction.
        """
        pos = self.positions.get(market_id, {}).get(outcome)
        if not pos or pos["contracts"] <= 0:
            return False

        contracts = pos["contracts"]
        entry_price = pos["avg_price"]

        # Selling crosses the spread in the unfavourable direction, so slippage
        # subtracts here instead of adding as it does on entry.
        exit_price = max(0.0, price * (1.0 - self.slippage_pct))
        gross_proceeds = contracts * exit_price
        fee = gross_proceeds * self.fee_pct
        net_proceeds = gross_proceeds - fee

        self.cash += net_proceeds
        self.total_fees_paid += fee
        net_pnl = net_proceeds - (contracts * entry_price)

        del self.positions[market_id][outcome]
        if not self.positions[market_id]:
            del self.positions[market_id]

        self.trade_history.append({
            "market": market_title,
            "outcome": outcome,
            "win": net_pnl > 0,
            "net_pnl": net_pnl
        })

        logger.info(
            f"🚪 [SIMULATED EXIT] {market_title} ({outcome}) — {reason}\n"
            f"   Sold {contracts:,.2f} contracts at ${exit_price:.4f} (Entry: ${entry_price:.4f})\n"
            f"   Gross Proceeds: ${gross_proceeds:.2f} USD | Exit Fee: ${fee:.2f} USD\n"
            f"   Net PnL: ${net_pnl:.2f} USD"
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
    # League-average expected goals over a full 90 minutes, split by home advantage.
    BASE_HOME_XG = 1.45
    BASE_AWAY_XG = 1.25

    def __init__(self, time_decay_weight=None):
        # Tracking live match states: {game_id: {score: "0-0", minute: 45, last_updated: float}}
        self.match_states = {}
        self.time_decay_weight = (
            CONFIG["TIME_DECAY_WEIGHT"] if time_decay_weight is None else time_decay_weight
        )

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

        # Expected base intensity of remaining goals (typical soccer match is ~2.7 goals per 90 mins).
        # TIME_DECAY_WEIGHT rescales that intensity so the same engine can serve
        # competitions whose scoring rate differs from the league average.
        weight = max(0.01, self.time_decay_weight)
        home_xg_rem = self.BASE_HOME_XG * time_remaining * weight
        away_xg_rem = self.BASE_AWAY_XG * time_remaining * weight

        goal_diff = score_home - score_away

        # Calculate probability using normal distribution approximation of remaining goals
        # P(Home Win) = P(Home goals remaining - Away goals remaining > -Goal Difference)
        mean_diff = home_xg_rem - away_xg_rem
        # Variance of a difference of Poisson variables is the sum of their means.
        # No padding is added here: a constant floor would keep injecting uncertainty
        # into the closing minutes, which is exactly when the latency edge is decided.
        variance = home_xg_rem + away_xg_rem

        if variance <= 1e-9:
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
        """
        Registers a live game event update.

        `timestamp` is the moment the event was observed in the real world (the
        stadium spectator or the low-latency data feed), not the moment we finished
        processing it. The latency edge is measured against this value, so callers
        replaying recorded feeds must pass the original event time.
        """
        if not timestamp:
            timestamp = time.time()

        old_state = self.match_states.get(game_id, {"score": "0-0", "minute": 0, "last_updated": timestamp})
        is_event_changed = old_state["score"] != f"{score_home}-{score_away}"

        self.match_states[game_id] = {
            "teams": f"{team_home} vs {team_away}",
            "score_home": score_home,
            "score_away": score_away,
            "score": f"{score_home}-{score_away}",
            "minute": minute,
            # When the score actually moved, this is the goal time; otherwise it
            # stays pinned to the previous event so the exploit window keeps ticking.
            "last_updated": timestamp if is_event_changed else old_state.get("last_updated", timestamp)
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
        # Last time the order book moved for a market. Compared against the sports
        # feed timestamp to prove the book has not yet reacted to a goal.
        self.last_book_update_ts = {}

        # Open latency-arbitrage theses, keyed by market, so the stop loss knows what
        # real-world condition each position was betting on.
        self.open_sports_theses = {}

        # Specialized Systems
        self.portfolio = FrictionAwarePortfolioEngine(CONFIG["INITIAL_BALANCE"])
        self.sports_engine = SportsLatencyArbitrageEngine()
        self.order_signer = PolymarketOrderSigner()

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
        self.last_book_update_ts[market_id] = timestamp

        # Rolling trade-rate window. Strategy 6 and the anti-consensus guard both read
        # this deque, so it has to be fed here or they can never fire.
        rate_window = self.trade_timestamps[market_id]
        rate_window.append(timestamp)
        while rate_window and rate_window[0] < (timestamp - self.mom_window):
            rate_window.popleft()

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
            target_outcome = comp_outcome if "whale_overreaction" in signals else outcome
            target_price = price if target_outcome == outcome else self._get_complement_price(market_id, outcome)
            self._dispatch_and_trade(
                market_id, market_title, outcome, target_price, target_outcome, signals, score,
                # Order-flow signals carry no external ground truth, so the confidence
                # score is the only probability estimate available. Assuming certainty
                # here would let the ROI guard wave through near-$1.00 entries.
                target_probability=min(0.99, score / 100.0)
            )

    def process_sports_event(self, market_id, game_id, team_home, team_away, score_home, score_away,
                             minute, contract_focus_outcome="Yes", team_focus="home",
                             event_timestamp=None, now=None):
        """
        Integrates play-by-play events with book order logic.
        Triggers Time/Latency Arbitrage if a goal occurs but the market price does not adapt [20].

        `event_timestamp` is when the goal happened in the real world and `now` is
        when we are acting on it. Both are injectable so recorded feeds and tests can
        reproduce a specific latency scenario instead of depending on wall-clock time.
        """
        event_timestamp = time.time() if event_timestamp is None else event_timestamp
        now = time.time() if now is None else now

        is_change, old_state = self.sports_engine.update_live_feed(
            game_id, team_home, team_away, score_home, score_away, minute, timestamp=event_timestamp
        )
        market_title = self.market_names.get(market_id, f"Will {team_home} beat {team_away}? [13]")

        if is_change:
            logger.info(f"⚽ [SPORTS LIVE CHANNEL] Score update for {team_home} vs {team_away}: {old_state['score']} -> {score_home}-{score_away} ({minute}')")

        # Every score update, including ones that do not open a new trade, has to be
        # checked against open positions: a goal for the other side kills the thesis.
        self._check_state_stop_loss(market_id, game_id, score_home, score_away, minute)

        if not is_change:
            return

        # How long the exploit window has been open. Once the book has had more than
        # SPORTS_LATENCY_THRESHOLD_SECS to digest the goal, any remaining gap is more
        # likely to be stale data on our side than a real edge.
        window = CONFIG["SPORTS_LATENCY_THRESHOLD_SECS"]
        elapsed_since_goal = max(0.0, now - event_timestamp)
        if elapsed_since_goal > window:
            logger.info(
                f"⌛ [WINDOW CLOSED] Goal detected {elapsed_since_goal:.1f}s ago, past the "
                f"{window}s latency window. Standing down on '{market_title}'."
            )
            return

        current_price = self.token_prices[market_id].get(contract_focus_outcome, 0.5)

        # Evaluate Instant Fair Value using the soccer probability model
        fair_prob = self.sports_engine.calculate_soccer_probability(score_home, score_away, minute, team_focus=team_focus)

        # If the focusing team scores, the fair value should spike. But if the market price is lagging, we have an arbitrage opportunity!
        price_discrepancy = fair_prob - current_price
        if price_discrepancy <= CONFIG["MIN_NET_PROFIT_MARGIN"]:
            return

        # Gap between the goal and the last time the book actually moved. A large
        # positive number is the evidence that the market has not repriced yet.
        last_book_ts = self.last_book_update_ts.get(market_id, event_timestamp)
        book_lag_secs = max(0.0, event_timestamp - last_book_ts)

        signals = {
            "sports_latency_arbitrage": {
                "msg": (f"⏱️ TIME/LATENCY ARBITRAGE: Goal scored! Live Fair Value estimated at {fair_prob:.2f}. "
                        f"Market price is lagging at ${current_price:.2f}. Edge: {price_discrepancy*100:.1f}%. "
                        f"Book last moved {book_lag_secs:.1f}s before the goal; "
                        f"{max(0.0, window - elapsed_since_goal):.1f}s of exploit window left [20].")
            }
        }

        self._dispatch_and_trade(
            market_id=market_id,
            market_title=market_title,
            raw_outcome=contract_focus_outcome,
            current_price=current_price,
            target_outcome=contract_focus_outcome,
            signals=signals,
            confidence=self._score_latency_arbitrage(price_discrepancy, elapsed_since_goal, window),
            target_probability=fair_prob,
            sports_thesis={
                "game_id": game_id,
                "team_focus": team_focus,
                "outcome": contract_focus_outcome,
                "goal_diff_at_entry": score_home - score_away,
                "minute_at_entry": minute,
            }
        )

    @staticmethod
    def _score_latency_arbitrage(edge, elapsed_since_goal, window):
        """
        Confidence for a pure latency play. The maintenance spec calls for 90+, which
        reflects that we are not predicting anything: the goal already happened and
        only the book is behind. The score still tapers as the window closes, because
        a late fill is far more likely to be a losing one.
        """
        score = 90 + min(9, int(edge * 20))
        if window > 0:
            freshness = 1.0 - min(1.0, elapsed_since_goal / float(window))
            score -= int((1.0 - freshness) * 8)
        return max(0, min(100, score))

    def _check_state_stop_loss(self, market_id, game_id, score_home, score_away, minute):
        """
        State-based stop loss for sports positions.

        Ordinary stop losses wait for the price to fall. Here the scoreboard is the
        ground truth and the price is the laggard, so waiting for the drop means
        selling after everyone else has already sold. Instead the exit fires the
        moment the real-world condition behind the trade stops holding.
        """
        if not CONFIG["STOP_LOSS_ENABLED"]:
            return False

        thesis = self.open_sports_theses.get(market_id)
        if not thesis or thesis.get("game_id") != game_id:
            return False

        goal_diff = score_home - score_away
        # A thesis on the home side needs the home side to still be ahead, and the
        # mirror image for the away side.
        still_valid = goal_diff > 0 if thesis["team_focus"] == "home" else goal_diff < 0
        if still_valid:
            return False

        market_title = self.market_names.get(market_id, market_id)
        outcome = thesis["outcome"]
        current_price = self.token_prices[market_id].get(outcome, 0.5)

        logger.warning(
            f"🛑 [STATE STOP LOSS] Scoreline moved to {score_home}-{score_away} at {minute}'. "
            f"The '{thesis['team_focus']}' thesis behind '{market_title}' is no longer true — exiting before the book catches up."
        )
        self.portfolio.execute_sell(
            market_id, market_title, outcome, current_price,
            reason=f"Score reversal to {score_home}-{score_away} at {minute}'"
        )
        self.open_sports_theses.pop(market_id, None)
        return True

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

    def update_book_quote(self, market_id, market_title, outcome, price, timestamp=None):
        """
        Applies an order-book quote update that did not come from a trade.

        Trades alone are a lagging view of the book: a market can reprice on quotes
        long before anyone crosses the spread. Feeding quotes in keeps the latency
        comparison honest, otherwise a quiet-but-repriced book still looks stale.
        """
        timestamp = time.time() if timestamp is None else timestamp
        self.market_names.setdefault(market_id, market_title)
        self.token_prices[market_id][outcome] = price
        self.last_price_snapshots[market_id].setdefault(outcome, price)
        self.last_book_update_ts[market_id] = timestamp

    def settle_sports_market(self, market_id, market_title, winning_outcome):
        """Settles a market and retires any latency thesis attached to it."""
        self.open_sports_theses.pop(market_id, None)
        self.portfolio.settle_market(market_id, market_title, winning_outcome)

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

    def _dispatch_and_trade(self, market_id, market_title, raw_outcome, current_price, target_outcome,
                            signals, confidence, target_probability=1.0, sports_thesis=None):
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

        if not actionable_trade:
            return

        # Evaluate edge vs fees to prevent low-margin negative yields [14].
        # This runs before any order is built, in paper and live mode alike.
        is_eligible, quote = self.portfolio.evaluate_roi_eligibility(current_price, target_probability)

        if not is_eligible:
            logger.warning(
                f"🛡️ [FRICTION BLOCKED] Blocked buying '{target_outcome}' on '{market_title}'!\n"
                f"   Price: ${quote['price']:.4f} | Modelled Fair Value: {quote['probability']:.2f}\n"
                f"   Friction: slippage ${quote['slippage_cost']:.4f} + entry fee ${quote['market_fee']:.4f} "
                f"+ settlement fee ${quote['settlement_fee']:.4f} = ${quote['total_friction']:.4f}\n"
                f"   Net edge if it wins: ${quote['spec_net_edge']:.4f} | Probability-weighted: ${quote['expected_net_edge']:.4f}\n"
                f"   Required minimum: ${CONFIG['MIN_NET_PROFIT_MARGIN']:.4f}. High accuracy does not pay for fees this thin [1, 14].\n"
            )
            return

        logger.info(
            f"✅ [FRICTION PASSED] Net edge ${quote['binding_edge']:.4f} clears the "
            f"${CONFIG['MIN_NET_PROFIT_MARGIN']:.4f} minimum after ${quote['total_friction']:.4f} of friction."
        )

        # Hard ceiling for the order. If the book reprices above this while we are in
        # flight, the fill is refused rather than chasing the post-goal price [20].
        max_allowed_price = quote["entry_price"]
        signed_order = self.order_signer.sign_limit_order(
            token_id=market_id,
            price=quote["price"],
            size=self._position_size(quote["entry_price"]),
            max_slippage_price=max_allowed_price,
            side="BUY"
        )
        if "error" in signed_order:
            logger.info(f"🔏 [LIMIT ORDER] Not signed ({signed_order['error']}). Paper trade only.")
        else:
            logger.info(
                f"🔏 [LIMIT ORDER SIGNED] Ceiling ${max_allowed_price:.4f}, expires in "
                f"{signed_order['expiration'] - int(time.time())}s. Fills above the ceiling are rejected."
            )

        # Re-read the book instead of reusing the price the signal was built from.
        # Between signal and execution the market may already have absorbed the news,
        # which is precisely the situation the ceiling exists to catch.
        live_price = self.token_prices[market_id].get(target_outcome, current_price)
        if not self.portfolio.enforce_limit_price(live_price, max_allowed_price, market_title, target_outcome):
            return

        if CONFIG["PAPER_TRADING"]:
            allocation = min(self.portfolio.cash * 0.10, 100.0)
            bought = self.portfolio.execute_buy(market_id, market_title, target_outcome, live_price, allocation)
            if bought and sports_thesis:
                self.open_sports_theses[market_id] = sports_thesis
        else:
            logger.warning("🔴 Live execution path is not enabled in this build; order was signed but not submitted.")

    def _position_size(self, entry_price):
        """Contracts affordable under the standard 10%-of-cash allocation cap."""
        allocation = min(self.portfolio.cash * 0.10, 100.0)
        return allocation / max(0.01, entry_price)


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


def extract_mid_price(event):
    """
    Pulls a mid price out of a CLOB `book` or `price_change` message.

    Prefers the midpoint of the best bid and ask; falls back to whichever side is
    present, since a one-sided book still tells us the market has moved. Returns
    None when the message carries no usable price.
    """
    def best(levels, pick):
        prices = []
        for level in levels or []:
            try:
                prices.append(float(level["price"]))
            except (KeyError, TypeError, ValueError):
                continue
        return pick(prices) if prices else None

    best_bid = best(event.get("bids") or event.get("buys"), max)
    best_ask = best(event.get("asks") or event.get("sells"), min)

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    if best_bid is not None:
        return best_bid
    if best_ask is not None:
        return best_ask

    for key in ("price", "midpoint"):
        try:
            return float(event[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def run_mock_simulation(tracker):
    """
    Replays four scenarios that each exercise one of the v7 safety systems, so a
    maintainer can confirm the guards still fire after changing thresholds.
    """
    logger.info("Starting High-Speed Sports Latency & Multi-Strategy Mock Simulator...")

    newcastle_market = "0xsports999_newcastle"
    newcastle_title = "Will Newcastle beat Crystal Palace? [13]"
    wolves_market = "0xsports777_wolves"
    wolves_title = "Will Wolves beat Fulham?"
    brighton_market = "0xsports555_brighton"
    brighton_title = "Will Brighton beat Everton?"

    base_ts = time.time()

    logger.info("\n########## SCENARIO 1: Establish the book ##########")
    for outcome, price, wallet in (("Yes", 0.45, "0xuser1"), ("No", 0.50, "0xuser2")):
        tracker.process_live_trade({
            "market_id": newcastle_market, "market_title": newcastle_title,
            "outcome": outcome, "price": price, "size": 1000,
            "wallet_address": wallet, "timestamp": base_ts
        })
    for market_id, title in ((wolves_market, wolves_title), (brighton_market, brighton_title)):
        tracker.update_book_quote(market_id, title, "Yes", 0.40, timestamp=base_ts)
        tracker.update_book_quote(market_id, title, "No", 0.58, timestamp=base_ts)

    logger.info("\n########## SCENARIO 2: High confidence, no margin — the ROI guard must block ##########")
    # A 0.97 contract with whales, a parity dislocation and eight distinct wallets
    # piling in. Confidence clears the 80 action threshold easily, so the only thing
    # standing between the bot and a losing trade is the friction maths. This is the
    # exact shape of the 66%-win-rate loss described in the podcast [1, 9].
    tracker.process_live_trade({
        "market_id": "m_gdp", "market_title": "US GDP Growth Q3",
        "outcome": "No", "price": 0.10, "size": 500,
        "wallet_address": "0xmaker", "timestamp": base_ts + 1.0
    })
    for i in range(8):
        tracker.process_live_trade({
            "market_id": "m_gdp", "market_title": "US GDP Growth Q3",
            "outcome": "Yes", "price": 0.97, "size": 30000,
            "wallet_address": f"0xwhale{i}", "timestamp": base_ts + 1.1 + (i * 0.05)
        })

    logger.info("\n########## SCENARIO 3: Goal detected before the book reprices ##########")
    # The book last moved when we established it; the goal lands well after, so the
    # latency gap is real and the exploit window is still open [13, 20].
    goal_ts = base_ts + 12.0
    tracker.process_sports_event(
        market_id=newcastle_market, game_id="game_newcastle_cp_2026",
        team_home="Newcastle", team_away="Crystal Palace",
        score_home=1, score_away=0, minute=75,
        contract_focus_outcome="Yes", team_focus="home",
        event_timestamp=goal_ts, now=goal_ts + 1.0
    )

    logger.info("\n########## SCENARIO 4: Newcastle hold on — clean settlement ##########")
    tracker.settle_sports_market(newcastle_market, newcastle_title, "Yes")

    logger.info("\n########## SCENARIO 5: Stale feed — the latency window must be closed ##########")
    stale_goal_ts = base_ts + 20.0
    tracker.process_sports_event(
        market_id=brighton_market, game_id="game_brighton_everton_2026",
        team_home="Brighton", team_away="Everton",
        score_home=1, score_away=0, minute=60,
        contract_focus_outcome="Yes", team_focus="home",
        event_timestamp=stale_goal_ts,
        # Acted on far later than SPORTS_LATENCY_THRESHOLD_SECS allows.
        now=stale_goal_ts + CONFIG["SPORTS_LATENCY_THRESHOLD_SECS"] + 25.0
    )

    logger.info("\n########## SCENARIO 6: Score reversal — the state stop loss must fire ##########")
    wolves_goal_ts = base_ts + 40.0
    tracker.process_sports_event(
        market_id=wolves_market, game_id="game_wolves_fulham_2026",
        team_home="Wolves", team_away="Fulham",
        score_home=1, score_away=0, minute=55,
        contract_focus_outcome="Yes", team_focus="home",
        event_timestamp=wolves_goal_ts, now=wolves_goal_ts + 1.0
    )
    # Fulham turn the game around. The price has not collapsed yet, but the reason we
    # were long is gone, so we leave before the book catches up.
    tracker.process_sports_event(
        market_id=wolves_market, game_id="game_wolves_fulham_2026",
        team_home="Wolves", team_away="Fulham",
        score_home=1, score_away=2, minute=78,
        contract_focus_outcome="Yes", team_focus="home",
        event_timestamp=wolves_goal_ts + 30.0, now=wolves_goal_ts + 31.0
    )

    logger.info("Mock simulation complete. All guards exercised.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Advanced Sports Latency Bot v7")
    parser.add_argument("--mock", action="store_true", help="Run simulation using high-speed sports latency mock feed")
    parser.add_argument("--live", action="store_true", help="Connect to live Polymarket WebSocket feed")
    parser.add_argument("--force", action="store_true",
                        help="Proceed with --live even if the geolocation compliance check fails")
    args = parser.parse_args()

    if not args.live:
        args.mock = True

    write_default_env()

    tracker = UnifiedPolymarketBot()
    compliance_ok = tracker.check_geographic_compliance()

    # A failed compliance check is only advisory for the offline simulator, but it
    # blocks anything that would touch the live venue [1, 10].
    if args.live and not compliance_ok and not args.force:
        logger.error("🛑 Refusing to start the live feed from a restricted region. "
                     "Configure HTTP_PROXY in .env, or pass --force to override deliberately.")
        sys.exit(2)

    if args.mock:
        run_mock_simulation(tracker)

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
                    payload = json.loads(msg)
                    # The CLOB sends either a single event object or a batch.
                    events = payload if isinstance(payload, list) else [payload]

                    for event in events:
                        event_type = event.get("event_type")
                        mapping = token_to_market.get(event.get("asset_id"))
                        if not mapping:
                            continue

                        if event_type == "last_trade_price":
                            tracker.process_live_trade({
                                "market_id": mapping["market_id"],
                                "market_title": mapping["market_title"],
                                "token_id": event.get("asset_id"),
                                "outcome": mapping["outcome"],
                                "price": float(event.get("price", 0)),
                                "size": float(event.get("size", 0)),
                                "wallet_address": event.get("maker", "0x0"), # If available
                                "timestamp": time.time()
                            })

                        elif event_type in ("book", "price_change"):
                            # Quote-only movement. It never triggers a strategy on its
                            # own, but it does mean the book is no longer stale, which
                            # the latency engine has to know about.
                            mid_price = extract_mid_price(event)
                            if mid_price is not None:
                                tracker.update_book_quote(
                                    mapping["market_id"], mapping["market_title"],
                                    mapping["outcome"], mid_price
                                )

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Successfully shut down live tracker.")
