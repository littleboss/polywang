# polywang

Polymarket 确定性二元套利与钱包流量情报机器人。

核心目标是先验证可锁定的 Yes/No 组合价差，再将真实钱包流量作为独立的统计确认信号。钱包协同不是无风险套利，也不会绕过确定性套利执行器的风控。

## 快速开始

```bash
# 纸面市场流需要 Python 3.10+、requests 和 websockets；不需要任何密钥
pip install requests websockets

# 跑单元测试
python3 -m unittest -v test_polymarket_edge test_arbitrage_core test_arbitrage_bot test_whale_intelligence test_market_replay test_sports_channel test_predictive_models
```

历史盘口回放使用 Gamma 市场 JSON 和原始 CLOB 事件 JSONL，并复用线上同一套盘口/手续费/深度扫描逻辑：

```bash
python3 market_replay.py --markets markets.json --events market-events.jsonl --consume-fills
```

live 或 paper 运行时设置 `MARKET_EVENT_LOG=market-events.jsonl` 可记录收到的 raw/typed market event、源类型和本机接收时间，之后可直接作为盘口回放输入。记录器只保存事件，不会把订单响应或成交假设写成历史成交；成交真实性仍需单独保存 `live-orders.json` 并做对账。

回放结果只是“按记录盘口可见的机会报告”，还需要用真实订单回报校验成交率、延迟、拒单和实际手续费，不能把回放净收益直接当成可实现 PnL。启用 `--consume-fills` 和执行延迟后，`executed_net_profit` 只代表通过模拟深度、价格上限和新鲜度检查的回放成交；顶层 `net_profit` 仍是所有可见信号的汇总，不能替代真实 PnL。

`sports_channel.py` 提供官方 Sports Channel 消费器、延迟闸门，以及可选的 `SPORTS_MARKET_MAP` 比赛-市场映射和粗略公允价值。映射后的候选只记日志，`executable` 恒为 false，不会交给二元 FOK 执行器。缺少源时间戳时只记录观察。

`macro_model.py` 和 `crypto_model.py` 只提供带时间戳的预测信号接口，不会自动下单。宏观模型会按 `event_id` 去重，并可绑定指标到市场；crypto 模型有退出 z-score 和库存上限，`SELL_MARKET` 不能走当前的买双腿 FOK 执行器。两者都要求通过持久化 `CalibrationTracker` 的滚动窗口、样本外 Brier 和漂移检测；没有独立数据源、真实结算回填和回放结果时，信号保持不可交易。

## 确定性二元套利引擎

项目现在另外提供一个独立的、默认纸面交易的二元套利引擎：

```bash
python3 arbitrage_bot.py --markets 100 --cash 1000
```

它只研究 Yes 和 No 两腿同时买入后合计支付 1.00 的二元市场。机会必须使用订单簿中的实际 ask 和深度，并覆盖两腿 taker 费用、可选 merge gas（`MERGE_GAS_USD`）、资金安全缓冲和最大仓位；纸面成交写入 `paper-ledger.json`，程序重启后会恢复账本。两腿仍是顺序 FOK，不是原子交易，`is_risk_free` 恒为 false。

新引擎只有在显式设置 `POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK`、官方 geoblock 放行、私钥存在且 `polymarket-client` 可用时，才会使用两腿 FOK 执行器。安装和验证官方客户端后才考虑运行：

```bash
.venv/bin/python -m pip install polymarket-client
POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK \
  .venv/bin/python arbitrage_bot.py --live --markets 100
```

正式启动前可先只做账户和账本检查，不启动行情流、不下单：

```bash
POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK \
  .venv/bin/python arbitrage_bot.py --live --preflight --markets 20
```

实盘启动顺序是：地理限制检查 → 官方 SDK 账户 preflight → 读取未完成 pair 的 open order / account trades / order 状态 → 核验两腿 conditional-token 余额 → 启动官方 typed market stream、私有 user stream 和持续对账任务。市场流断线后会废弃本地盘口，必须重新收到快照才恢复扫描；增量若出现 sequence 断档、hash 链断裂或未知 schema 版本，同样会清空该 token 的盘口。持续对账失败会持久化风险 halt，并撤销账户内全部 open order。启动及定期恢复轮次会扫描 journal 之外的 open order 和条件 token 持仓。实盘 pair 日志默认写入 `live-orders.json`，也可用 `--live-journal PATH` 指定。日志会保存 pair ID、condition/token ID、两腿订单 ID、请求数量、已确认成交数量、实际成交价格、手续费字段、trade ID、交易哈希、回滚状态和 `PENDING/HEDGED/RESOLVED_PENDING_REDEMPTION/SETTLED/UNHEDGED` 状态。

实盘还会从 `live-risk.json` 恢复风险状态。默认总暴露上限为账户权益的 25%，单市场上限为 5%，最多 10 个未结 pair；可分别用 `LIVE_MAX_TOTAL_EXPOSURE_FRACTION`、`LIVE_MAX_MARKET_EXPOSURE_FRACTION`、`LIVE_MAX_OPEN_PAIRS` 和 `LIVE_RISK_EQUITY_USD` 配置。User Stream 负责实时成交回报，REST 对账默认每 15 秒检查已知订单、增量成交和余额，并每 `LIVE_FULL_RECOVERY_CYCLES` 轮（默认 20 轮）做一次 open-order 孤儿恢复；启动时仍会做完整恢复。创建 `live-kill-switch` 文件，或设置 `POLYMARKET_KILL_SWITCH=1`，会持久化停止新单并撤销所有未完成订单；已成交库存不会自动市价甩卖。清除文件不会自动解除已经持久化的 halt，必须人工检查 `live-risk.json` 后再决定是否恢复。已提交但等待超时的 merge/redeem 交易会在后续对账中查询其原 transaction ID/hash，确认成功后自动完成账本，绝不会重新提交。

两腿中的第一腿成交而第二腿失败时，程序会使用 FAK 反向平掉第一腿；如果回滚不完整，或私有 user stream / 重启对账发现一条腿成交而另一条腿不足，程序会停止并要求人工对账。`HEDGED` 只表示两腿成交数量均已被确认，市场结算后才变成 `SETTLED`；订单被接受不等于成交。425 matching-engine restart 会有限次数退避重试，503 cancel-only/post-only 状态不会盲目重试。

对于普通（二元、非 NegRisk）市场，可显式设置 `AUTO_MERGE_COMPLETE_SETS=1`。当两腿成交均确认后，程序会调用官方 `merge_positions(condition_id=..., amount=...)` 将完整 Yes/No 集合合并回抵押品，并等待链上交易哈希；链上余额尚未到账或交易未确认时只保留 `HEDGED`，不会提前记为已结算。市场发布 resolution 时，pair 会进入 `RESOLVED_PENDING_REDEMPTION`，程序再调用官方 `redeem_positions(condition_id=...)`；只有赎回交易确认后才变为 `SETTLED`、释放风险预算并计算已实现 PnL。赎回交易一旦提交即持久化为 `REDEEM_SUBMITTED`，等待超时不会重复提交，需通过链上或账户对账完成确认。合并和赎回交易的 gas/账户配置仍需在实盘前用小额账户验证。

官方 SDK 的 collateral allowance 以最小单位返回，且可能包含多个 spender。若返回多个 spender，必须显式设置正确的 `POLYMARKET_ALLOWANCE_SPENDER`，否则实盘会拒绝启动。程序不会自动无限批准；只有明确设置 `POLYMARKET_SETUP_APPROVALS=1` 才会调用官方 `setup_trading_approvals()`。实盘前仍必须使用专用小额账户验证余额、allowance、订单状态和结算。

不能把这种回滚当成无损保证，也不要把私钥写入仓库，更不要用代理绕过 Polymarket 地理限制。

这个引擎是确定性套利，不等于体育、宏观或加密预测策略。后者必须分别接入独立数据源并完成样本外校准，不能把任意 confidence 分数当作概率。

## 大鲸情报层

`whale_intelligence.py` 是独立的钱包流量情报层，由新引擎消费公开市场交易事件。它会过滤匿名或格式错误的钱包地址，按 `trade_id` 去重，保存每个钱包的交易量、持仓、结算次数、已实现 PnL 和收缩后的质量分数，并计算最近窗口内按钱包质量加权的市场净买卖压力。

默认只有同时满足以下条件的交易才会被标记为可跟随的大鲸流：名义金额达到阈值、钱包有足够的历史结算样本、历史质量分数过线、近期压力方向一致。协同信号还必须属于同一 outcome、同一方向，达到最少钱包数和总金额，并限制最大单钱包金额占比；至少需要一个历史上合格的钱包。否则只记录为 `LARGE TRADE` 观察，不再把它升级为大鲸反转交易。钱包历史默认保存到 `whale-intelligence.json`，可通过 `WHALE_STATE_PATH` 修改。

注意：公开 CLOB 行情事件经常没有真实交易者地址。没有地址时系统只承认“大额匿名交易”，不会把多个匿名事件当成多个钱包，也不会据此生成高质量大鲸信号。数据适配器必须提供 `wallet_address`、`trade_id`、`side` 和结算结果。

## 先看这个：手续费不是固定百分比

Polymarket 的[官方费率公式](https://docs.polymarket.com/trading/fees)是：

```
fee = 股数 × 费率 × p × (1 - p)      # 只对 taker 收取，maker 免费
```

换算成占投入资金的比例就是 **`费率 × (1 - p)`**。也就是说手续费在 p=0.50 时最贵，
越靠近 0 或 1 越便宜：

| 价格 | 体育(0.05) | 政治(0.04) | 地缘政治 |
|------|-----------|-----------|---------|
| 0.10 | 4.50% | 3.60% | 0% |
| 0.50 | 2.50% | 2.00% | 0% |
| 0.90 | 0.50% | 0.40% | 0% |
| 0.97 | 0.15% | 0.12% | 0% |

三条直接影响策略的推论：

1. **高概率合约的手续费极低**，用固定百分比建模会把 0.97 的成本高估 10 倍。
2. **挂限价单完全免手续费**，穿价成交的手续费本质上是「立刻成交」的价格。只有当
   机会会在等待中消失时（比如刚进球）才值得付。
3. **持有到结算比提前平仓便宜**，因为结算不收费，而平仓要再付一次 taker 费。

如果使用环境变量文件，请由操作者自行创建 `.env`，并确保它不提交到 git。**`.env` 包含私钥，绝对不要提交到 git。**

小额实盘的启动、监控、链上交易超时和停机流程见 [LIVE_RUNBOOK.md](LIVE_RUNBOOK.md)。

查看本地 live journal 状态（不联网、不读取私钥）：`python3 arbitrage_bot.py --status --live-journal live-orders.json`。

## 安全防线

**边际护栏（`polymarket_edge.py`）** — 一笔交易要同时越过三道门槛：

| 门槛 | 拦截的亏损方式 |
|------|--------------|
| `MIN_NET_PROFIT_MARGIN` | 手续费吃掉全部利润（高胜率仍亏钱） |
| `HURDLE_APR` | 交易赚钱但太慢，资金机会成本更高 |
| `MIN_EDGE_OVER_BREAKEVEN` | 声称的精度超出模型能力 |

第三条最容易被忽略。在 0.97 买入，盈亏平衡概率是 0.9715，声称有边际就等于声称
自己能算准到小数点后三位。要求一个看得见的余量，能让机器人远离它其实无法判断的赌注。

**凯利仓位** — 二元合约的凯利比例可以化简为 `f* = (q - p) / (1 - p)`。价格越高分母
越小：在 0.97 处分母只有 0.03，估计值差一个百分点，建议仓位就摆动三分之一本金。
所以估计值会先向市场价收缩，再取四分之一凯利，最后还有硬上限。

**限价单保护** — 每笔订单都带 `max_allowed_fill_price` 上限。执行前会重新读一次
订单薄：如果市场在信号产生和下单之间已经涨过上限，订单作废。live 模式默认只使用每条腿的最佳 ask 档位（可用 `LIVE_MAX_BOOK_LEVELS` 显式调整）；BUY 的 SDK 参数是花费金额，若价格改善导致成交份额超过目标，程序会拒绝该 pair 并尝试回滚。延迟套利的前提就是
市场还没反应，一旦反应了就不再是套利，而是高位接盘。

**校准反馈** — 每个策略的预测都会记 Brier 分数（均方误差，越低越好）。0.25 是"所有
问题都答 50%"的得分，超过它说明这个策略还不如不预测，会被自动停用。没有这个回路，
就无法区分"真的有效"和"运气好"。

**地理合规** — 启动时检查 IP 归属地。检查失败时模拟模式仅告警，但会直接阻断
`--live`。

## 关于"抓高概率机会"

高概率不等于高收益，这两件事经常被混为一谈：

- **赔付极度不对称。** 在 0.97 买入是拿 0.97 去博 0.03。输一次要 34 次盈利才能补回来。
- **真正的门槛是精度。** 盈亏平衡在 0.9715，你必须比市场准 0.15 个百分点。
- **资金被锁住。** 3% 的收益，两天兑现是极好的生意，半年兑现还不如放着不动。

但高概率确实有一个结构性优势，而且被现有代码错过了：**手续费在价格极端处趋近于零**。
所以正确的做法不是回避高概率合约，而是：

1. 只做**短期内结算**的高概率合约（体育赛事内、当日经济数据），把年化拉起来。
2. **挂限价单**，手续费直接归零，同时拿 maker 返佣。
3. 用**凯利公式**控制仓位，因为高价位处凯利极不稳定，必须收缩。
4. 优先选**地缘政治类市场**，完全免手续费。

另外有一类机会不需要预测任何事：**NegRisk 多结果市场**。互斥结果的 YES 价格总和
必须等于 1.00，不等于时差额可以锁定。但注意每一条腿都要付一次 taker 费，10 条腿
的组合光手续费就 3%，所以这个策略**只有挂单才做得通**。`NegRiskScanner` 会同时
给出 taker 和 maker 两种口径。

## 需要定期校准的参数

| 参数 | 默认值 | 什么时候要改 |
|------|--------|-------------|
| `MARKET_CATEGORY` | sports | 决定费率。必须和实际交易的市场类别一致 |
| `MIN_NET_PROFIT_MARGIN` | 0.02 | 每投入一美元要求的最低净收益 |
| `HURDLE_APR` | 0.15 | 资金机会成本。你有更好的去处就调高 |
| `MIN_EDGE_OVER_BREAKEVEN` | 0.02 | 模型精度的自我认知。模型越粗糙应该调得越高 |
| `KELLY_FRACTION` | 0.25 | 凯利比例。全凯利在高价位会爆仓，不要调到 1.0 |
| `ESTIMATE_CONFIDENCE` | 0.5 | 你的估计相对市场价的可信度权重 |
| `WHALE_THRESHOLD_USD` | 5000 | 市场流动性上升时调高 |
| `WHALE_MIN_COORDINATION_TRADE_USD` | 500 | 协同中的单笔最小金额 |
| `WHALE_MAX_CONCENTRATION` | 0.75 | 单一钱包最多占协同金额的比例 |

费率**不再需要手工配置**：如果 Gamma 返回逐市场 `feeSchedule`，扫描器优先使用该费率和指数；否则才由市场类别的 `CATEGORY_TAKER_FEE_RATES` 提供缺省值。

## 文件结构

| 文件 | 职责 |
|------|------|
| `arbitrage_bot.py` | 市场流、钱包流、纸面交易和实盘启动编排 |
| `arbitrage_core.py` | 二元套利扫描、账本、官方 FOK 执行、对账和订单状态机 |
| `polymarket_edge.py` | 费率模型、边际评估、凯利仓位、校准跟踪、NegRisk 扫描 |
| `whale_intelligence.py` | 钱包过滤、历史质量、质量加权净流向、结算反馈和持久化 |
| `market_replay.py` | JSONL 盘口历史回放与机会统计 |
| `sports_channel.py` | Sports Channel 消费与时间戳延迟闸门 |
| `macro_model.py` | 宏观事件 surprise 概率模型与校准闸门 |
| `crypto_model.py` | crypto 市场概率价差 z-score 模型与校准闸门 |
| `test_*.py` | 单元测试，只用标准库、不联网 |

## 修改代码时的注意事项

1. 先跑 paper mode，再跑 `python3 -m unittest`。
2. 费率测试是**直接对照官方费率表**写的，不是对照实现写的。如果公式被改错，测试会发现。
3. 实盘下单必须确认订单响应、私有 user stream 和重启对账；不能把订单被接受当作双腿成交。
4. `debias_market_price` 用 Wang 变换估计物理概率（λ≈0.176，基于已结算合约标定）。
   **它默认不启用**，因为该偏差在高成交量市场趋近于零，对流动性好的市场套用固定 λ
   会凭空造出并不存在的边际。
