# 小额实盘运行手册

当前默认可实盘的策略范围只有普通二元市场的确定性 Yes/No 组合套利。体育延迟、宏观预测和 crypto 统计套利需要单独打开执行开关，并且走方向性单腿执行器，不会塞进买双腿 FOK。即使打开，它们也不是无风险套利。

## 启动前

1. 使用专用小额钱包，不要在仓库、`.env`、日志或 shell 历史中保存私钥。复制 `.env.example` 为 `.env` 后只在本机填写。
2. 确认账户所在地符合 Polymarket 官方地理限制；不要使用代理绕过限制。
3. 用 UV 安装锁定依赖：`uv sync --extra live`（会装上 `polymarket-client==0.6.0`）。
4. 先用 `MARKET_EVENT_LOG` 记录至少一段真实 CLOB 事件，再用 `uv run polywang-replay --markets fixtures/replay/markets.json --events fixtures/replay/events.jsonl` 检查盘口快照、增量 sequence、费用回填和深度。
5. 先以 paper 模式核对机会数量、盘口年龄、可见深度和预期净边际；回放收益不等于可成交收益。
6. 正式启动前先运行 `uv run polywang --live --preflight`；它只检查 geoblock、账户余额/allowance、账本完整性和已有订单恢复，不启动行情流，也不下单。
7. `uv run polywang --health` 读取 `live-health.json`。进程在跑时应周期性更新该文件。

## 首次小额实盘

建议从极小的 `MAX_ORDER_USD`、单市场暴露和总暴露开始，并保留默认的 `LIVE_MAX_BOOK_LEVELS=1`。用 `MERGE_GAS_USD` 把预计 merge 的 gas 计入扫描净收益。先关闭 `AUTO_MERGE_COMPLETE_SETS`，确认双腿成交、User Stream、撤单、回滚、持续对账和条件 token 余额都正常后，再单独验证小额 merge。两腿 FOK 不是原子成交，不能称为无风险套利。

实盘需要显式设置：

```bash
POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK
POLYMARKET_PRIVATE_KEY=<由操作者通过安全方式注入>
```

启动示例：

```bash
POLYMARKET_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK \
POLYMARKET_PRIVATE_KEY="$POLYMARKET_PRIVATE_KEY" \
MAX_ORDER_USD=5 \
LIVE_MAX_TOTAL_EXPOSURE_FRACTION=0.01 \
LIVE_MAX_MARKET_EXPOSURE_FRACTION=0.01 \
uv run polywang --live --markets 20 --max-order 5
```

启动时程序必须通过 geoblock、账户余额/allowance、未完成订单和成交恢复、账本完整性以及风险状态检查。任何 `UNHEDGED`、条件 token 余额不足、对账错误或 kill switch 都应停止新单。

## 日常监控

- 可先运行 `uv run polywang --status --live-journal live-orders.json` 查看 pair 状态、暴露、PnL、未确认结算和 `UNHEDGED` 列表；该命令不联网、不读取私钥。
- `live-orders.json`：确认每个 pair 的两腿订单、实际成交、手续费、交易 hash 和状态。
- `live-risk.json`：确认暴露、每日亏损和 halt 状态没有异常。
- `market-events.jsonl`：保留原始/typed 市场事件和本机接收时间，用于事后回放。
- 重点区分 `HEDGED`、`RESOLVED_PENDING_REDEMPTION` 和 `SETTLED`；市场已判定不等于抵押品已到账。
- User Stream 是实时来源，REST 是兜底；常规对账使用已知订单和增量成交水位，启动及定期恢复轮次会扫描账户内全部 open order 和外部持仓。任何 journal 之外的订单或条件 token 都会 halt。不要把 REST 查询返回的历史成交数量直接当成本轮新增成交。
- 市场频道增量必须连续：`sequence` 断档、`prev_hash` 对不上或未知 `schema_version` 会清空本地盘口，直到下一张 snapshot。丢失增量后不得继续用残缺盘口下单。

## 链上交易超时

`MERGE_SUBMITTED` 或 `REDEEM_SUBMITTED` 表示交易可能已经提交。等待超时后程序会在后续对账中查询原 transaction ID/hash，不会再次提交同一操作；确认成功会自动完成账本。若交易处于未知或失败状态，应通过官方账户/链上记录人工确认，不要手工把 JSON 状态改成 `SETTLED`。

## 停机

创建 `live-kill-switch` 文件或设置 `POLYMARKET_KILL_SWITCH=1` 会持久化停止新单，并立刻撤销账户内所有未完成订单。匹配到的条件 token 库存不会自动市价平仓，会写入 `live-risk.json` / journal 的 `halt_inventory` 供人工处理。收到 SIGINT/SIGTERM 且 `LIVE_CANCEL_ON_SHUTDOWN=1`（默认）时，同样会撤销未完成订单。清除 kill-switch 文件不会自动解除已持久化的风险 halt。

## 方向性策略（默认关闭）

体育、宏观、crypto 默认只观测。若要接入方向性执行器，必须同时满足：

- 对应的 `ENABLE_*_EXECUTION=1`
- 实盘还要 `ENABLE_*_LIVE=1`
- `CalibrationTracker` 样本外 Brier / 漂移闸门通过
- 暴露计入 `live-directional.json`，和组合套利共用 `LiveRiskController`

crypto 的 SELL 进入是买对侧 Polymarket token，退出是卖掉已有库存，不会去中心化交易所做期货对冲。
