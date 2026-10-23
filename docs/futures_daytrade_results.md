# 股票期貨空方當沖結果 / Stock-Futures Short Day-Trade Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.15](experiment_protocol.md) 的判定：用期交所官方的股票期貨開盤價與日結算價，把 B0 `t-lgbm-alpha` 與 D0 `t-xs-scratch` 的 h = 1 分數變成空方當沖（前一收盤排名最低的 q 分位，次日期貨開盤放空、結算價平倉），含多空版、台指期對沖版、六個資本情境與成本敏感度。

結論：

1. **訊號是真的。** 以實際期貨價格計算，空方版（q = 10%，每日約 13 檔）的年化毛報酬 B0 +61.5%、D0 +64.5%（P1 ∪ P2，777 個交易日），P1 與 P2 都為正（B0 +59%／+67%，D0 +69%／+53%）。每檔每日毛價差約 0.24%。
2. **H-E 未通過，而且差距很大。** 主要成本情境（期交稅 0.002%、手續費每口 30 元、每邊 1 tick 滑價）下年化淨報酬 B0 −63.4% [−91.5%, −36.0%]、D0 −60.5% [−86.9%, −33.7%]，Sharpe −2.3，最大回撤 −87% 到 −89%，勝率 42%。成本拖累約 125%／年，幾乎全部來自 tick 滑價：期交稅加手續費每邊只有 0.02%，但股票期貨 tick 在 100–500 元價位是 0.5 元、每邊約 0.1–0.5%，平均每邊約 0.23%。
3. **損益兩平的單邊成本約 0.12%**，是模型假設（1 tick）的一半。換句話說，這條路線成立的前提是能以市價單以外的方式（掛單、不穿價）把每邊成交成本壓到半個 tick 以下，而本協定的成本假設不允許。
4. **多空版更差、對沖版沒救回來。** 多前十分位的多方腿毛報酬接近零（與策略層報告第 7 節一致），卻多付一倍成本，多空版淨報酬 −76%／−78%；加台指期多頭的對沖版毛報酬升到 +72%／+75%，淨報酬仍為 −57%／−54%。
5. **q = 20% 比 10% 差**（毛報酬 +35–37%、淨 −88–90%），訊號集中在最差的一成。
6. **成本敏感度全部為負。** 2 tick 滑價時淨報酬 −175%；手續費 60 元時 −76%。
7. **資本情境**：1,000 萬元每日約 69 口、11% 的名單買不到一口、名目部署率 74%；1 億元 463 口、部署率 73%；10 億元 1,552 口但受 5% 參與率限制只部署 33%。保證金使用率 7–16%（以最高級距 20.25% 計）。沒有資本情境被標記為不可行，但也沒有一個淨報酬為正。
8. **時點噪音**：期貨開盤價對現股開盤價的對數差標準差 1.04%（8:45 對 9:00）、結算價對現股收盤價 0.49%；以期貨價格算出的毛報酬（+61%）低於以現股價格估計的空方價差（約 +90% 到 +110%），差額是期貨開盤已反映部分隔夜資訊。
9. **依 §13.15 第 9 點的處置**：h = 1 訊號在現有工具與成本下也不可交易；模型與策略層回到研究狀態，下一步是訊號本身的登記實驗（§13.16 殘差化標籤與多 horizon 融合，已登記）。前瞻候選模型：無。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 期貨資料 | 期交所每日行情下載（全契約，2022-12 至 2026-06）：每契約每日開盤價、結算價、成交量，一般時段；近月契約在到期日當天改用次月；標準契約（2,000 股）優先 |
| 標的對應 | 期交所股票期貨標的清單（2026-10-20 快照）：契約代碼 → 股票代號與每口股數；314 個契約代碼有資料、320 個在清單中、38 個三碼代碼無法對應（已下市，存活者偏誤） |
| 宇宙 | 訊號日近月契約有一般時段成交、且近月 20 日平均成交量 ≥ 100 口（標準口換算）；每日約 130 檔；沒有開盤價的標的跳過（占 0.04%） |
| 訊號 | B0、D0 的 h = 1 集成分數，宇宙內排名；q ∈ {10%, 20%}，檔數 ⌈n × q⌉ |
| 部位 | 空方版：放空後 q 分位；多空版：加做多前 q 分位等額名目；對沖版：空方版加等額名目的台指期多頭（近月，開盤到結算價） |
| 價格 | 期貨開盤價（8:45）進場、當日結算價平倉；不留倉 |
| 資本 | 分數位置（主分析）；1,000 萬、3,000 萬、1 億、3 億、10 億元：總名目 = 資本，每檔名目 = 總名目／檔數，口數取整，每檔 ≤ 當日成交量 5%；保證金以最高級距 20.25% 計 |
| 成本（每邊） | 期交稅 0.002% × 名目；手續費每口 30 元（另報 60 元）；滑價 1 tick（另報 2 tick），tick 依價位：< 10 元 0.01、10–50 元 0.05、50–100 元 0.1、100–500 元 0.5、500–1,000 元 1、≥ 1,000 元 5；台指期每邊 0.002% + 50 元 + 1 點 |
| 指標 | 每日報酬 = 損益／資本；年化 × 252；區間為每日淨報酬的 circular block bootstrap（區塊 5，1,000 次，seed 11）；期間依訊號日 P1、P2、P1 ∪ P2 |
| 選擇與判定 | q 以 P1 空方版淨 Sharpe 選定（兩個模型都選 10%）；H-E：P1 ∪ P2 年化淨報酬區間下界 > 0 且 P2 淨報酬 > 0 |

### 3. 結果

#### 3.1 分數位置、主要成本（q = 10%）

| 模型 | 版本 | 期間 | 年化毛報酬 | 年化淨報酬 | 95% 區間 | Sharpe | 最大回撤 | 勝率 | 成本拖累／年 | 平均檔數 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 空方 | P1 | +59.2% | −66.0% | [−95.3%, −33.9%] | −2.65 | −79.2% | 41.6% | 125% | 12.9 |
| B0 | 空方 | P2 | +67.5% | −56.6% | [−121.7%, +6.8%] | −1.75 | −50.6% | 41.9% | 124% | 14.9 |
| B0 | 空方 | P1 ∪ P2 | +61.5% | −63.4% | [−91.5%, −36.0%] | −2.34 | −89.4% | 41.7% | 125% | 13.4 |
| B0 | 多空 | P1 ∪ P2 | +49.4% | −76.0% | [−89.0%, −63.4%] | −6.35 | −90.9% | 33.2% | 125% | 26.8 |
| B0 | 對沖 | P1 ∪ P2 | +71.7% | −57.2% | [−80.5%, −33.5%] | −2.52 | −86.5% | 41.3% | 129% | 13.4 |
| D0 | 空方 | P1 | +69.0% | −56.6% | [−84.8%, −27.7%] | −2.34 | −74.3% | 41.8% | 126% | 12.9 |
| D0 | 空方 | P2 | +52.6% | −70.8% | [−128.4%, −12.1%] | −2.22 | −51.0% | 41.4% | 123% | 14.9 |
| D0 | 空方 | P1 ∪ P2 | +64.5% | −60.5% | [−86.9%, −33.7%] | −2.28 | −87.0% | 41.7% | 125% | 13.4 |
| D0 | 多空 | P1 ∪ P2 | +47.7% | −78.0% | [−90.8%, −65.8%] | −6.67 | −91.2% | 32.4% | 126% | 26.8 |
| D0 | 對沖 | P1 ∪ P2 | +74.7% | −54.3% | [−76.4%, −31.0%] | −2.44 | −83.4% | 42.2% | 129% | 13.4 |

q = 20% 的空方版：B0 毛 +37.3%、淨 −87.8%；D0 毛 +35.2%、淨 −90.1%。

#### 3.2 成本敏感度（空方版，q = 10%，分數位置，P1 ∪ P2）

| 模型 | 手續費／口 | 滑價 | 年化毛報酬 | 年化淨報酬 | Sharpe |
| --- | --- | --- | --- | --- | --- |
| B0 | 30 元 | 1 tick | +61.5% | −63.4% | −2.34 |
| B0 | 30 元 | 2 tick | +61.5% | −174.6% | −6.43 |
| B0 | 60 元 | 1 tick | +61.5% | −76.0% | −2.80 |
| D0 | 30 元 | 1 tick | +64.5% | −60.5% | −2.28 |
| D0 | 30 元 | 2 tick | +64.5% | −171.3% | −6.46 |
| D0 | 60 元 | 1 tick | +64.5% | −73.6% | −2.78 |

毛報酬換算每日每檔約 0.24%，損益兩平的單邊成本約 0.12%；主要情境的實際單邊成本約 0.25%（其中 tick 滑價約 0.23%）。

**價差實測（事後檢查 1 tick 假設）**：期交所每日行情檔含「最後最佳買價／賣價」。取 2025 到 2026 年每三週抽一週、近月股票期貨一般時段的收盤報價（32,107 筆，其中成交量 ≥ 100 口的流動契約 20,109 筆）：流動契約的收盤價差中位數 2 tick（47% 為 1 tick、71% 在 2 tick 以內），市價單付的半價差每邊中位數 0.174%、平均 0.242%，與本協定假設的 1 tick／每邊（中位數 0.180%、平均 0.215%）幾乎相同；依價位級距，100–500 元半價差 0.18%、500–1,000 元 0.14%、10–50 元 0.17%。1 tick／每邊因此是收盤時實際報價的水準，不是偏嚴的假設；8:45 開盤時的價差只會更寬，日資料看不到。

#### 3.3 資本情境（空方版，q = 10%，主要成本，P1 ∪ P2）

| 模型 | 資本 | 年化淨報酬 | Sharpe | 平均口數／日 | 買不到一口的名單 | 參與率被擋 | 名目部署率 | 保證金使用率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 分數位置 | −63.4% | −2.34 | — | 0% | 0% | 100% | 20.3% |
| B0 | 1,000 萬 | −45.6% | −2.16 | 69 | 11.2% | 0.2% | 74% | 15.1% |
| B0 | 3,000 萬 | −56.5% | −2.41 | 185 | 3.9% | 0.2% | 80% | 16.3% |
| B0 | 1 億 | −62.6% | −2.72 | 463 | 0.2% | 0.3% | 73% | 14.7% |
| B0 | 3 億 | −60.4% | −3.03 | 922 | 0% | 0.3% | 56% | 11.3% |
| B0 | 10 億 | −45.7% | −3.16 | 1,552 | 0% | 0.3% | 33% | 6.7% |
| D0 | 分數位置 | −60.5% | −2.28 | — | 0% | 0% | 100% | 20.3% |
| D0 | 1,000 萬 | −46.3% | −2.19 | 72 | 8.9% | 0.2% | 76% | 15.5% |
| D0 | 1 億 | −57.4% | −2.60 | 475 | 0.1% | 0.3% | 71% | 14.3% |
| D0 | 10 億 | −39.8% | −3.02 | 1,506 | 0% | 0.3% | 30% | 6.1% |

資本越大、部署率越低（參與率上限），虧損的絕對值跟著變小，但 Sharpe 更差；沒有情境為正。

### 4. 意涵

- 這是四條策略路線中毛報酬最高、也是最乾淨的一次驗證：用可成交的期貨價格、不需券源、不留倉，訊號在 P1 與 P2 都存在。失敗的唯一原因是股票期貨的買賣價差：tick 相對價格太寬，每天進出一次就吃掉 0.5%。
- 要讓它成立，需要的不是更好的模型，而是執行層：以限價單在 8:45–9:00 之間被動成交、只做 tick 相對價格最小的價位帶（例如 500–1,000 元、tick 1 元為 0.1–0.2%），或改用現股當沖加借券（成本 0.52% 來回，一樣不夠）。這些都是執行假設，無法用日資料回測驗證，需要盤中資料或實單。
- 多方腿沒有價值的結論再次被確認；對沖版的台指期多頭只是加上市場 beta。
- 依協定，模型層回到訊號本身：§13.16 的殘差化標籤若能救回多日 horizon，低換手的多頭或多空組合才有重新成立的可能。

### 5. 限制

- 契約代碼以 2026-10 的清單對應，期間內下市的股票期貨無法納入（存活者偏誤）。
- 8:45 開盤價可成交但流動性薄；1 tick 滑價是對市價單的保守假設，被動成交的情境沒有模擬。
- 結算價是官方 mark，實際平倉在 13:30 前後成交。
- 保證金以最高級距一律計算，未模擬盤中追繳。
- q 在 P1 選擇，P1 ∪ P2 判定含 P1。

### 6. 重現方式

```bash
cd backtest
uv run python -m twbacktest.taifex download --config config/capturability.json   # 全契約（已有 2025-05 至 2025-12 的根目錄時另建 taifex-all）
uv run python -m twbacktest.futures_daytrade --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `216fd7a`（§13.15） |
| 程式 commit | `ddfeb21`（`twbacktest.futures_daytrade`、`taifex.stock_futures_sessions`、設定檔） |
| 設定檔 | `backtest/config/futures_daytrade.json` |
| 期貨資料（不受 Git 追蹤） | `artifacts/diagnostics/taifex-all/`（188 個週檔 + 標的清單）、`artifacts/diagnostics/taifex-tx/` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/futures-daytrade-b0-d0-20261023T090019Z/`（`report.json`、`grid.csv`、`daily.csv`、`summary.md`） |

## English

### 1. Purpose and conclusions

This report records the decision of [experiment protocol §13.15](experiment_protocol.md): the h = 1 scores of B0 `t-lgbm-alpha` and D0 `t-xs-scratch` turned into a short-side day trade on official TAIFEX stock-futures opens and daily settlements (the bottom q of the previous close's ranking, shorted at the next session's futures open and covered at its settlement), with long-short and TX-hedged versions, six capital scenarios and cost sensitivities.

Conclusions:

1. **The signal is real.** On actual futures prices the short-only version (q = 10%, about 13 names a day) earns an annualized gross return of +61.5% (B0) and +64.5% (D0) over P1 ∪ P2 (777 sessions), positive in both P1 and P2 (B0 +59% / +67%, D0 +69% / +53%). The gross spread per name is about 0.24% a day.
2. **H-E fails, by a wide margin.** Under the primary costs (0.002% tax, NT$30 commission per contract, 1 tick slippage per side) the annualized net return is −63.4% [−91.5%, −36.0%] for B0 and −60.5% [−86.9%, −33.7%] for D0, Sharpe −2.3, maximum drawdown −87% to −89%, hit rate 42%. The cost drag of about 125% a year is almost entirely tick slippage: tax plus commission is only 0.02% per side, but the stock-futures tick is NT$0.5 in the NT$100–500 band, 0.1–0.5% per side, about 0.23% per side on average.
3. **The break-even cost per side is about 0.12%**, half the modelled tick. The route works only if executions can be filled passively (resting orders, no crossing) at under half a tick per side, which this protocol's cost assumptions do not allow.
4. **Long-short is worse and the hedge does not rescue it.** The long leg on the top decile earns almost nothing gross (consistent with section 7 of the strategy-layer report) while doubling costs: long-short net −76% / −78%. Adding a TX long raises the gross to +72% / +75% but the net stays at −57% / −54%.
5. **q = 20% is worse than 10%** (gross +35–37%, net −88–90%): the signal sits in the worst decile.
6. **Every cost sensitivity is negative.** Two ticks of slippage give −175%; a NT$60 commission −76%.
7. **Capital scenarios**: NT$10 million trades about 69 contracts a day, 11% of names cannot afford one contract, 74% of the notional deploys; NT$100 million 463 contracts, 73% deployed; NT$1 billion 1,552 contracts but only 33% deployed under the 5% participation cap. Margin usage is 7–16% (highest tier, 20.25%). No scenario is flagged infeasible, and none is net positive.
8. **Timing noise**: the log difference between the futures open and the stock open has a 1.04% standard deviation (08:45 vs 09:00), the settlement against the stock close 0.49%; the gross on futures prices (+61%) is below the short-side spread estimated on stock prices (about +90% to +110%) because the futures open already reflects part of the overnight information.
9. **Disposition under §13.15 item 9**: the h = 1 signal is not tradeable with the available instruments and costs either; model and strategy layers return to research, and the next step is the registered experiment on the signal itself (§13.16 residualized labels and multi-horizon blending). Forward candidate: none.

### 2. Data and method

| Item | Content |
| --- | --- |
| Futures data | TAIFEX daily download of every contract (2022-12 to 2026-06): daily open, settlement and volume per contract, regular session; the near month switches to the next month on its expiry day; the standard contract (2,000 shares) is preferred |
| Underlying mapping | TAIFEX stock-futures underlying list (2026-10-20 snapshot): contract code → symbol and shares per contract; 314 codes have data, 320 are listed, 38 three-letter codes cannot be mapped (delisted, survivorship bias) |
| Universe | The near-month contract traded in the regular session on the signal date and its 20-session mean volume is ≥ 100 contracts (standard equivalents); about 130 names a day; names without an open are skipped (0.04%) |
| Signal | The h = 1 ensemble scores of B0 and D0, ranked within the universe; q ∈ {10%, 20%}, ⌈n × q⌉ names |
| Positions | Short-only: short the bottom q; long-short: plus an equal-notional long of the top q; hedged: short-only plus an equal-notional TX long (front month, open to settlement) |
| Prices | Enter at the futures open (08:45), cover at the daily settlement; no overnight position |
| Capital | Fractional (primary); NT$10 million, NT$30 million, NT$100 million, NT$300 million, NT$1 billion: total notional = capital, notional per name = total / names, whole contracts, each name ≤ 5% of the day's volume; margin at the highest tier, 20.25% |
| Costs (per side) | 0.002% tax on notional; NT$30 commission per contract (NT$60 also); 1 tick slippage (2 ticks also), tick by price band: < NT$10 0.01, 10–50 0.05, 50–100 0.1, 100–500 0.5, 500–1,000 1, ≥ 1,000 5; TX per side 0.002% + NT$50 + 1 point |
| Metrics | Daily return = P&L / capital; annualized × 252; intervals from a circular block bootstrap of daily net returns (block 5, 1,000 replicates, seed 11); periods P1, P2, P1 ∪ P2 by signal date |
| Selection and decision | q chosen by the P1 short-only net Sharpe (both models select 10%); H-E: P1 ∪ P2 annualized net return lower bound > 0 and P2 net return > 0 |

### 3. Results

#### 3.1 Fractional capital, primary costs (q = 10%)

| Model | Version | Period | Gross return (annual) | Net return (annual) | 95% CI | Sharpe | Max drawdown | Hit rate | Cost drag / year | Mean names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | short-only | P1 | +59.2% | −66.0% | [−95.3%, −33.9%] | −2.65 | −79.2% | 41.6% | 125% | 12.9 |
| B0 | short-only | P2 | +67.5% | −56.6% | [−121.7%, +6.8%] | −1.75 | −50.6% | 41.9% | 124% | 14.9 |
| B0 | short-only | P1 ∪ P2 | +61.5% | −63.4% | [−91.5%, −36.0%] | −2.34 | −89.4% | 41.7% | 125% | 13.4 |
| B0 | long-short | P1 ∪ P2 | +49.4% | −76.0% | [−89.0%, −63.4%] | −6.35 | −90.9% | 33.2% | 125% | 26.8 |
| B0 | hedged | P1 ∪ P2 | +71.7% | −57.2% | [−80.5%, −33.5%] | −2.52 | −86.5% | 41.3% | 129% | 13.4 |
| D0 | short-only | P1 | +69.0% | −56.6% | [−84.8%, −27.7%] | −2.34 | −74.3% | 41.8% | 126% | 12.9 |
| D0 | short-only | P2 | +52.6% | −70.8% | [−128.4%, −12.1%] | −2.22 | −51.0% | 41.4% | 123% | 14.9 |
| D0 | short-only | P1 ∪ P2 | +64.5% | −60.5% | [−86.9%, −33.7%] | −2.28 | −87.0% | 41.7% | 125% | 13.4 |
| D0 | long-short | P1 ∪ P2 | +47.7% | −78.0% | [−90.8%, −65.8%] | −6.67 | −91.2% | 32.4% | 126% | 26.8 |
| D0 | hedged | P1 ∪ P2 | +74.7% | −54.3% | [−76.4%, −31.0%] | −2.44 | −83.4% | 42.2% | 129% | 13.4 |

Short-only with q = 20%: B0 gross +37.3%, net −87.8%; D0 gross +35.2%, net −90.1%.

#### 3.2 Cost sensitivity (short-only, q = 10%, fractional, P1 ∪ P2)

| Model | Commission / contract | Slippage | Gross return (annual) | Net return (annual) | Sharpe |
| --- | --- | --- | --- | --- | --- |
| B0 | NT$30 | 1 tick | +61.5% | −63.4% | −2.34 |
| B0 | NT$30 | 2 ticks | +61.5% | −174.6% | −6.43 |
| B0 | NT$60 | 1 tick | +61.5% | −76.0% | −2.80 |
| D0 | NT$30 | 1 tick | +64.5% | −60.5% | −2.28 |
| D0 | NT$30 | 2 ticks | +64.5% | −171.3% | −6.46 |
| D0 | NT$60 | 1 tick | +64.5% | −73.6% | −2.78 |

The gross return is about 0.24% per name per day, so the break-even cost per side is about 0.12%; the primary scenario's actual cost per side is about 0.25%, of which tick slippage is about 0.23%.

**Spread census (post-hoc check of the 1-tick assumption)**: the TAIFEX daily files carry the last best bid and ask. Sampling every third week of 2025–2026, near-month stock futures in the regular session (32,107 rows, of which 20,109 liquid contracts with volume ≥ 100), the liquid contracts' closing quoted spread has a median of 2 ticks (47% at 1 tick, 71% within 2 ticks); the half-spread a market order pays per side has a median of 0.174% and a mean of 0.242%, almost identical to this protocol's 1 tick per side (median 0.180%, mean 0.215%); by price band the half-spread is 0.18% at NT$100–500, 0.14% at NT$500–1,000 and 0.17% at NT$10–50. One tick per side is therefore the level of the actual closing quotes, not a harsh assumption; spreads at the 08:45 open can only be wider, which daily data cannot show.

#### 3.3 Capital scenarios (short-only, q = 10%, primary costs, P1 ∪ P2)

| Model | Capital | Net return (annual) | Sharpe | Mean contracts / day | Names below one contract | Participation-blocked | Deployed notional | Margin usage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | fractional | −63.4% | −2.34 | — | 0% | 0% | 100% | 20.3% |
| B0 | NT$10M | −45.6% | −2.16 | 69 | 11.2% | 0.2% | 74% | 15.1% |
| B0 | NT$30M | −56.5% | −2.41 | 185 | 3.9% | 0.2% | 80% | 16.3% |
| B0 | NT$100M | −62.6% | −2.72 | 463 | 0.2% | 0.3% | 73% | 14.7% |
| B0 | NT$300M | −60.4% | −3.03 | 922 | 0% | 0.3% | 56% | 11.3% |
| B0 | NT$1B | −45.7% | −3.16 | 1,552 | 0% | 0.3% | 33% | 6.7% |
| D0 | fractional | −60.5% | −2.28 | — | 0% | 0% | 100% | 20.3% |
| D0 | NT$10M | −46.3% | −2.19 | 72 | 8.9% | 0.2% | 76% | 15.5% |
| D0 | NT$100M | −57.4% | −2.60 | 475 | 0.1% | 0.3% | 71% | 14.3% |
| D0 | NT$1B | −39.8% | −3.02 | 1,506 | 0% | 0.3% | 30% | 6.1% |

Larger capital deploys less (participation cap), so the absolute loss shrinks while the Sharpe worsens; no scenario is positive.

### 4. Implications

- Of the four strategy routes this is the highest gross and the cleanest test: tradable futures prices, no borrow, no overnight risk, and a signal present in both P1 and P2. Its only failure is the stock-futures bid-ask spread: the tick is wide relative to price, and one round trip a day costs 0.5%.
- Making it work needs execution, not a better model: passive limit fills between 08:45 and 09:00, restricting to the price bands with the smallest relative tick (e.g. NT$500–1,000 with a NT$1 tick, 0.1–0.2%), or stock day trades with borrow (0.52% round trip, equally insufficient). These are execution assumptions that daily data cannot verify; they need intraday data or live orders.
- The long leg's lack of value is confirmed again; the hedged version's TX long only adds market beta.
- Under the protocol the model layer returns to the signal itself: only if §13.16's residualized labels recover the multi-session horizons can low-turnover long-only or long-short books be re-established.

### 5. Limits

- Contract codes are mapped through the 2026-10 list, so stock futures delisted during the period are excluded (survivorship bias).
- The 08:45 open is tradable but thin; one tick of slippage is a conservative market-order assumption, and passive fills are not modelled.
- The settlement price is the official mark; actual covers trade around 13:30.
- Margin uses the highest tier for every name; intraday calls are not modelled.
- q is chosen on P1 and the P1 ∪ P2 decision includes P1.

### 6. Reproduction

```bash
cd backtest
uv run python -m twbacktest.taifex download --config config/capturability.json   # every contract; a separate root (taifex-all) was used for 2022-12 to 2026-06
uv run python -m twbacktest.futures_daytrade --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Item | Content |
| --- | --- |
| Pre-registration commit | `216fd7a` (§13.15) |
| Code commit | `ddfeb21` (`twbacktest.futures_daytrade`, `taifex.stock_futures_sessions`, configuration) |
| Configuration | `backtest/config/futures_daytrade.json` |
| Futures data (not tracked by Git) | `artifacts/diagnostics/taifex-all/` (188 weekly files plus the underlying list), `artifacts/diagnostics/taifex-tx/` |
| Output (not tracked by Git) | `artifacts/diagnostics/backtest/futures-daytrade-b0-d0-20261023T090019Z/` (`report.json`, `grid.csv`, `daily.csv`, `summary.md`) |
