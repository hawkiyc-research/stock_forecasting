# 台股策略層結果：band 組合與台指期疊加 / Taiwan Strategy-Layer Results: Band Portfolio and TAIEX Futures Overlay

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.14](experiment_protocol.md) 的判定：把 B0 `t-lgbm-alpha` 與 D0 `t-xs-scratch` 的 walk-forward 排序分數，用業界的成本緩解做法（流動性宇宙、EMA 平滑、買進／持有門檻、每日換入上限）建成多頭組合，並以官方期交所資料的台指期（TX）近月連續序列做對沖版、現金替代版與 TX 多頭參考。參數在 TWSE P1 選定，判定在 P1 ∪ P2。

結論：

1. **成本問題解決了，訊號問題沒有。** 兩個模型選出的都是 h = 10、K = 50、m = 3：每日換手由 Top-k dropout 的 10.2% 降到 1.9–2.5%，成本拖累由 17%／年降到 3.5–4.5%／年，平均持有 39–52 個交易日。P1 的淨超額轉正：B0 +5.4%、D0 +5.5%／年（相對流動性宇宙等權），IR 0.42／0.52。
2. **H-D 未通過（兩個模型）。** P1 ∪ P2 的年化淨超額 B0 −4.7% [−19.7%, +10.5%]、D0 −5.4% [−18.2%, +7.7%]，區間下界皆小於 0。原因在 P2：毛超額 B0 −27%、D0 −31%／年，和 §13.13 的發現一致（P2 的 h = 5–14 Rank IC 掉到 0.02–0.05）。
3. **絕對報酬是正的，輸的是基準。** 多頭版 P1 ∪ P2 年化淨報酬 B0 +23.8%（CAGR 25.7%，Sharpe 1.72，最大回撤 −13.0%）、D0 +23.1%（CAGR 24.5%，Sharpe 1.50，最大回撤 −20.0%）。同期流動性宇宙等權約 +28%／年，TAIEX 報酬指數（含息）約 +41%／年；P2 單獨看，組合 +23–26% 對上 TAIEX +115%，落後 63 個百分點。
4. **H-D-hedged 未通過，而且是大幅虧損。** 放空 TX 對沖後 P1 ∪ P2 年化淨報酬 B0 −12.5% [−28.7%, +3.0%]、D0 −13.3%，最大回撤 −41%，28／27 次追繳。對沖的是加權指數，而組合是流動性宇宙內接近等權的 50 檔；P2 大型股獨漲時，這個對沖等於放空最強的那一端。
5. **現金替代版與多頭版幾乎相同**（閒置現金只有 1–2%），不改變結論。
6. **整張可行性在 1,000 萬元資本下是實質限制。** 位置金額 20 萬元買不到一張的候選股占 34–37%（高價股被跳過），年化淨報酬降 1.5–2.7 個百分點；3,000 萬元時被擋比例 5–6%，差異在 1 個百分點內。
7. **TX 多頭參考**：對 TAIEX 報酬指數年化 −2.4%（P1 −3.4%、P2 +0.3%），即期貨相對含息指數的隱含資金成本；P1 ∪ P2 共 38 次換月。這是用台指期取得市場曝險的真實代價，對沖版則反向收到這一項。
8. **依 §13.14 第 9 點的處置**：多頭現股持有在台股成本下無法把這兩個模型的排序能力變成相對基準的報酬；下一個候選是股票期貨標的內的多空（§13.9 的可取性）或 h = 1 的當沖路線，都要另行登記。前瞻候選模型：無。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 預測 | B0 `launch-20261022T234922Z-22048038`、D0 `launch-20261023T015746Z-224647797`，各為三 seed 集成的 walk-forward 預測 |
| 價格 | 資料集 raw `market.parquet`：未調整開收盤價與成交量（流動性、整張）、調整後開收盤價（報酬）；指數的 adjusted_close 為官方報酬指數（含息） |
| 台指期 | 期交所每日行情下載（`twbacktest.taifex download-contract --commodity TX`，2022-12 至 2026-06，47 個檔案、862 個交易日）；近月在到期月第三個星期三開盤換月（41 次）；每日以結算價 mark（結算價取自 13:30 前的成交，與現股收盤對齊；開盤價對開盤價的 TX 與指數日差標準差 0.84%，結算價對收盤價只有 0.31%） |
| 宇宙 | 只含股票；20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬，收盤價 ≥ 10 元 |
| 訊號 | 單一 horizon 分數 → 市場內百分位名次 → 每檔 EMA（半衰期 3 日，中斷逾 5 日重設） |
| 組合 | 進前 K 名買、掉出前 m × K 名賣；每日最多新買 ⌈K／10⌉ 檔；K 個等權位置，逐部位開盤到開盤記帳；漲停開盤不買、跌停開盤不賣 |
| 格子與選擇 | h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3}，以 TWSE P1 多頭版淨超額 IR 最高者固定 |
| 成本 | 現股主情境：手續費 0.1425% × 0.6、證交稅 0.3%、滑價 TWSE 0.1%／TPEX 0.2%；TX 每邊：期交稅 0.002%、手續費每口 50 元、滑價 1 點；換月兩邊 |
| 對沖版 | 放空名目 = 1 × 持股市值，每日開盤調整；準備金 15% 名目，低於 10% 時賣股補足並計一次追繳，高於 25% 時把超額移回股票 |
| 基準 | 多頭版與現金替代版對同市場流動性宇宙等權（開盤到開盤）；另報對 TAIEX 報酬指數；對沖版對 0 |
| 區間 | 月區塊（21 日）circular block bootstrap，1,000 次、seed 11 |
| 執行 | 本機 `backtest/`，4 個 spawn worker，`BACKTEST_WORKERS=4` |

### 3. 結果

#### 3.1 P1 選擇格子（TWSE，多頭版，主要成本，分數位置）

| 模型 | h | K | m | 年化毛超額 | 年化淨超額 | 淨超額 IR | 年化淨報酬 | 每日換手 | 成本拖累／年 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 5 | 30 | 2 | +6.8% | +0.5% | 0.04 | +18.1% | 3.7% | 6.3% |
| B0 | 5 | 30 | 3 | +4.7% | +0.7% | 0.05 | +18.3% | 2.4% | 3.9% |
| B0 | 5 | 50 | 2 | +6.8% | +1.7% | 0.13 | +19.2% | 3.1% | 5.2% |
| B0 | 5 | 50 | 3 | +6.7% | +3.5% | 0.27 | +21.1% | 2.0% | 3.3% |
| B0 | 10 | 30 | 2 | +7.5% | +1.6% | 0.12 | +19.2% | 3.5% | 5.9% |
| B0 | 10 | 30 | 3 | +6.6% | +2.6% | 0.18 | +20.2% | 2.5% | 4.0% |
| B0 | 10 | 50 | 2 | +8.7% | +3.6% | 0.29 | +21.2% | 3.1% | 5.1% |
| **B0** | **10** | **50** | **3** | **+8.8%** | **+5.4%** | **0.42** | **+22.9%** | **2.0%** | **3.4%** |
| D0 | 5 | 30 | 2 | +10.7% | +0.8% | 0.07 | +18.4% | 5.7% | 9.8% |
| D0 | 5 | 30 | 3 | +9.6% | +2.8% | 0.23 | +20.4% | 4.1% | 6.8% |
| D0 | 5 | 50 | 2 | +8.4% | +0.9% | 0.08 | +18.5% | 4.5% | 7.5% |
| D0 | 5 | 50 | 3 | +7.7% | +3.0% | 0.26 | +20.6% | 2.8% | 4.7% |
| D0 | 10 | 30 | 2 | +10.9% | +1.3% | 0.11 | +18.9% | 5.7% | 9.5% |
| D0 | 10 | 30 | 3 | +12.0% | +5.1% | 0.44 | +22.7% | 4.1% | 6.9% |
| D0 | 10 | 50 | 2 | +9.2% | +1.8% | 0.17 | +19.4% | 4.4% | 7.5% |
| **D0** | **10** | **50** | **3** | **+10.3%** | **+5.5%** | **0.52** | **+23.1%** | **2.8%** | **4.8%** |

格子內的規律一致：門檻越寬（m = 3）、位置越多（K = 50）換手越低、淨超額越高；D0 的毛超額高於 B0 但換手也高，淨值相近。

#### 3.2 選定參數的各版本（TWSE，h = 10、K = 50、m = 3，主要成本，分數位置）

| 模型 | 版本 | 期間 | 年化淨報酬 | CAGR | Sharpe | 最大回撤 | 年化毛超額 | 年化淨超額 | 95% 區間 | IR | 對報酬指數超額 | 追繳 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 多頭 | P1 | +22.9% | 24.4% | 1.58 | −13.0% | +8.8% | +5.4% | [−9.1%, +20.7%] | 0.42 | +1.1% | — |
| B0 | 多頭 | P2 | +26.1% | 28.9% | 2.21 | −5.9% | −27.4% | −31.0% | [−56.1%, −5.9%] | −2.13 | −63.2% | — |
| B0 | 多頭 | P1 ∪ P2 | +23.8% | 25.7% | 1.72 | −13.0% | −1.3% | −4.7% | [−19.7%, +10.5%] | −0.36 | −16.8% | — |
| B0 | 對沖 | P1 | +3.3% | 1.9% | 0.19 | −12.3% | +6.4% | +3.3% | [−9.2%, +17.1%] | 0.19 | — | 28 |
| B0 | 對沖 | P2 | −53.8% | −42.6% | −2.97 | −40.9% | −50.1% | −53.8% | [−82.6%, −30.7%] | −2.97 | — | 28 |
| B0 | 對沖 | P1 ∪ P2 | −12.5% | −13.1% | −0.72 | −40.9% | −9.3% | −12.5% | [−28.7%, +3.0%] | −0.72 | — | 28 |
| B0 | 現金替代 | P1 ∪ P2 | +27.3% | 30.0% | 1.92 | −12.7% | +2.2% | −1.2% | [−16.5%, +15.1%] | −0.09 | −13.3% | — |
| D0 | 多頭 | P1 | +23.1% | 24.3% | 1.39 | −20.0% | +10.3% | +5.5% | [−6.2%, +17.4%] | 0.52 | +1.2% | — |
| D0 | 多頭 | P2 | +23.0% | 25.0% | 1.96 | −5.8% | −30.5% | −34.0% | [−56.3%, −11.6%] | −2.32 | −66.3% | — |
| D0 | 多頭 | P1 ∪ P2 | +23.1% | 24.5% | 1.50 | −20.0% | −1.0% | −5.4% | [−18.2%, +7.7%] | −0.46 | −17.5% | — |
| D0 | 對沖 | P1 ∪ P2 | −13.3% | −13.9% | −0.74 | −41.9% | −9.0% | −13.3% | [−28.7%, +2.0%] | −0.74 | — | 27 |
| D0 | 現金替代 | P1 ∪ P2 | +25.8% | 27.9% | 1.65 | −20.3% | +1.8% | −2.7% | [−16.1%, +11.6%] | −0.22 | −14.7% | — |

判定格（多頭、P1 ∪ P2）的區間下界 B0 −19.7%、D0 −18.2%，H-D 未通過；對沖版 P1 ∪ P2 的下界 −28.7%，H-D-hedged 未通過。

#### 3.3 整張可行性（TWSE，多頭，P1 ∪ P2）

| 模型 | 資本 | 年化淨報酬 | 年化淨超額 | 被擋的候選買單比例 | 平均現金比例 |
| --- | --- | --- | --- | --- | --- |
| B0 | 分數位置 | +23.8% | −4.7% | 0% | 1.5% |
| B0 | 1,000 萬 | +21.1% | −7.4% | 33.7% | 1.4% |
| B0 | 3,000 萬 | +24.9% | −3.7% | 5.7% | 1.6% |
| D0 | 分數位置 | +23.1% | −5.4% | 0% | 1.1% |
| D0 | 1,000 萬 | +21.6% | −6.9% | 37.1% | 1.1% |
| D0 | 3,000 萬 | +22.5% | −6.0% | 4.6% | 1.2% |

#### 3.4 TX 多頭參考（對 TAIEX 報酬指數，結算價對收盤價）

| 期間 | TX 年化淨報酬 | 對報酬指數年化超額 | Sharpe | 最大回撤 | 換月次數 |
| --- | --- | --- | --- | --- | --- |
| P1 | +19.3% | −3.4% | 0.88 | −30.5% | 28 |
| P2 | +86.7% | +0.3% | 3.90 | −10.5% | 10 |
| P1 ∪ P2 | +37.9% | −2.4% | 1.71 | −30.5% | 38 |

#### 3.5 TPEX（只作描述，同參數）

| 模型 | 期間 | 年化毛超額 | 年化淨報酬 | 年化淨超額 | 95% 區間 | Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | P1 | +10.7% | +22.5% | +4.7% | [−6.4%, +15.6%] | 1.24 |
| B0 | P2 | −32.7% | +27.2% | −37.2% | [−64.8%, −12.4%] | 1.67 |
| B0 | P1 ∪ P2 | −1.3% | +23.8% | −6.9% | [−19.8%, +5.7%] | 1.35 |
| D0 | P1 | +8.1% | +19.9% | +2.1% | [−7.9%, +12.0%] | 1.00 |
| D0 | P2 | −21.7% | +37.7% | −26.6% | [−48.6%, −4.3%] | 2.26 |
| D0 | P1 ∪ P2 | −0.1% | +24.8% | −5.9% | [−16.6%, +4.8%] | 1.30 |

### 4. 意涵

- 策略層做對了它能做的事：換手與成本降到業界水準（每日 2%、3–5%／年），P1 的淨超額轉正。但區間仍跨 0，而且 P2 的毛超額是負的 27–31%：2025-06 之後 h = 10 的排序對 TWSE 流動性宇宙幾乎沒有多日預測力，任何多頭持有規則都救不回來。
- 兩個模型對基準的虧損集中在 P2 的大型股獨漲：組合落後 TAIEX 報酬指數 63–66 個百分點，落後等權宇宙 31–34 個百分點。這是訊號在特定市場狀態下的失效，模型層（D0 對 B0 的 0.004 Rank IC 差距）無法補。
- 用台指期對沖等權型多頭組合，在集中式多頭市場是結構性虧損，28 次追繳說明保證金準備要遠大於 15% 才撐得住；對沖的對象應是組合自己的 beta 與風格曝險，不是加權指數。
- 1,000 萬元資本下 K = 50 的位置太小，三分之一的候選股買不到一張；若實際資本在這個量級，K 要降或改用零股，兩者都要重新登記。
- 依 §13.14 第 9 點，下一步不是再調多頭規則，而是另行登記：（a）股票期貨標的內的多空（期交稅十萬分之二，可做空），（b）h = 1 的當沖路線（P2 的 h = 1 Rank IC 仍有 0.11，但每批毛超額 0.26% 對當沖來回 0.52%，現股不可行，期貨可再算）。

### 5. 限制

- 8 組參數在 P1 選一組，P1 的 +5.4%／+5.5% 帶有選擇偏誤，區間也跨 0。
- 流動性門檻與 β = 1 是事前固定值；對沖的保證金模型是簡化版（固定比例準備金與追繳規則），未用期交所逐日公告的保證金表。
- TX 以結算價 mark、現股以開盤價記帳，兩者相差一個交易時段；對沖版的日報酬因此含約 0.3% 的時點噪音（開盤價對開盤價是 0.84%，所以選結算價）。
- 等權基準是流動性宇宙的無成本每日再平衡，比可投資的等權 ETF 更難打；對 TAIEX 報酬指數的超額另列。
- 未模擬成交量參與率、股利稅與借券；整張可行性只在兩個資本水準下報告。

### 6. 重現方式

```bash
cd backtest
uv run python -m twbacktest.taifex download-contract --commodity TX --start 2022-12-01 --end-exclusive 2026-07-01 --root ../artifacts/diagnostics/taifex-tx
BACKTEST_WORKERS=4 uv run python -m twbacktest.tw_strategy --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `970052a`（§13.14） |
| 程式 commit | `9b95f6a`（期交所單一契約下載與連續近月）、`9e123f1`（策略層模組與設定檔） |
| 設定檔 | `backtest/config/tw_strategy.json` |
| 台指期資料（不受 Git 追蹤） | `artifacts/diagnostics/taifex-tx/`（`download-TX.json` 與 47 個 raw CSV） |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/tw-strategy-b0-d0-20261023T074716Z/`（`report.json`、`grid.csv`、`daily_series.csv`、`summary.md`） |

## English

### 1. Purpose and conclusions

This report records the decisions of [experiment protocol §13.14](experiment_protocol.md): the walk-forward ranking scores of B0 `t-lgbm-alpha` and D0 `t-xs-scratch` are turned into long-only portfolios with the industry's cost-mitigation practice (liquidity universe, EMA smoothing, buy/hold band, daily entry cap), with a TAIEX futures (TX) front-month series from the official TAIFEX data providing a hedged version, a cash-equitized version and a TX-long reference. Parameters are selected on TWSE P1 and judged on P1 ∪ P2.

Conclusions:

1. **The cost problem is solved; the signal problem is not.** Both models select h = 10, K = 50, m = 3: daily turnover falls from 10.2% (top-k dropout) to 1.9–2.5%, the cost drag from 17% to 3.5–4.5% per year, with a mean holding of 39–52 sessions. The P1 net excess turns positive: B0 +5.4%, D0 +5.5% per year over the equal-weight liquid universe, IR 0.42 / 0.52.
2. **H-D fails for both models.** The P1 ∪ P2 annualized net excess is B0 −4.7% [−19.7%, +10.5%] and D0 −5.4% [−18.2%, +7.7%]; both lower bounds are below 0. The cause is P2: a gross excess of −27% (B0) and −31% (D0) per year, consistent with §13.13 (the P2 Rank IC for h = 5–14 falls to 0.02–0.05).
3. **Absolute returns are positive; the benchmarks win.** Long-only P1 ∪ P2 annualized net return: B0 +23.8% (CAGR 25.7%, Sharpe 1.72, max drawdown −13.0%), D0 +23.1% (CAGR 24.5%, Sharpe 1.50, max drawdown −20.0%). The equal-weight liquid universe made about +28% per year and the TAIEX total-return index about +41%; in P2 alone the portfolios' +23–26% trail TAIEX's +115% by 63 points.
4. **H-D-hedged fails, with a large loss.** Shorting TX against the book gives a P1 ∪ P2 annualized net return of −12.5% [−28.7%, +3.0%] (B0) and −13.3% (D0), a −41% drawdown and 28 / 27 margin calls. The hedge is the cap-weighted index while the book is 50 near-equal-weight liquid names; in P2's mega-cap rally the hedge shorts the strongest end of the market.
5. **The cash-equitized version is almost identical to long-only** (idle cash is 1–2%) and changes nothing.
6. **Board-lot feasibility binds at NT$10 million.** With NT$200k per slot, 34–37% of candidate buys cannot afford one lot (high-priced names are skipped) and the annualized net return falls by 1.5–2.7 points; at NT$30 million the blocked share is 5–6% and the difference is within 1 point.
7. **TX-long reference**: −2.4% per year against the TAIEX total-return index (P1 −3.4%, P2 +0.3%), i.e. the implied financing of futures relative to a dividend-inclusive index; 38 rolls over P1 ∪ P2. This is the true cost of taking market exposure through TX; the hedged version earns it with the opposite sign.
8. **Disposition under §13.14 item 9**: long-only stock holding cannot turn these two models' ranking skill into returns against the benchmark under Taiwan costs; the next candidates are a long-short inside the stock-futures universe (§13.9 capturability) or an h = 1 day-trade route, each needing its own registration. Forward candidate: none.

### 2. Data and method

| Item | Content |
| --- | --- |
| Predictions | B0 `launch-20261022T234922Z-22048038`, D0 `launch-20261023T015746Z-224647797`, three-seed ensembles |
| Prices | The dataset's raw `market.parquet`: unadjusted open/close and volume (liquidity, lots), adjusted open/close (returns); the indices' adjusted_close is the official total-return index |
| TAIEX futures | TAIFEX daily download (`twbacktest.taifex download-contract --commodity TX`, 2022-12 to 2026-06, 47 files, 862 sessions); the front month rolls at the open of the expiry month's third Wednesday (41 rolls); marked daily at the settlement price (computed from trades just before 13:30, aligned with the stock close; TX-vs-index daily differences have a 0.84% standard deviation open-to-open but 0.31% settlement-to-close) |
| Universe | Stocks only; 20-session median dollar volume ≥ NT$30 million (TWSE) / NT$10 million (TPEX); close ≥ NT$10 |
| Signal | One horizon's score → within-market percentile rank → per-symbol EMA (3-session half-life, reset after a gap over 5 sessions) |
| Portfolio | Buy into the top K, sell when out of the top m × K; at most ⌈K/10⌉ new buys per session; K equal slots with per-position open-to-open accounting; no buys at limit-up opens, no sells at limit-down opens |
| Grid and selection | h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3}, fixed at the highest TWSE P1 long-only net excess IR |
| Costs | Stocks, primary scenario: commission 0.1425% × 0.6, transaction tax 0.3%, slippage TWSE 0.1% / TPEX 0.2%; TX per side: 0.002% tax, NT$50 commission per contract, 1 tick slippage; both sides on a roll |
| Hedged | Short notional = 1 × stock value, adjusted at each open; 15% reserve, refilled from stocks below 10% (one margin call), excess above 25% returned to stocks |
| Benchmarks | Long-only and cash-equitized against the same market's equal-weight liquid universe (open to open); excess over the TAIEX total-return index also reported; hedged against 0 |
| Intervals | Monthly-block (21 sessions) circular block bootstrap, 1,000 replicates, seed 11 |
| Execution | Local `backtest/`, 4 spawn workers (`BACKTEST_WORKERS=4`) |

### 3. Results

#### 3.1 P1 selection grid (TWSE, long-only, primary costs, fractional slots)

| Model | h | K | m | Gross excess (annual) | Net excess (annual) | Net excess IR | Net return (annual) | Daily turnover | Cost drag / year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 5 | 30 | 2 | +6.8% | +0.5% | 0.04 | +18.1% | 3.7% | 6.3% |
| B0 | 5 | 30 | 3 | +4.7% | +0.7% | 0.05 | +18.3% | 2.4% | 3.9% |
| B0 | 5 | 50 | 2 | +6.8% | +1.7% | 0.13 | +19.2% | 3.1% | 5.2% |
| B0 | 5 | 50 | 3 | +6.7% | +3.5% | 0.27 | +21.1% | 2.0% | 3.3% |
| B0 | 10 | 30 | 2 | +7.5% | +1.6% | 0.12 | +19.2% | 3.5% | 5.9% |
| B0 | 10 | 30 | 3 | +6.6% | +2.6% | 0.18 | +20.2% | 2.5% | 4.0% |
| B0 | 10 | 50 | 2 | +8.7% | +3.6% | 0.29 | +21.2% | 3.1% | 5.1% |
| **B0** | **10** | **50** | **3** | **+8.8%** | **+5.4%** | **0.42** | **+22.9%** | **2.0%** | **3.4%** |
| D0 | 5 | 30 | 2 | +10.7% | +0.8% | 0.07 | +18.4% | 5.7% | 9.8% |
| D0 | 5 | 30 | 3 | +9.6% | +2.8% | 0.23 | +20.4% | 4.1% | 6.8% |
| D0 | 5 | 50 | 2 | +8.4% | +0.9% | 0.08 | +18.5% | 4.5% | 7.5% |
| D0 | 5 | 50 | 3 | +7.7% | +3.0% | 0.26 | +20.6% | 2.8% | 4.7% |
| D0 | 10 | 30 | 2 | +10.9% | +1.3% | 0.11 | +18.9% | 5.7% | 9.5% |
| D0 | 10 | 30 | 3 | +12.0% | +5.1% | 0.44 | +22.7% | 4.1% | 6.9% |
| D0 | 10 | 50 | 2 | +9.2% | +1.8% | 0.17 | +19.4% | 4.4% | 7.5% |
| **D0** | **10** | **50** | **3** | **+10.3%** | **+5.5%** | **0.52** | **+23.1%** | **2.8%** | **4.8%** |

The pattern is uniform: a wider band (m = 3) and more slots (K = 50) lower turnover and raise the net excess; D0 has a higher gross excess than B0 but also higher turnover, so the net values are close.

#### 3.2 Selected parameters by version (TWSE, h = 10, K = 50, m = 3, primary costs, fractional slots)

| Model | Version | Period | Net return (annual) | CAGR | Sharpe | Max drawdown | Gross excess (annual) | Net excess (annual) | 95% CI | IR | Excess vs TR index | Margin calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | long-only | P1 | +22.9% | 24.4% | 1.58 | −13.0% | +8.8% | +5.4% | [−9.1%, +20.7%] | 0.42 | +1.1% | — |
| B0 | long-only | P2 | +26.1% | 28.9% | 2.21 | −5.9% | −27.4% | −31.0% | [−56.1%, −5.9%] | −2.13 | −63.2% | — |
| B0 | long-only | P1 ∪ P2 | +23.8% | 25.7% | 1.72 | −13.0% | −1.3% | −4.7% | [−19.7%, +10.5%] | −0.36 | −16.8% | — |
| B0 | hedged | P1 | +3.3% | 1.9% | 0.19 | −12.3% | +6.4% | +3.3% | [−9.2%, +17.1%] | 0.19 | — | 28 |
| B0 | hedged | P2 | −53.8% | −42.6% | −2.97 | −40.9% | −50.1% | −53.8% | [−82.6%, −30.7%] | −2.97 | — | 28 |
| B0 | hedged | P1 ∪ P2 | −12.5% | −13.1% | −0.72 | −40.9% | −9.3% | −12.5% | [−28.7%, +3.0%] | −0.72 | — | 28 |
| B0 | cash-equitized | P1 ∪ P2 | +27.3% | 30.0% | 1.92 | −12.7% | +2.2% | −1.2% | [−16.5%, +15.1%] | −0.09 | −13.3% | — |
| D0 | long-only | P1 | +23.1% | 24.3% | 1.39 | −20.0% | +10.3% | +5.5% | [−6.2%, +17.4%] | 0.52 | +1.2% | — |
| D0 | long-only | P2 | +23.0% | 25.0% | 1.96 | −5.8% | −30.5% | −34.0% | [−56.3%, −11.6%] | −2.32 | −66.3% | — |
| D0 | long-only | P1 ∪ P2 | +23.1% | 24.5% | 1.50 | −20.0% | −1.0% | −5.4% | [−18.2%, +7.7%] | −0.46 | −17.5% | — |
| D0 | hedged | P1 ∪ P2 | −13.3% | −13.9% | −0.74 | −41.9% | −9.0% | −13.3% | [−28.7%, +2.0%] | −0.74 | — | 27 |
| D0 | cash-equitized | P1 ∪ P2 | +25.8% | 27.9% | 1.65 | −20.3% | +1.8% | −2.7% | [−16.1%, +11.6%] | −0.22 | −14.7% | — |

The decision cell (long-only, P1 ∪ P2) has lower bounds of −19.7% (B0) and −18.2% (D0), so H-D fails; the hedged P1 ∪ P2 lower bound is −28.7%, so H-D-hedged fails.

#### 3.3 Board-lot feasibility (TWSE, long-only, P1 ∪ P2)

| Model | Capital | Net return (annual) | Net excess (annual) | Blocked candidate buys | Mean cash share |
| --- | --- | --- | --- | --- | --- |
| B0 | fractional | +23.8% | −4.7% | 0% | 1.5% |
| B0 | NT$10 million | +21.1% | −7.4% | 33.7% | 1.4% |
| B0 | NT$30 million | +24.9% | −3.7% | 5.7% | 1.6% |
| D0 | fractional | +23.1% | −5.4% | 0% | 1.1% |
| D0 | NT$10 million | +21.6% | −6.9% | 37.1% | 1.1% |
| D0 | NT$30 million | +22.5% | −6.0% | 4.6% | 1.2% |

#### 3.4 TX-long reference (against the TAIEX total-return index, settlement to close)

| Period | TX net return (annual) | Excess vs TR index (annual) | Sharpe | Max drawdown | Rolls |
| --- | --- | --- | --- | --- | --- |
| P1 | +19.3% | −3.4% | 0.88 | −30.5% | 28 |
| P2 | +86.7% | +0.3% | 3.90 | −10.5% | 10 |
| P1 ∪ P2 | +37.9% | −2.4% | 1.71 | −30.5% | 38 |

#### 3.5 TPEX (descriptive, same parameters)

| Model | Period | Gross excess (annual) | Net return (annual) | Net excess (annual) | 95% CI | Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | P1 | +10.7% | +22.5% | +4.7% | [−6.4%, +15.6%] | 1.24 |
| B0 | P2 | −32.7% | +27.2% | −37.2% | [−64.8%, −12.4%] | 1.67 |
| B0 | P1 ∪ P2 | −1.3% | +23.8% | −6.9% | [−19.8%, +5.7%] | 1.35 |
| D0 | P1 | +8.1% | +19.9% | +2.1% | [−7.9%, +12.0%] | 1.00 |
| D0 | P2 | −21.7% | +37.7% | −26.6% | [−48.6%, −4.3%] | 2.26 |
| D0 | P1 ∪ P2 | −0.1% | +24.8% | −5.9% | [−16.6%, +4.8%] | 1.30 |

### 4. Implications

- The strategy layer did what it can: turnover and costs are down to industry levels (2% per day, 3–5% per year) and the P1 net excess is positive. The interval still spans 0, and the P2 gross excess is −27% to −31%: after 2025-06 the h = 10 ranking has almost no multi-session predictive power in the TWSE liquid universe, and no long-only holding rule recovers that.
- Both models' losses against the benchmarks concentrate in P2's mega-cap rally: the portfolios trail the TAIEX total-return index by 63–66 points and the equal-weight universe by 31–34 points. This is a signal failure in a particular market state, which the model layer (D0's 0.004 Rank IC edge over B0) cannot offset.
- Hedging a near-equal-weight long book with TAIEX futures is a structural loss in a concentrated bull market, and 28 margin calls show a 15% reserve is far too small; the hedge should target the book's own beta and style exposures, not the cap-weighted index.
- At NT$10 million, K = 50 slots are too small: a third of the candidates cannot afford one lot. Capital at that level needs a smaller K or odd lots, each requiring re-registration.
- Under §13.14 item 9 the next step is not another long-only rule but a separate registration: (a) a long-short inside the stock-futures universe (0.002% tax, shortable), (b) an h = 1 day-trade route (the P2 h = 1 Rank IC is still 0.11, but 0.26% gross per batch against a 0.52% day-trade round trip rules out stocks; futures can be costed).

### 5. Limits

- One of 8 parameter sets is chosen on P1, so the P1 +5.4% / +5.5% carries selection bias and its interval spans 0.
- The liquidity thresholds and β = 1 are fixed in advance; the margin model is simplified (fixed reserve ratio and call rule) and does not use TAIFEX's daily margin tables.
- TX is marked at the settlement and stocks at the open, one session apart; the hedged daily returns therefore carry about 0.3% of timing noise (0.84% open-to-open, hence the settlement choice).
- The equal-weight benchmark is a cost-free daily-rebalanced liquid universe, a harder bar than an investable equal-weight ETF; the excess over the TAIEX total-return index is listed separately.
- Volume participation, dividend taxes and securities lending are not modelled; lot feasibility is reported at two capital levels only.

### 6. Reproduction

```bash
cd backtest
uv run python -m twbacktest.taifex download-contract --commodity TX --start 2022-12-01 --end-exclusive 2026-07-01 --root ../artifacts/diagnostics/taifex-tx
BACKTEST_WORKERS=4 uv run python -m twbacktest.tw_strategy --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Item | Content |
| --- | --- |
| Pre-registration commit | `970052a` (§13.14) |
| Code commits | `9b95f6a` (TAIFEX single-contract download and continuous front month), `9e123f1` (strategy-layer module and configuration) |
| Configuration | `backtest/config/tw_strategy.json` |
| TAIEX futures data (not tracked by Git) | `artifacts/diagnostics/taifex-tx/` (`download-TX.json` and 47 raw CSV files) |
| Output (not tracked by Git) | `artifacts/diagnostics/backtest/tw-strategy-b0-d0-20261023T074716Z/` (`report.json`, `grid.csv`, `daily_series.csv`, `summary.md`) |
