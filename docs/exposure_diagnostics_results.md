# §13.17 步驟 0：殘差 alpha 幅度與成本診斷結果 / Step 0 Results: Residual Alpha Magnitude versus Costs

## 中文

### 1. 目的與結論

[實驗協定 §13.17](experiment_protocol.md) 登記的暴露中性策略（0050 核心加股票期貨多空疊加，門檻為勝過 0050 買進持有）在建策略前先做一個描述性診斷：殘差化標籤下 B0 與 D0 的多日排序能力，換成可交易的毛價差之後有多大，相對於台灣股票期貨的執行成本又是多少。診斷有事先寫好的停止規則。本報告記錄其結果與處置。

結論：

1. **停止規則對兩個模型都觸發，不建策略。** 在股票期貨流動宇宙、h = 10、P1 ∪ P2，產業中性且金額中性的前後四分位多空毛年化 B0 −0.6% [−7.0%, +5.4%]、D0 +0.4% [−5.6%, +6.2%]；同一組合的年化成本拖累 18.3%，門檻（兩倍）36.7%。h = 5 的毛年化 +5.7%／+5.2%，但成本拖累 30.8%。無中性化與規模中性的版本結論相同。
2. **殘差 alpha 的幅度太小。** 即使在全部流動性宇宙、P1 這段最好的期間，前後四分位的殘差價差也只有每 10 日 0.88%（B0）／0.96%（D0），產業中性多空毛年化 +8.7% [+4.5%, +13.3%]／+10.1% [+6.3%, +13.9%]；P1 ∪ P2 降到 +5.0%／+7.6%，P2 為 −4.5%／+0.9%。Rank IC 0.08–0.10 對應的可交易價差就是這個量級。
3. **alpha 不在可放空的宇宙裡。** 約 123–144 檔有流動股票期貨的大型股中，殘差 Rank IC 在 P2 掉到 0（B0 −0.007、D0 +0.006，區間跨 0），P1 ∪ P2 只有 0.045／0.039；殘差四分位價差 P2 為負（−1.17%／−0.69% 每 10 日）。§13.16 看到的 P2 多日排序能力全部來自沒有股票期貨、只能做多且成本更高的中小型股。
4. **S1 與 S2 都不可能勝過 0050。** S1（多空疊加）的 alpha 腿在可放空宇宙內是零；S2（多頭衛星）能用的上限是流動性宇宙多頭這一側、毛年化約 3–5%，低於現股持有 10 日的成本拖累（證交稅 0.3% 加手續費與滑價，每日換 1/10 部位約 12%／年）。兩者都還要承擔現金準備在上漲年份的確定拖累。
5. **處置**：依 §13.17 第 0 項，不實作 S1／S2／S3、H-G 與 `exposure_neutral.json`。這條模型線的結論是：有 h = 1 排序能力（§13.15 證明執行成本下不可交易）與殘差多日排序能力（本報告證明幅度太小、且不在可放空宇宙內），在台灣現有工具與成本下沒有勝過 0050 買進持有的可部署策略。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 訊號 | B0 `t-lgbm-alpha-residual_v1a` 與 D0 `t-xs-scratch`（三 seed 集成）在 `residual_v1a` 標籤上的 walk-forward 分數，h ∈ {5, 10}；殘差目標取自同一 run 的 `targets.parquet` |
| 價格 | 資料集 raw `market.parquet`；報酬為次一開盤買進、h 日後收盤賣出（標籤定義的持有），未扣成本；以現股價格代表股票期貨（多日持有下基差噪音小） |
| 宇宙 | liquidity：§13.14 的市場內流動性宇宙（20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬，收盤 ≥ 10 元，排除 ETF），市場內排名；futures_liquid：§13.15 的股票期貨流動宇宙（近月 20 日平均成交量 ≥ 100 口、訊號日一般時段有成交），兩市場合併排名。期交所全契約日資料與標的清單同 §13.15（862 個交易日、314 個對應契約、每日約 243 檔有股票期貨、38 個已下市代碼無法對應） |
| 組合 | 前後四分位、金額中性；plain = 直接排名；industry = 依參考表兩碼產業在產業內排名、產業權重等於檔數比例（不足 4 檔者併入其他），兩邊產業權重相同；size = 當日成交金額中位數分上下兩半、各半內排名。多空毛年化以總名目 = 資本（每邊一半）、每批報酬 × 252／h 計 |
| 成本 | 股票期貨每邊：期交稅 0.002%、手續費每口 30 元、1 tick 滑價（tick 級距同 §13.15，口數依標的清單的標準契約股數）；每日換 1/h 的兩邊部位加每月換月；年化成本拖累 = (252 × 2／h + 24) × 平均單邊成本率 |
| 區間 | 每日序列的 circular block bootstrap，1,000 次、seed 11；價差區塊 max(5, 2h)、Rank IC 區塊 20 |
| 停止規則（事前） | futures_liquid、h = 10、P1 ∪ P2 的產業中性多空毛年化 < 2 × 年化成本拖累 → 不建策略 |
| 程式與產物 | `backtest/twbacktest/exposure_diagnostics.py`（合成資料測試 `backtest/tests/test_exposure_diagnostics.py`）；輸出 `artifacts/diagnostics/backtest/exposure-step0-b0-d0-20261024T120000Z/`（`grid.csv`、`daily.csv`、`report.json`、`summary.md`） |

### 3. 結果

#### 3.1 股票期貨流動宇宙（合併排名，約 123–144 檔）

| 模型 | h | 期間 | Rank IC 殘差 [95%] | Rank IC 原始 | 殘差四分位價差／批 | 原始四分位價差／批 | 多空毛年化 plain | 多空毛年化 產業中性 [95%] | 多空毛年化 規模中性 | 年化成本拖累 | 產業中性淨年化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 10 | P1 | 0.065 [0.038, 0.094] | 0.087 | +0.52% | +0.69% | +8.7% | +4.5% [−1.1%, +10.6%] | +9.2% | 18.3% | −13.8% |
| B0 | 10 | P2 | **−0.007 [−0.049, +0.039]** | −0.017 | −1.17% | −2.30% | −28.9% | −14.0% [−26.7%, −2.2%] | −27.4% | 18.5% | −32.6% |
| B0 | 10 | P1 ∪ P2 | 0.045 [0.019, 0.070] | 0.058 | +0.05% | −0.14% | −1.7% | **−0.6% [−7.0%, +5.4%]** | −0.9% | 18.3% | −19.0% |
| D0 | 10 | P1 | 0.052 [0.030, 0.077] | 0.071 | +0.36% | +0.44% | +5.5% | +3.6% [−0.9%, +8.9%] | +5.7% | 18.3% | −14.7% |
| D0 | 10 | P2 | **+0.006 [−0.038, +0.051]** | −0.000 | −0.69% | −1.37% | −17.3% | −7.9% [−21.9%, +5.6%] | −15.9% | 18.5% | −26.4% |
| D0 | 10 | P1 ∪ P2 | 0.039 [0.017, 0.062] | 0.051 | +0.07% | −0.06% | −0.8% | **+0.4% [−5.6%, +6.2%]** | −0.2% | 18.3% | −17.9% |
| B0 | 5 | P1 | 0.076 [0.055, 0.096] | 0.094 | +0.47% | +0.62% | +15.6% | +10.9% [+5.8%, +16.4%] | +15.7% | 30.6% | −19.8% |
| B0 | 5 | P2 | 0.028 [−0.000, 0.056] | 0.022 | −0.24% | −0.67% | −16.8% | −7.9% [−21.1%, +7.3%] | −16.4% | 31.1% | −39.0% |
| B0 | 5 | P1 ∪ P2 | 0.062 [0.045, 0.080] | 0.074 | +0.27% | +0.26% | +6.6% | +5.7% [−0.4%, +11.9%] | +6.8% | 30.8% | −25.1% |
| D0 | 5 | P1 | 0.066 [0.050, 0.083] | 0.082 | +0.37% | +0.48% | +12.2% | +8.0% [+3.5%, +13.0%] | +11.7% | 30.6% | −22.6% |
| D0 | 5 | P2 | 0.031 [−0.003, 0.064] | 0.027 | −0.12% | −0.45% | −11.2% | −2.0% [−17.6%, +13.0%] | −12.9% | 31.1% | −33.1% |
| D0 | 5 | P1 ∪ P2 | 0.056 [0.041, 0.072] | 0.067 | +0.24% | +0.23% | +5.7% | +5.2% [−0.2%, +11.2%] | +4.9% | 30.8% | −25.5% |

#### 3.2 流動性宇宙（市場內排名，約 350–363 檔）

| 模型 | h | 期間 | Rank IC 殘差 [95%] | Rank IC 原始 | 殘差四分位價差／批 | 原始四分位價差／批 | 多空毛年化 plain | 多空毛年化 產業中性 [95%] | 多空毛年化 規模中性 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 10 | P1 | 0.091 [0.074, 0.109] | 0.102 | +0.88% | +0.79% | +10.0% | +8.7% [+4.5%, +13.3%] | +9.5% |
| B0 | 10 | P2 | 0.053 [0.027, 0.080] | 0.026 | +0.06% | −1.30% | −16.3% | −4.5% [−12.0%, +2.8%] | −10.4% |
| B0 | 10 | P1 ∪ P2 | 0.080 [0.063, 0.096] | 0.081 | +0.65% | +0.21% | +2.7% | +5.0% [+0.5%, +9.4%] | +4.0% |
| D0 | 10 | P1 | 0.089 [0.074, 0.104] | 0.103 | +0.96% | +0.95% | +12.0% | +10.1% [+6.3%, +13.9%] | +11.5% |
| D0 | 10 | P2 | 0.057 [0.030, 0.085] | 0.039 | +0.34% | −0.54% | −6.8% | +0.9% [−7.2%, +9.5%] | −2.8% |
| D0 | 10 | P1 ∪ P2 | 0.080 [0.065, 0.095] | 0.085 | +0.79% | +0.54% | +6.8% | +7.6% [+3.3%, +11.4%] | +7.6% |
| B0 | 5 | P1 ∪ P2 | 0.084 [0.072, 0.095] | 0.088 | +0.51% | +0.34% | +8.7% | +10.0% [+5.7%, +14.6%] | +9.7% |
| D0 | 5 | P1 ∪ P2 | 0.084 [0.073, 0.095] | 0.091 | +0.55% | +0.46% | +11.5% | +11.5% [+7.3%, +15.8%] | +12.2% |

流動性宇宙的成本欄以股票期貨成本列出只作參考（這些名字多數沒有股票期貨）；現股成本更高。

#### 3.3 停止規則

| 模型 | 產業中性毛年化（futures_liquid，h = 10，P1 ∪ P2） | 年化成本拖累 | 門檻（2 倍） | 判定 |
| --- | --- | --- | --- | --- |
| B0 | −0.6% | 18.3% | 36.7% | 停止 |
| D0 | +0.4% | 18.3% | 36.7% | 停止 |

### 4. 判讀

- Rank IC 與可交易價差是兩件事：IC 0.09 在 10 日殘差報酬標準差約 3–4% 的宇宙裡，前後四分位的價差期望值就是不到 1%，年化後 9–10%，這在 P1 已是最好的情況。
- 多空組合把因子雜訊中性化後，剩下的殘差 alpha 不足以支付每邊 0.25% 的期貨成本與月換月；持有拉長到 10 日已把成本攤到 18%／年，再拉長 horizon 殘差排序能力本身會繼續衰減（§13.16 長格）。
- 在股票期貨宇宙內 P2 的 Rank IC 為 0，表示 P2 殘差排序能力集中在中小型股；這些名字只能做多、現股成本更高，而且多頭組合會把市值暴露帶回來（§13.14 已失敗）。
- 本診斷以現股價格代表期貨、用 tick 模型估滑價（§13.15 的價差普查顯示 tick 模型與實際收盤報價相符）、用統一的換手假設估成本；band 規則可降低換手，但毛年化接近零時沒有可保留的部分。

### 5. 判讀限制

- 契約代碼以 2026-10 的標的清單對應，期間內下市的股票期貨無法對應（存活者偏誤，38 個代碼）。
- 現股價格代表期貨：忽略基差與 8:45–9:00 的流動性差異；多日持有下影響小，但不是零。
- 四分位、產業併類門檻（4 檔）、成本參數都是事前固定的合理值，不是最佳化結果。
- P2 已經看過；本診斷是描述性的，不含選擇，但結論仍以 P3 為準。

### 6. 處置

依 §13.17 第 0 項停止。模型線的最終結論見[殘差化標籤報告](residual_labels_results.md)第 1 節與 [`MODEL_DEVELOPMENT_ROADMAP.md`](../MODEL_DEVELOPMENT_ROADMAP.md)。

## English

### 1. Purpose and conclusions

Before the exposure-neutral strategies registered in [experiment protocol §13.17](experiment_protocol.md) (0050 core plus a stock-futures long-short overlay, with the bar of beating 0050 buy-and-hold) were built, a descriptive diagnostic asked two questions: how large is the tradable spread that the residual-label B0 and D0 multi-day rankings produce, and how does it compare with Taiwan stock-futures execution costs. The diagnostic had a pre-registered stop rule. This report records its results and the disposition.

Conclusions:

1. **The stop rule fires for both models; no strategy is built.** Inside the liquid stock-futures universe at h = 10 on P1 ∪ P2, the industry-neutral, dollar-neutral top-minus-bottom-quartile long-short earns a gross −0.6% [−7.0%, +5.4%] a year for B0 and +0.4% [−5.6%, +6.2%] for D0; the same book's annual cost drag is 18.3%, and the bar (twice the drag) 36.7%. At h = 5 the gross is +5.7% / +5.2% against a 30.8% drag. The plain and size-neutral constructions reach the same verdict.
2. **The residual alpha is too small.** Even in the full liquidity universe during P1, the best stretch, the top-minus-bottom quartile residual spread is only 0.88% (B0) / 0.96% (D0) per 10 sessions, an industry-neutral long-short gross of +8.7% [+4.5%, +13.3%] / +10.1% [+6.3%, +13.9%] a year; on P1 ∪ P2 it falls to +5.0% / +7.6%, and P2 alone is −4.5% / +0.9%. This is what a Rank IC of 0.08–0.10 is worth as a tradable spread.
3. **The alpha is not in the shortable universe.** Among the roughly 123–144 large caps with liquid stock futures, the residual Rank IC drops to zero in P2 (B0 −0.007, D0 +0.006, intervals spanning 0) and is only 0.045 / 0.039 on P1 ∪ P2; the P2 residual quartile spread is negative (−1.17% / −0.69% per 10 sessions). The P2 multi-day ranking skill found in §13.16 lives entirely in small and mid caps without stock futures, which can only be bought and cost more to trade.
4. **Neither S1 nor S2 can beat 0050.** S1's alpha leg (the long-short overlay) is zero inside the shortable universe; S2's (a long-only satellite) ceiling is the long side of the liquidity universe, a gross of about 3–5% a year, below the cost drag of holding stocks for 10 sessions (0.3% transaction tax plus commission and slippage; replacing one tenth of the book daily costs about 12% a year). Both would also carry the cash reserve's certain drag in an up year.
5. **Disposition**: under §13.17 item 0, S1 / S2 / S3, H-G and `exposure_neutral.json` are not implemented. The model line's conclusion: it has h = 1 ranking skill (shown untradable at execution costs in §13.15) and residual multi-day ranking skill (shown here to be too small and absent from the shortable universe); under Taiwan's instruments and costs there is no deployable strategy that beats 0050 buy-and-hold.

### 2. Data and method

| Item | Content |
| --- | --- |
| Signals | Walk-forward scores of B0 `t-lgbm-alpha-residual_v1a` and D0 `t-xs-scratch` (three-seed ensemble) on the `residual_v1a` labels, h ∈ {5, 10}; residual targets from the same run's `targets.parquet` |
| Prices | The dataset's raw `market.parquet`; returns from the next open to the close h sessions later (the label's holding), before costs; stock prices stand in for the stock futures (small basis noise over multi-day holds) |
| Universes | liquidity: the §13.14 per-market universe (20-session median dollar volume ≥ NT$30M on TWSE / 10M on TPEX, close ≥ NT$10, no ETFs), ranked within the market; futures_liquid: the §13.15 liquid stock-futures universe (near-month 20-session mean volume ≥ 100 contracts, traded in the regular session on the signal date), ranked across both markets. TAIFEX all-contract daily data and the underlying list as in §13.15 (862 sessions, 314 matched contracts, about 243 names with stock futures per day, 38 delisted codes unmatched) |
| Portfolios | Top and bottom quartiles, dollar-neutral; plain = direct ranking; industry = ranking inside every two-digit industry of the reference table with weights proportional to name counts (fewer than 4 names pooled), so both sides carry the same industry weights; size = ranking inside the two halves by the date's median dollar volume. Gross annual for gross notional = capital (half per side), cohort return × 252 / h |
| Costs | Stock futures per side: 0.002% tax, NT$30 commission per contract, one tick of slippage (tick bands as in §13.15, contracts sized by the listing's standard share count); 1/h of both sides replaced daily plus a monthly roll; annual drag = (252 × 2 / h + 24) × mean per-side cost rate |
| Intervals | Circular block bootstrap of the daily series, 1,000 replicates, seed 11; block max(5, 2h) for spreads and 20 for Rank IC |
| Stop rule (pre-registered) | futures_liquid, h = 10, P1 ∪ P2: industry-neutral long-short gross < 2 × annual cost drag → no strategy |
| Code and outputs | `backtest/twbacktest/exposure_diagnostics.py` (synthetic tests in `backtest/tests/test_exposure_diagnostics.py`); outputs under `artifacts/diagnostics/backtest/exposure-step0-b0-d0-20261024T120000Z/` (`grid.csv`, `daily.csv`, `report.json`, `summary.md`) |

### 3. Results

#### 3.1 Liquid stock-futures universe (ranked across markets, about 123–144 names)

| Model | h | Period | Rank IC residual [95%] | Rank IC raw | Residual quartile spread / cohort | Raw quartile spread / cohort | L/S gross annual plain | L/S gross annual industry-neutral [95%] | L/S gross annual size-neutral | Annual cost drag | Industry-neutral net annual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 10 | P1 | 0.065 [0.038, 0.094] | 0.087 | +0.52% | +0.69% | +8.7% | +4.5% [−1.1%, +10.6%] | +9.2% | 18.3% | −13.8% |
| B0 | 10 | P2 | **−0.007 [−0.049, +0.039]** | −0.017 | −1.17% | −2.30% | −28.9% | −14.0% [−26.7%, −2.2%] | −27.4% | 18.5% | −32.6% |
| B0 | 10 | P1 ∪ P2 | 0.045 [0.019, 0.070] | 0.058 | +0.05% | −0.14% | −1.7% | **−0.6% [−7.0%, +5.4%]** | −0.9% | 18.3% | −19.0% |
| D0 | 10 | P1 | 0.052 [0.030, 0.077] | 0.071 | +0.36% | +0.44% | +5.5% | +3.6% [−0.9%, +8.9%] | +5.7% | 18.3% | −14.7% |
| D0 | 10 | P2 | **+0.006 [−0.038, +0.051]** | −0.000 | −0.69% | −1.37% | −17.3% | −7.9% [−21.9%, +5.6%] | −15.9% | 18.5% | −26.4% |
| D0 | 10 | P1 ∪ P2 | 0.039 [0.017, 0.062] | 0.051 | +0.07% | −0.06% | −0.8% | **+0.4% [−5.6%, +6.2%]** | −0.2% | 18.3% | −17.9% |
| B0 | 5 | P1 | 0.076 [0.055, 0.096] | 0.094 | +0.47% | +0.62% | +15.6% | +10.9% [+5.8%, +16.4%] | +15.7% | 30.6% | −19.8% |
| B0 | 5 | P2 | 0.028 [−0.000, 0.056] | 0.022 | −0.24% | −0.67% | −16.8% | −7.9% [−21.1%, +7.3%] | −16.4% | 31.1% | −39.0% |
| B0 | 5 | P1 ∪ P2 | 0.062 [0.045, 0.080] | 0.074 | +0.27% | +0.26% | +6.6% | +5.7% [−0.4%, +11.9%] | +6.8% | 30.8% | −25.1% |
| D0 | 5 | P1 | 0.066 [0.050, 0.083] | 0.082 | +0.37% | +0.48% | +12.2% | +8.0% [+3.5%, +13.0%] | +11.7% | 30.6% | −22.6% |
| D0 | 5 | P2 | 0.031 [−0.003, 0.064] | 0.027 | −0.12% | −0.45% | −11.2% | −2.0% [−17.6%, +13.0%] | −12.9% | 31.1% | −33.1% |
| D0 | 5 | P1 ∪ P2 | 0.056 [0.041, 0.072] | 0.067 | +0.24% | +0.23% | +5.7% | +5.2% [−0.2%, +11.2%] | +4.9% | 30.8% | −25.5% |

#### 3.2 Liquidity universe (ranked within the market, about 350–363 names)

| Model | h | Period | Rank IC residual [95%] | Rank IC raw | Residual quartile spread / cohort | Raw quartile spread / cohort | L/S gross annual plain | L/S gross annual industry-neutral [95%] | L/S gross annual size-neutral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 10 | P1 | 0.091 [0.074, 0.109] | 0.102 | +0.88% | +0.79% | +10.0% | +8.7% [+4.5%, +13.3%] | +9.5% |
| B0 | 10 | P2 | 0.053 [0.027, 0.080] | 0.026 | +0.06% | −1.30% | −16.3% | −4.5% [−12.0%, +2.8%] | −10.4% |
| B0 | 10 | P1 ∪ P2 | 0.080 [0.063, 0.096] | 0.081 | +0.65% | +0.21% | +2.7% | +5.0% [+0.5%, +9.4%] | +4.0% |
| D0 | 10 | P1 | 0.089 [0.074, 0.104] | 0.103 | +0.96% | +0.95% | +12.0% | +10.1% [+6.3%, +13.9%] | +11.5% |
| D0 | 10 | P2 | 0.057 [0.030, 0.085] | 0.039 | +0.34% | −0.54% | −6.8% | +0.9% [−7.2%, +9.5%] | −2.8% |
| D0 | 10 | P1 ∪ P2 | 0.080 [0.065, 0.095] | 0.085 | +0.79% | +0.54% | +6.8% | +7.6% [+3.3%, +11.4%] | +7.6% |
| B0 | 5 | P1 ∪ P2 | 0.084 [0.072, 0.095] | 0.088 | +0.51% | +0.34% | +8.7% | +10.0% [+5.7%, +14.6%] | +9.7% |
| D0 | 5 | P1 ∪ P2 | 0.084 [0.073, 0.095] | 0.091 | +0.55% | +0.46% | +11.5% | +11.5% [+7.3%, +15.8%] | +12.2% |

The cost columns of the liquidity universe (in `grid.csv`) use stock-futures costs for reference only; most of these names have no stock futures, and stock trading costs more.

#### 3.3 Stop rule

| Model | Industry-neutral gross annual (futures_liquid, h = 10, P1 ∪ P2) | Annual cost drag | Bar (2×) | Verdict |
| --- | --- | --- | --- | --- |
| B0 | −0.6% | 18.3% | 36.7% | stop |
| D0 | +0.4% | 18.3% | 36.7% | stop |

### 4. Reading

- Rank IC and a tradable spread are different things: with a 10-session residual-return dispersion of about 3–4%, an IC of 0.09 implies a top-minus-bottom quartile spread below 1%, or 9–10% a year, and that is the P1 best case.
- Once the factor noise is neutralized, the remaining residual alpha cannot pay a 0.25% per-side futures cost plus monthly rolls; a 10-session hold already spreads the cost to 18% a year, and longer horizons lose ranking skill themselves (§13.16 long cell).
- A P2 Rank IC of zero inside the stock-futures universe means the P2 residual skill is concentrated in small and mid caps, which can only be bought, cost more, and bring the size exposure back into a long portfolio (which already failed in §13.14).
- The diagnostic uses stock prices for futures, a tick model for slippage (the §13.15 spread census showed the tick model matches actual closing quotes) and a uniform turnover assumption; a band rule lowers turnover, but there is nothing to keep when the gross is near zero.

### 5. Reading limits

- Contract codes are matched with the 2026-10 underlying list; stock futures delisted during the period cannot be matched (survivorship bias, 38 codes).
- Stock prices stand in for futures, ignoring the basis and the thinner 08:45–09:00 liquidity; small over multi-day holds but not zero.
- The quartile, the industry pooling threshold (4 names) and the cost parameters are pre-fixed reasonable values, not optimized ones.
- P2 had already been seen; the diagnostic is descriptive and involves no selection, but P3 remains the confirmation.

### 6. Disposition

Stopped under §13.17 item 0. The model line's final conclusion is in section 1 of the [residualized labels report](residual_labels_results.md) and in [`MODEL_DEVELOPMENT_ROADMAP.md`](../MODEL_DEVELOPMENT_ROADMAP.md).
