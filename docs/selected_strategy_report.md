# 選定模型與策略：P3 回測結果 / Selected Model and Strategy: P3 Backtest Results

> 狀態（2026-10-29）：§13.20 計分完成。選定規則選出 F2 × S-A；P3 確認 H-J 未通過（回撤條件），**維持研究狀態，沒有策略進入試運行**。完整的實驗史、設計理由、全部表格與 roadmap 見 `docs/research_full_report.md`。
> Status (2026-10-29): §13.20 scored. The selection rule picks F2 × S-A; the P3 confirmation H-J fails on the drawdown condition, so **research status stays and no strategy enters paper trading**. The full experiment history, design rationale, all tables and the roadmap are in `docs/research_full_report.md`.

[繁體中文](#繁體中文) · [English](#english)

---

## 繁體中文

### 1. 選定結果

- **模型**：**F2** = B0 LightGBM 橫斷面排序模型（`t-lgbm-alpha-residual_v1a`），特徵 `alpha_chip_tw_v1`（183 欄價量 + 67 欄免費籌碼），標籤 `residual_v1a`（h 日持有報酬扣掉市場、市值、產業效應的殘差），seed 42／43／44 集成，walk-forward `tw_quarterly_v2`（2016 起擴張訓練、季度折到 2026Q4、16 折、672 個 fit；`artifacts/lgbm/launch-20261028T211653Z-192011424/`）。對照 F0 = 同一模型、只用 183 欄價量（`launch-20261028T154640Z-242638914`）。
- **策略**：**S-A 週頻低換手多頭 band**，K = 30、m = 2（在 TWSE、P1、機構成本、分數位置以淨 Sharpe 選出，之後固定）、單週換手上限 50%。
- **選定規則（事前登記）**：機構主要成本、TW（TWSE 與 TPEX 各半）、分數位置，P1 ∪ P2（2023-01-02 至 2026-05-29）淨 Sharpe 最高的（模型, 策略）格：

  | 候選 | 策略 | P1 ∪ P2 淨 Sharpe | 年化淨報酬 | 95% 區間 | 最大回撤 |
  | --- | --- | --- | --- | --- | --- |
  | **F2** | **S-A** | **2.38** | +51.8% | [+26.1%, +77.7%] | −22.6% |
  | F0 | S-A | 2.07 | +35.5% | [+17.2%, +54.7%] | −18.7% |
  | F2 | S2 | 1.98 | +25.4% | [+13.9%, +38.0%] | −6.1% |
  | F0 | S2 | 1.78 | +24.3% | [+10.4%, +38.0%] | −8.5% |
  | F2 | S1 | 1.57 | +19.1% | [+7.5%, +31.8%] | −8.4% |
  | F0 | S1 | 1.44 | +18.3% | [+3.6%, +32.2%] | −10.2% |

- **H-J（P3 確認，2026-06-01 至 2026-10-02 的 85 個開盤到開盤區間）**：(i) P3 年化淨報酬 +3.5% [−110.8%, +110.6%] > 0，成立；(ii) P3 最大回撤 −25.7%（峰 2026-07-03、谷 07-29）對 P1 ∪ P2 的 −22.6%，不成立。**未通過 → 維持研究狀態**，依登記不重選參數、不改建構。原因：2026 年 7 月等權流動性宇宙 −17.0%、0050 −6.5%，100% 多頭的 S-A 跌 −19.5%；市場中性的 S2 同月 +15.6%。
- 其餘五格的 H-J 只作資訊（不改變選定）：F0 S-A、F0 S2、F0 S1 兩條件都成立；F2 S2（P3 回撤 −8.1% 對 −6.1%）與 F2 S1（−8.6% 對 −8.4%）(ii) 不成立。

### 2. 策略規則

| 項目 | S1 多日多空 | S2 = S1 + 台指期 | **S-A 週頻低換手多頭（選定）** |
| --- | --- | --- | --- |
| 訊號 | 每日：h = 10 分數的宇宙內百分位，EMA 平滑（半衰期 3 日），對 60 日 beta、前 20 日與前 60 日報酬每日做橫斷面 OLS 取殘差（V1） | 同 S1 | 每個 ISO 週最後一個交易日：h = 5 分數，同樣的平滑與正交化 |
| 宇宙 | 只含股票；前 20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬；收盤價 ≥ 10 元 | 同 | 同 |
| 多頭 | 前 K = 50 等額位置，掉出前 150（m = 3）才賣；每日新進最多 5 檔 | 同 | 前 **30** 等額位置，掉出前 **60**（m = 2）才賣；每週最多換掉 15 檔（50% × K），超過只賣分數最差的；停牌出場與期初建倉的補位不計入上限 |
| 空頭 | 反向排序的 band，位置每日重設；只放空可借券名單內（機構 `sbl`、散戶 `margin`）；交易所次日限額封頂；借不到機率依規模 2%／10%／25%；借券費 1.5%／3%／6% | 同 | 無 |
| 疊加 | 無 | 台指期 −β × 資本，β = 組合毛報酬對指數的前 60 日滾動 β | 無 |
| 成交 | 訊號日次一交易日開盤；漲停開盤不買、跌停開盤不賣（順延） | 同 | 同；再平衡之間只持有，跌停擋住的賣單次日再試 |
| 曝險 | 金額中性，毛 2 倍 | 金額中性加 β 對沖 | 多頭 100%（空位留現金） |

### 3. 成本與資本

| 成本檔 | 手續費（每邊） | 證交稅（賣出） | 滑價（每邊） | 空頭 | 台指期 |
| --- | --- | --- | --- | --- | --- |
| 機構（主要） | 0.03% | 0.3% | TWSE 0.05%、TPEX 0.05% | 借券年費 2%（分層 1.5／3／6%） | 期交稅 0.002%、每口 30 元、1 tick |
| 散戶 | 0.0855% | 0.3% | TWSE 0.1%、TPEX 0.2% | 融券手續費 0.08% | 每口 50 元、1 tick |

資本情境：分數位置（主要）、機構 3 億與 10 億（每檔名目封頂於 5% × 20 日成交金額中位數，超出留現金）。滑價 0／1／2 倍另列於完整報告。

### 4. 回測結果（選定格 F2 × S-A，分數位置、主要滑價）

| 成本檔 | 市場 | 期間 | 年化淨報酬 | 95% 區間 | 淨 Sharpe | 最大回撤 | 換手／日 | 成本拖累 | 日數 | TAIEX 報酬指數 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 機構 | TW | P1 | +27.6% | [+5.3%, +50.0%] | 1.44 | −22.6% | 5.9% | 6.6% | 561 | +18.6% |
| 機構 | TW | P2 | +111.2% | [+60.0%, +173.9%] | 4.13 | −11.7% | 5.9% | 8.0% | 229 | +86.8% |
| 機構 | TW | P1 ∪ P2 | +51.8% | [+26.1%, +77.7%] | 2.38 | −22.6% | 5.9% | 7.0% | 790 | +38.3% |
| 機構 | TW | **P3** | **+3.5%** | [−110.8%, +110.6%] | 0.08 | **−25.7%** | 5.9% | 9.0% | 85 | +13.9% |
| 機構 | TWSE | P1 ∪ P2 | +54.5% | [+29.1%, +82.7%] | 2.59 | −20.8% | 5.6% | 6.9% | 790 | +43.0% |
| 機構 | TWSE | P3 | −14.8% | [−151.3%, +114.6%] | −0.33 | −28.7% | 5.6% | 9.0% | 85 | +28.6% |
| 機構 | TPEX | P1 ∪ P2 | +49.1% | [+19.7%, +77.3%] | 1.96 | −29.5% | 6.1% | 7.1% | 790 | +33.7% |
| 機構 | TPEX | P3 | +21.7% | [−73.9%, +117.2%] | 0.49 | −23.3% | 6.1% | 9.0% | 85 | −0.8% |
| 散戶 | TW | P1 ∪ P2 | +47.0% | [+21.6%, +72.9%] | 2.17 | −22.8% | 5.9% | 11.8% | 790 | +38.3% |
| 散戶 | TW | P3 | −2.3% | [−114.8%, +103.4%] | −0.05 | −25.9% | 5.9% | 14.8% | 85 | +13.9% |

- **對照（機構 TW）**：P1 ∪ P2 等權流動性宇宙 +31.5%（回撤 −18.7%）、0050 +48.3%（−25.2%）、TAIEX 報酬指數 +38.3%；P3 等權宇宙 +12.1%、0050 +25.8%（−16.1%）。
- **P3 各月（機構 TW）**：6 月 +9.7%、7 月 −19.5%（等權宇宙 −17.0%、0050 −6.5%）、8 月 +8.6%、9 月 +2.4%、10 月（兩日）0.0%。
- **逐年（機構 TW）**：2023 +38.2%（Sharpe 2.54、回撤 −8.9%）、2024 +35.1%（1.84、−13.8%）、2025 +27.9%（1.34、−22.6%）、2026 至 10-02 +93.1%（2.52、−25.7%）。
- **資本情境（機構 TW）**：3 億 P1 ∪ P2 +37.9%（Sharpe 2.34、部署率 59%、30% 候選買單被參與率上限擋下）、P3 +15.8%（回撤 −18.7%）；10 億 +32.1%（2.12、部署率 50%、擋下 64%）、P3 +16.6%（−15.9%）。
- **週換手**：TWSE 25.8%、TPEX 27.7%（每次再平衡換掉的比例，上限 50%）；每日換手 5.6% 到 6.1%，成本拖累 6% 到 7%／年。
- **籌碼增量（資訊性）**：S-A F2 − F0 每日淨報酬配對差，P1 ∪ P2 年化 TW +16.3% [+5.5%, +27.7%]、TWSE +18.6% [+6.3%, +32.2%]、TPEX +14.1% [+0.4%, +27.9%]；P3 TW −24.1%、TWSE −72.9%、TPEX +24.8%。

### 5. 限制

1. **P3 只有 85 個區間**（2026-06-01 至 10-02；最後 14 個交易日沒有標籤就沒有預測列），市場狀態單一且含 7 月一次 −17% 的修正；區間寬 ±110 個百分點，通過或不通過都不能外推成長期結論。
2. **P2 已看過**：S1／S2 的正交化設計在看過 P2 之後決定，P1 ∪ P2 的數字只是資訊；S-A 參數在 P1 選定，P2 與 P3 都是樣本外。
3. **S-A 是 100% 多頭、β 接近 1**：回撤由市場決定，H-J (ii) 以最大回撤為準，在含修正的短窗口幾乎必然不過；市場中性的 S1／S2 同窗口 +23% 到 +34%、回撤 −6% 到 −9%，但依登記不重選。
4. **資料集修訂 v1**：`adjusted_close` 只還原股息，不含分割、減資與面額變更（2023 年起 109 檔有超過 25% 的單日還原開盤跳動；P3 期間 48 次、42 檔）；部位報酬以相鄰交易日的開盤價計算、停牌即出場，不受影響，但模型分數在事件前後受影響。v2 修正尚未訓練任何模型。
5. 籌碼特徵只含免費官方品項，公布時刻用保守規則（T+1 開盤前）；分點與集保未購買。
6. 空頭腿的借不到機率、召回機率與借券費分層是參數假設（五個隨機重複平均），不是實際成交紀錄；S-A 不放空，不受此影響。
7. 樣本外視窗含 2025-04 急跌與 2026-07 修正，不含 2022 年空頭。

### 6. 重現

```bash
cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/launch-20261028T154640Z-242638914/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/launch-20261028T211653Z-192011424/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/98d284dfb9d280063b7eb708414e68b3374d50a934e8fe06c743aa73a6f3e760/raw/market.parquet
```

```bash
cd backtest && uv run python -m twbacktest.strategy_decision --suite ../artifacts/diagnostics/backtest/scenario-suite-p3-f0-f2-20261029T050256Z
```

輸出目錄：回測 `artifacts/diagnostics/backtest/scenario-suite-p3-f0-f2-20261029T050256Z/`、選定與 H-J `artifacts/diagnostics/backtest/strategy-decision-p3-f0-f2-20261029T050256Z/`。

---

## English

### 1. Selection

- **Model**: **F2** = the B0 LightGBM cross-sectional ranker (`t-lgbm-alpha-residual_v1a`) on `alpha_chip_tw_v1` (183 price-volume + 67 free chip columns), labels `residual_v1a` (the h-session holding return net of market, size and industry effects), seeds 42 / 43 / 44 ensembled, walk-forward `tw_quarterly_v2` (expanding training from 2016, quarterly folds through 2026Q4, 16 folds, 672 fits; `artifacts/lgbm/launch-20261028T211653Z-192011424/`). The control F0 is the same model on the 183 price-volume columns only (`launch-20261028T154640Z-242638914`).
- **Strategy**: **S-A, the weekly low-turnover long band**, K = 30, m = 2 (selected on TWSE, P1, institution cost, fractional by net Sharpe, then fixed), 50% weekly turnover cap.
- **Selection rule (pre-registered)**: institution primary cost, TW (TWSE and TPEX half each), fractional; the (model, strategy) cell with the highest P1 ∪ P2 (2023-01-02 to 2026-05-29) net Sharpe:

  | Candidate | Strategy | P1 ∪ P2 net Sharpe | Net return (annual) | 95% interval | Max drawdown |
  | --- | --- | --- | --- | --- | --- |
  | **F2** | **S-A** | **2.38** | +51.8% | [+26.1%, +77.7%] | −22.6% |
  | F0 | S-A | 2.07 | +35.5% | [+17.2%, +54.7%] | −18.7% |
  | F2 | S2 | 1.98 | +25.4% | [+13.9%, +38.0%] | −6.1% |
  | F0 | S2 | 1.78 | +24.3% | [+10.4%, +38.0%] | −8.5% |
  | F2 | S1 | 1.57 | +19.1% | [+7.5%, +31.8%] | −8.4% |
  | F0 | S1 | 1.44 | +18.3% | [+3.6%, +32.2%] | −10.2% |

- **H-J (P3 confirmation, the 85 open-to-open intervals from 2026-06-01 to 2026-10-02)**: (i) P3 annualized net return +3.5% [−110.8%, +110.6%] > 0 holds; (ii) P3 maximum drawdown −25.7% (peak 2026-07-03, trough 07-29) against P1 ∪ P2's −22.6% fails. **Not passed → research status stays**, with no re-selection and no construction change, as registered. Cause: in July 2026 the equal-weight liquidity universe lost −17.0% and 0050 −6.5%; the 100% long S-A lost −19.5% while the market-neutral S2 made +15.6% in the same month.
- H-J of the other five cells, informational only (no re-selection): F0 S-A, F0 S2 and F0 S1 satisfy both conditions; F2 S2 (P3 drawdown −8.1% vs −6.1%) and F2 S1 (−8.6% vs −8.4%) fail (ii).

### 2. Strategy rules

| Item | S1 multi-day long-short | S2 = S1 + TX overlay | **S-A weekly low-turnover long (selected)** |
| --- | --- | --- | --- |
| Signal | daily: universe percentile of the h = 10 score, EMA-smoothed (half-life 3 sessions), cross-sectional OLS residual each day on 60-session beta and trailing 20- and 60-session returns (v1) | as S1 | last session of each ISO week: the h = 5 score, same smoothing and orthogonalization |
| Universe | stocks only; 20-session median dollar volume TWSE ≥ NT$30M, TPEX ≥ NT$10M; close ≥ NT$10 | same | same |
| Long book | top K = 50 equal slots, sell only out of the top 150 (m = 3); at most 5 entries a day | same | top **30** equal slots, sell only out of the top **60** (m = 2); at most 15 replacements a week (50% × K), worst-ranked first; forced exits on halts and the initial build are refilled outside the cap |
| Short book | band on the reversed ranking, slots reset daily; only names on the borrowable list (institution `sbl`, retail `margin`); capped at the exchange's next-session quota; borrow-failure probability 2% / 10% / 25% by size; borrow fee 1.5% / 3% / 6% | same | none |
| Overlay | none | TX at −β × capital, β = trailing 60-session rolling beta of the book's gross return on the index | none |
| Execution | next session's open after the signal date; no buy at a limit-up open, no sell at a limit-down open (deferred) | same | same; between rebalances positions are only carried, a limit-down-blocked sale is retried the next session |
| Exposure | dollar neutral, gross 2× | dollar neutral plus β hedge | 100% long (empty slots in cash) |

### 3. Costs and capital

| Profile | Commission (per side) | Transaction tax (sale) | Slippage (per side) | Shorts | TX |
| --- | --- | --- | --- | --- | --- |
| Institution (primary) | 0.03% | 0.3% | TWSE 0.05%, TPEX 0.05% | borrow fee 2% a year (tiers 1.5 / 3 / 6%) | tax 0.002%, NT$30 per contract, 1 tick |
| Retail | 0.0855% | 0.3% | TWSE 0.1%, TPEX 0.2% | margin short fee 0.08% | NT$50 per contract, 1 tick |

Capital scenarios: fractional (primary), institution NT$300M and NT$1B (each name capped at 5% × the 20-session median dollar volume, the rest in cash). Slippage × 0 / 1 / 2 is reported in the full report.

### 4. Backtest results (the selected cell F2 × S-A, fractional, primary slippage)

| Profile | Market | Period | Net return (annual) | 95% interval | Net Sharpe | Max drawdown | Turnover / day | Cost drag | Days | TAIEX TR index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Institution | TW | P1 | +27.6% | [+5.3%, +50.0%] | 1.44 | −22.6% | 5.9% | 6.6% | 561 | +18.6% |
| Institution | TW | P2 | +111.2% | [+60.0%, +173.9%] | 4.13 | −11.7% | 5.9% | 8.0% | 229 | +86.8% |
| Institution | TW | P1 ∪ P2 | +51.8% | [+26.1%, +77.7%] | 2.38 | −22.6% | 5.9% | 7.0% | 790 | +38.3% |
| Institution | TW | **P3** | **+3.5%** | [−110.8%, +110.6%] | 0.08 | **−25.7%** | 5.9% | 9.0% | 85 | +13.9% |
| Institution | TWSE | P1 ∪ P2 | +54.5% | [+29.1%, +82.7%] | 2.59 | −20.8% | 5.6% | 6.9% | 790 | +43.0% |
| Institution | TWSE | P3 | −14.8% | [−151.3%, +114.6%] | −0.33 | −28.7% | 5.6% | 9.0% | 85 | +28.6% |
| Institution | TPEX | P1 ∪ P2 | +49.1% | [+19.7%, +77.3%] | 1.96 | −29.5% | 6.1% | 7.1% | 790 | +33.7% |
| Institution | TPEX | P3 | +21.7% | [−73.9%, +117.2%] | 0.49 | −23.3% | 6.1% | 9.0% | 85 | −0.8% |
| Retail | TW | P1 ∪ P2 | +47.0% | [+21.6%, +72.9%] | 2.17 | −22.8% | 5.9% | 11.8% | 790 | +38.3% |
| Retail | TW | P3 | −2.3% | [−114.8%, +103.4%] | −0.05 | −25.9% | 5.9% | 14.8% | 85 | +13.9% |

- **Comparison (institution TW)**: P1 ∪ P2 equal-weight liquidity universe +31.5% (drawdown −18.7%), 0050 +48.3% (−25.2%), TAIEX total-return index +38.3%; P3 equal-weight universe +12.1%, 0050 +25.8% (−16.1%).
- **P3 by month (institution TW)**: June +9.7%, July −19.5% (equal-weight universe −17.0%, 0050 −6.5%), August +8.6%, September +2.4%, October (two sessions) 0.0%.
- **By calendar year (institution TW)**: 2023 +38.2% (Sharpe 2.54, drawdown −8.9%), 2024 +35.1% (1.84, −13.8%), 2025 +27.9% (1.34, −22.6%), 2026 to 10-02 +93.1% (2.52, −25.7%).
- **Capital scenarios (institution TW)**: NT$300M P1 ∪ P2 +37.9% (Sharpe 2.34, deployed 59%, 30% of candidate buys blocked by the participation cap), P3 +15.8% (drawdown −18.7%); NT$1B +32.1% (2.12, deployed 50%, 64% blocked), P3 +16.6% (−15.9%).
- **Weekly turnover**: TWSE 25.8%, TPEX 27.7% (share of K replaced per rebalance, cap 50%); daily turnover 5.6% to 6.1%, cost drag 6% to 7% a year.
- **Chip increment (informational)**: S-A F2 − F0 paired daily net difference, P1 ∪ P2 annualized, TW +16.3% [+5.5%, +27.7%], TWSE +18.6% [+6.3%, +32.2%], TPEX +14.1% [+0.4%, +27.9%]; P3 TW −24.1%, TWSE −72.9%, TPEX +24.8%.

### 5. Limitations

1. **P3 has only 85 intervals** (2026-06-01 to 10-02; the last 14 sessions have no labels and therefore no prediction rows), one market regime that contains July's −17% correction; intervals are ±110 points wide, so passing or failing cannot be extrapolated into a long-run conclusion.
2. **P2 has been seen**: the orthogonalized design of S1 / S2 was decided after seeing P2, so P1 ∪ P2 figures are informational; S-A's parameters are selected on P1 and both P2 and P3 are out of sample for it.
3. **S-A is 100% long with a beta near one**: its drawdown is the market's, and H-J (ii) judges on maximum drawdown, which a short window with a correction almost certainly fails; the market-neutral S1 / S2 returned +23% to +34% with −6% to −9% drawdowns in the same window, but there is no re-selection as registered.
4. **Dataset revision v1**: `adjusted_close` carries dividends only, not splits, capital reductions or par-value changes (109 stocks since 2023 show a single-session adjusted-open jump above 25%; 48 such jumps across 42 stocks inside P3); position returns use adjacent sessions' opens and exit on halts, so they are unaffected, but the model scores around the events are. The v2 correction has trained no model yet.
5. The chip features cover only the free official items under the conservative availability rule (before the T+1 open); broker-branch and shareholder-dispersion data were not bought.
6. The short book's borrow-failure, recall and fee tiers are parameter assumptions (five random replicates averaged), not actual execution records; S-A holds no shorts and is unaffected.
7. The out-of-sample window contains the April 2025 crash and the July 2026 correction but not the 2022 bear market.

### 6. Reproduction

```bash
cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/launch-20261028T154640Z-242638914/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/launch-20261028T211653Z-192011424/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/98d284dfb9d280063b7eb708414e68b3374d50a934e8fe06c743aa73a6f3e760/raw/market.parquet
```

```bash
cd backtest && uv run python -m twbacktest.strategy_decision --suite ../artifacts/diagnostics/backtest/scenario-suite-p3-f0-f2-20261029T050256Z
```

Output directories: backtest `artifacts/diagnostics/backtest/scenario-suite-p3-f0-f2-20261029T050256Z/`, selection and H-J `artifacts/diagnostics/backtest/strategy-decision-p3-f0-f2-20261029T050256Z/`.
