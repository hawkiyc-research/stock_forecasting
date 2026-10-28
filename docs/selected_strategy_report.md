# 選定模型與策略：P3 回測結果 / Selected Model and Strategy: P3 Backtest Results

> 狀態（2026-10-28）：模型、策略與選定規則已固定；回測數字待 §13.20 的 P3 預測計分後填入（【待填】）。完整的實驗史、設計理由與 roadmap 見 `docs/research_full_report.md`。
> Status (2026-10-28): model, strategy and selection rule are fixed; the backtest figures are filled once the §13.20 P3 predictions are scored (【TBD】). The full experiment history, design rationale and roadmap are in `docs/research_full_report.md`.

[繁體中文](#繁體中文) · [English](#english)

---

## 繁體中文

### 1. 選定結果

- **模型**：【待填：F0 或 F2】——B0 LightGBM 橫斷面排序模型（`t-lgbm-alpha-residual_v1a`），標籤 `residual_v1a`（h 日持有報酬扣掉市場、市值、產業效應的殘差），seed 42／43／44 集成，walk-forward `tw_quarterly_v2`（2016 起擴張訓練、季度折到 2026Q4、16 折）。F0 = 183 欄價量特徵（`alpha_tw_v1`）；F2 = F0 + 67 欄免費籌碼特徵（`alpha_chip_tw_v1`）。
- **策略**：【待填：S1／S2／S-A】。
- **選定規則（事前登記）**：機構主要成本、TW（TWSE 與 TPEX 各半）、分數位置，P1 ∪ P2（2023-01-02 至 2026-05-29）淨 Sharpe 最高的（模型, 策略）格。候選：
  | 模型 | 策略 | P1 ∪ P2 淨 Sharpe | 年化淨報酬 | 最大回撤 |
  | --- | --- | --- | --- | --- |
  | 【待填】 | | | | |
- **H-J（P3 確認，2026-06-01 至 2026-10-22）**：(i) P3 年化淨報酬 > 0；(ii) P3 最大回撤不比 P1 ∪ P2 深。結果：【待填：通過／未通過】→ 狀態【待填：可進入試運行／維持研究】。

### 2. 策略規則

| 項目 | S1 多日多空 | S2 = S1 + 台指期 | S-A 週頻低換手多頭 |
| --- | --- | --- | --- |
| 訊號 | 每日：h = 10 分數的宇宙內百分位，EMA 平滑（半衰期 3 日），對 60 日 beta、前 20 日與前 60 日報酬每日做橫斷面 OLS 取殘差（V1） | 同 S1 | 每週最後一個交易日：h = 5 分數，同樣的平滑與正交化 |
| 宇宙 | 只含股票；前 20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬；收盤價 ≥ 10 元 | 同 | 同 |
| 多頭 | 前 K = 50 等額位置，掉出前 150（m = 3）才賣；每日新進最多 5 檔 | 同 | 前 K、掉出前 m × K 才賣；每週最多換掉 50% × K，超過只賣分數最差的；【待填：K、m】 |
| 空頭 | 反向排序的 band，位置每日重設；只放空可借券名單內（機構 `sbl`、散戶 `margin`）；交易所次日限額封頂；借不到機率依規模 2%／10%／25%；借券費 1.5%／3%／6% | 同 | 無 |
| 疊加 | 無 | 台指期 −β × 資本，β = 組合毛報酬對指數的前 60 日滾動 β | 無 |
| 成交 | 訊號日次一交易日開盤；漲停開盤不買、跌停開盤不賣（順延） | 同 | 同；再平衡之間只持有，跌停擋住的賣單次日再試 |
| 曝險 | 金額中性，毛 2 倍 | 金額中性加 β 對沖 | 多頭 100%（空位留現金） |

### 3. 成本與資本

| 成本檔 | 手續費（每邊） | 證交稅（賣出） | 滑價（每邊） | 空頭 | 台指期 |
| --- | --- | --- | --- | --- | --- |
| 機構（主要） | 0.03% | 0.3% | TWSE 0.05%、TPEX 0.05% | 借券年費 2%（分層 1.5／3／6%） | 期交稅 0.002%、每口 30 元、1 tick |
| 散戶 | 0.0855% | 0.3% | TWSE 0.1%、TPEX 0.2% | 融券手續費 0.08% | 每口 50 元、1 tick |

資本情境：分數位置（主要）、機構 3 億與 10 億（每檔名目封頂於 5% × 20 日成交金額中位數，超出留現金）。滑價 0／1／2 倍另列。

### 4. 回測結果（選定格）

| 成本檔 | 市場 | 期間 | 年化淨報酬 | 95% 區間 | 淨 Sharpe | 最大回撤 | 換手 | 成本拖累 | 日數 | TAIEX 報酬指數 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 機構 | TW | P1 | 【待填】 | | | | | | | |
| 機構 | TW | P2 | 【待填】 | | | | | | | |
| 機構 | TW | P1 ∪ P2 | 【待填】 | | | | | | | |
| 機構 | TW | **P3** | 【待填】 | | | | | | | |
| 機構 | TWSE | P1 ∪ P2／P3 | 【待填】 | | | | | | | |
| 機構 | TPEX | P1 ∪ P2／P3 | 【待填】 | | | | | | | |
| 散戶 | TW | P1 ∪ P2／P3 | 【待填】 | | | | | | | |

- 執行層敏感度（空頭腿借不到倍率 0／1／2、召回 0／0.5%）：【待填】。
- 資本情境 3 億／10 億：【待填：淨報酬、部署率、被封頂比例】。
- 逐年（2023、2024、2025、2026）：【待填：`yearly.csv`】。

### 5. 限制

1. **P3 只有約 95 個交易日**（2026-06 至 2026-10），市場狀態單一；通過或不通過都不能外推成長期結論。
2. **P2 已看過**：S1／S2 的正交化設計在看過 P2 之後決定，P1 ∪ P2 的數字只是資訊；S-A 參數在 P1 選定，P2 與 P3 都是樣本外。
3. **資料集修訂 v1**：`adjusted_close` 只還原股息，不含分割、減資與面額變更（2023 年起 109 檔有超過 25% 的單日還原開盤跳動）；部位報酬以相鄰交易日的開盤價計算、停牌即出場，不受影響，但模型分數在事件前後受影響。v2 修正尚未訓練任何模型。
4. 籌碼特徵只含免費官方品項，公布時刻用保守規則（T+1 開盤前）；分點與集保未購買。
5. 空頭腿的借不到機率、召回機率與借券費分層是參數假設（五個隨機重複平均），不是實際成交紀錄。
6. 樣本外視窗含 2025-04 急跌但不含 2022 年空頭。

### 6. 重現

```bash
cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/<F0-launch>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/<F2-launch>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

```bash
cd backtest && uv run python -m twbacktest.strategy_decision --suite ../artifacts/diagnostics/backtest/<scenario-suite 輸出目錄>
```

輸出目錄：【待填】。

---

## English

### 1. Selection

- **Model**: 【TBD: F0 or F2】, the B0 LightGBM cross-sectional ranker (`t-lgbm-alpha-residual_v1a`), labels `residual_v1a` (the h-session holding return net of market, size and industry effects), seeds 42 / 43 / 44 ensembled, walk-forward `tw_quarterly_v2` (expanding training from 2016, quarterly folds through 2026Q4, 16 folds). F0 = 183 price-volume features (`alpha_tw_v1`); F2 = F0 + 67 free chip features (`alpha_chip_tw_v1`).
- **Strategy**: 【TBD: S1 / S2 / S-A】.
- **Selection rule (pre-registered)**: institution primary cost, TW (TWSE and TPEX half each), fractional capital; the (model, strategy) cell with the highest P1 ∪ P2 (2023-01-02 to 2026-05-29) net Sharpe. Candidates:
  | Model | Strategy | P1 ∪ P2 net Sharpe | Net return (annual) | Max drawdown |
  | --- | --- | --- | --- | --- |
  | 【TBD】 | | | | |
- **H-J (P3 confirmation, 2026-06-01 to 2026-10-22)**: (i) P3 annualized net return > 0; (ii) P3 maximum drawdown not deeper than P1 ∪ P2's. Result: 【TBD: passed / not passed】 → status 【TBD: paper-trading candidate / research】.

### 2. Strategy rules

| Item | S1 multi-day long-short | S2 = S1 + TX overlay | S-A weekly low-turnover long |
| --- | --- | --- | --- |
| Signal | daily: universe percentile of the h = 10 score, EMA-smoothed (half-life 3 sessions), cross-sectional OLS residual each day on 60-session beta and trailing 20- and 60-session returns (v1) | as S1 | last session of each week: the h = 5 score, same smoothing and orthogonalization |
| Universe | stocks only; 20-session median dollar volume TWSE ≥ NT$30M, TPEX ≥ NT$10M; close ≥ NT$10 | same | same |
| Long book | top K = 50 equal slots, sell only out of the top 150 (m = 3); at most 5 entries a day | same | top K, sell only out of the top m × K; at most 50% × K replacements a week, worst-ranked first; 【TBD: K, m】 |
| Short book | band on the reversed ranking, slots reset daily; only names on the borrowable list (institution `sbl`, retail `margin`); capped at the exchange's next-session quota; borrow-failure probability 2% / 10% / 25% by size; borrow fee 1.5% / 3% / 6% | same | none |
| Overlay | none | TX at −β × capital, β = trailing 60-session rolling beta of the book's gross return on the index | none |
| Execution | next session's open after the signal date; no buy at a limit-up open, no sell at a limit-down open (deferred) | same | same; between rebalances positions are only carried, a limit-down-blocked sale is retried the next session |
| Exposure | dollar neutral, gross 2× | dollar neutral plus β hedge | 100% long (empty slots in cash) |

### 3. Costs and capital

| Profile | Commission (per side) | Transaction tax (sale) | Slippage (per side) | Shorts | TX |
| --- | --- | --- | --- | --- | --- |
| Institution (primary) | 0.03% | 0.3% | TWSE 0.05%, TPEX 0.05% | borrow fee 2% a year (tiers 1.5 / 3 / 6%) | tax 0.002%, NT$30 per contract, 1 tick |
| Retail | 0.0855% | 0.3% | TWSE 0.1%, TPEX 0.2% | margin short fee 0.08% | NT$50 per contract, 1 tick |

Capital scenarios: fractional (primary), institution NT$300M and NT$1B (each name capped at 5% × the 20-session median dollar volume, the rest in cash). Slippage × 0 / 1 / 2 reported separately.

### 4. Backtest results (the selected cell)

| Profile | Market | Period | Net return (annual) | 95% interval | Net Sharpe | Max drawdown | Turnover | Cost drag | Days | TAIEX TR index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Institution | TW | P1 | 【TBD】 | | | | | | | |
| Institution | TW | P2 | 【TBD】 | | | | | | | |
| Institution | TW | P1 ∪ P2 | 【TBD】 | | | | | | | |
| Institution | TW | **P3** | 【TBD】 | | | | | | | |
| Institution | TWSE | P1 ∪ P2 / P3 | 【TBD】 | | | | | | | |
| Institution | TPEX | P1 ∪ P2 / P3 | 【TBD】 | | | | | | | |
| Retail | TW | P1 ∪ P2 / P3 | 【TBD】 | | | | | | | |

- Execution-layer sensitivities (short-book failure multiplier 0 / 1 / 2, recall 0 / 0.5%): 【TBD】.
- Capital scenarios NT$300M / NT$1B: 【TBD: net return, deployed share, capped share】.
- By calendar year (2023, 2024, 2025, 2026): 【TBD: `yearly.csv`】.

### 5. Limitations

1. **P3 has only about 95 sessions** (June to October 2026) in a single market regime; passing or failing cannot be extrapolated into a long-run conclusion.
2. **P2 has been seen**: the orthogonalized design of S1 / S2 was decided after seeing P2, so P1 ∪ P2 figures are informational; S-A's parameters are selected on P1 and both P2 and P3 are out of sample for it.
3. **Dataset revision v1**: `adjusted_close` carries dividends only, not splits, capital reductions or par-value changes (109 stocks since 2023 show a single-session adjusted-open jump above 25%); position returns use adjacent sessions' opens and exit on halts, so they are unaffected, but the model scores around the events are. The v2 correction has trained no model yet.
4. The chip features cover only the free official items under the conservative availability rule (before the T+1 open); broker-branch and shareholder-dispersion data were not bought.
5. The short book's borrow-failure, recall and fee tiers are parameter assumptions (five random replicates averaged), not actual execution records.
6. The out-of-sample window contains the April 2025 crash but not the 2022 bear market.

### 6. Reproduction

```bash
cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/<F0-launch>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/<F2-launch>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

```bash
cd backtest && uv run python -m twbacktest.strategy_decision --suite ../artifacts/diagnostics/backtest/<scenario-suite output directory>
```

Output directories: 【TBD】.
