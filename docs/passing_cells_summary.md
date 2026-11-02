# 通過門檻的模型×策略各期摘要（§13.23 第 18 步）

## 繁體中文

資料來源：21 格情境回測 `artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z`（grid.csv 與 daily.csv）以及策略決策 `strategy-decision-p3b-21cells-20261102T082429Z`。所有數字為 institution 成本檔、TW 合併市場（TWSE 與 TPEX 各 50%）、fractional 資金、v1 正交化分數、主滑價倍數 1.0；與決策報告的數字一致。

### 定義與單位

- 期間：P1 = 2023-01-02 至 2025-07-01、P2 = 2025-07-01 至 2026-07-01、P1+P2 = 兩者合併、P3 = 2026-07-01 至 2026-10-23（樣本外；預測只到 2026-10-01，故實際 64 個開盤到開盤區間）。區間端點為該期第一個與最後一個開盤到開盤區間的起日。
- 累積淨報酬 (%)：該期每日淨報酬複利連乘減一，未年化。
- 年化淨報酬 (%/年)：該期日均淨報酬 × 252；95% 區間為 21 日區塊 bootstrap（1000 次）的年化淨報酬區間。
- 年化毛報酬 (%/年)：扣成本前的日均毛報酬 × 252。
- 大盤：TAIEX 報酬指數與 TPEX 報酬指數各 50% 的日報酬；0050 為含息、已校正 2025-06-18 一拆四的 0050.TW；等權宇宙為可交易宇宙的等權日報酬。
- 超額報酬：年化超額 = 年化淨報酬 − 大盤年化報酬，單位為百分點/年；累積超額 = 累積淨報酬 − 大盤累積報酬，單位為百分點。多空策略（S1、S2）近似市場中性（beta 約 0 至 0.2），對大盤的超額在 P2 這種大多頭期間會明顯為負，這是設計使然，不是訊號失效；純多頭策略（S-A）才適合直接與大盤比較。
- Sharpe（年化）：日均淨報酬 ÷ 日淨報酬標準差 × √252，無風險利率取 0，策略與大盤同一公式。
- 對大盤 beta：該期策略日淨報酬對大盤日報酬的最小平方迴歸斜率（無單位）。
- 最大回撤 (%)：該期淨值由高點回落的最大幅度。成本拖累 (%/年)：年化毛報酬與年化淨報酬之差。日均換手 (%/日)：每日成交金額占組合淨值比例的平均。勝率 (%)：該期淨報酬為正的交易日比例。
- 入選與通過：七格在 P1、P2、P1+P2 淨 Sharpe 皆 ≥ 1.5 且 P1+P2 年化淨報酬 95% 區間下界 > 0；其中六格通過 H-M（P3 淨報酬 > 0 且 P3 最大回撤不深於 P1+P2 最大回撤的 2 倍）。F2 × S-A 入選但 P3 淨報酬為負，未通過，列出供對照。

### 表 1：基準（大盤）各期

| 期間 | 區間（開盤到開盤） | 交易日 (日) | 大盤累積報酬 (%) | 大盤年化報酬 (%/年) | 大盤 Sharpe（年化） | 大盤最大回撤 (%) | 0050 年化報酬 (%/年) | 0050 Sharpe（年化） | 等權宇宙 Sharpe（年化） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 2023-01-04 → 2025-06-30 | 582 | +72.2% | +25.7% | +1.24 | -27.7% | +30.0% | +1.42 | +0.91 |
| P2 | 2025-07-01 → 2026-06-30 | 229 | +116.8% | +88.0% | +3.78 | -10.2% | +95.9% | +3.80 | +2.42 |
| P1+P2 | 2023-01-04 → 2026-06-30 | 811 | +273.2% | +43.3% | +2.01 | -27.7% | +48.6% | +2.17 | +1.38 |
| P3 | 2026-07-01 → 2026-10-01 | 64 | +4.0% | +19.9% | +0.66 | -14.6% | +14.8% | +0.53 | +0.17 |

### 表 2：報酬與超額報酬

| 模型 | 策略 | 期間 | 累積淨報酬 (%) | 年化淨報酬 (%/年) | 年化淨報酬 95% 區間 (%/年) | 年化毛報酬 (%/年) | 大盤年化報酬 (%/年) | 年化超額報酬（淨−大盤，百分點/年） | 累積超額報酬（淨−大盤，百分點） |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P1 | +75.2% | +24.9% | [+12.1%, +38.5%] | +35.2% | +25.7% | -0.7 | +3.0 |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P2 | +46.7% | +43.4% | [+14.2%, +67.7%] | +52.9% | +88.0% | -44.6 | -70.0 |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P1+P2 | +157.0% | +30.2% | [+18.5%, +43.0%] | +40.2% | +43.3% | -13.1 | -116.2 |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P3 | +4.9% | +19.6% | [-38.2%, +87.8%] | +33.8% | +19.9% | -0.3 | +0.9 |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P1 | +58.4% | +20.5% | [+7.0%, +34.6%] | +30.7% | +25.7% | -5.2 | -13.8 |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P2 | +40.1% | +38.2% | [+9.2%, +65.9%] | +47.7% | +88.0% | -49.7 | -76.6 |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P1+P2 | +122.0% | +25.5% | [+13.4%, +39.2%] | +35.5% | +43.3% | -17.8 | -151.3 |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P3 | +3.4% | +14.2% | [-42.1%, +81.4%] | +28.5% | +19.9% | -5.7 | -0.6 |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P1 | +60.4% | +21.3% | [+8.2%, +35.8%] | +33.1% | +25.7% | -4.4 | -11.8 |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P2 | +33.0% | +32.5% | [+7.2%, +60.1%] | +42.0% | +88.0% | -55.5 | -83.8 |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P1+P2 | +113.3% | +24.5% | [+11.1%, +37.8%] | +35.6% | +43.3% | -18.8 | -159.9 |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P3 | +3.8% | +15.7% | [-38.2%, +77.9%] | +28.4% | +19.9% | -4.2 | -0.2 |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P1 | +62.3% | +21.8% | [+8.2%, +35.6%] | +33.5% | +25.7% | -3.9 | -9.9 |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P2 | +31.2% | +30.9% | [+2.0%, +62.7%] | +40.4% | +88.0% | -57.0 | -85.6 |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P1+P2 | +112.9% | +24.4% | [+11.2%, +39.0%] | +35.4% | +43.3% | -18.9 | -160.3 |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P3 | +3.5% | +14.5% | [-49.0%, +91.8%] | +27.1% | +19.9% | -5.5 | -0.5 |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P1 | +57.9% | +20.7% | [+6.8%, +35.1%] | +31.8% | +25.7% | -5.0 | -14.3 |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P2 | +26.1% | +26.5% | [-3.7%, +57.6%] | +37.2% | +88.0% | -61.5 | -90.7 |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P1+P2 | +99.1% | +22.3% | [+8.2%, +36.6%] | +33.4% | +43.3% | -20.9 | -174.1 |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P3 | +1.7% | +7.6% | [-33.4%, +52.6%] | +23.1% | +19.9% | -12.3 | -2.3 |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P1 | +85.6% | +28.5% | [+3.1%, +51.5%] | +32.8% | +25.7% | +2.8 | +13.4 |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P2 | +108.6% | +84.2% | [+42.8%, +131.6%] | +88.8% | +88.0% | -3.8 | -8.2 |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P1+P2 | +287.1% | +44.2% | [+22.4%, +67.7%] | +48.6% | +43.3% | +0.9 | +13.9 |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P3 | +0.1% | +4.8% | [-57.6%, +63.2%] | +10.5% | +19.9% | -15.1 | -3.9 |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P1 | +89.3% | +29.5% | [+4.8%, +52.4%] | +34.5% | +25.7% | +3.8 | +17.1 |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P2 | +169.9% | +114.0% | [+61.5%, +175.6%] | +120.2% | +88.0% | +26.0 | +53.1 |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P1+P2 | +410.8% | +53.4% | [+28.1%, +82.9%] | +58.7% | +43.3% | +10.1 | +137.6 |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P3 | -5.3% | -13.4% | [-120.3%, +75.8%] | -5.5% | +19.9% | -33.3 | -9.3 |

### 表 3：風險、Sharpe 與 beta

| 模型 | 策略 | 期間 | 策略 Sharpe（年化） | 大盤 Sharpe（年化） | 對大盤 beta（日報酬迴歸） | 最大回撤 (%) | 成本拖累 (%/年) | 日均換手 (%/日) | 勝率 (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P1 | +2.17 | +1.24 | +0.18 | -8.1% | +10.3% | 2.79% | 55.0% |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P2 | +2.84 | +3.78 | +0.10 | -6.7% | +9.5% | 2.79% | 55.0% |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P1+P2 | +2.38 | +2.01 | +0.15 | -8.1% | +10.1% | 2.79% | 55.0% |
| F2 籌碼特徵 LightGBM | S2 多空＋台指期避險 | P3 | +1.48 | +0.66 | +0.04 | -8.0% | +14.3% | 2.79% | 54.7% |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P1 | +1.90 | +1.24 | -0.02 | -11.0% | +10.2% | 2.79% | 54.0% |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P2 | +2.58 | +3.78 | +0.07 | -6.8% | +9.4% | 2.79% | 54.6% |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P1+P2 | +2.11 | +2.01 | +0.02 | -11.0% | +10.0% | 2.79% | 54.1% |
| F2 籌碼特徵 LightGBM | S1 多日多空 | P3 | +1.06 | +0.66 | +0.02 | -8.3% | +14.2% | 2.79% | 53.1% |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P1 | +1.62 | +1.24 | +0.22 | -8.2% | +11.8% | 3.30% | 52.7% |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P2 | +2.21 | +3.78 | +0.07 | -8.0% | +9.5% | 3.30% | 55.5% |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P1+P2 | +1.80 | +2.01 | +0.17 | -8.2% | +11.2% | 3.30% | 53.5% |
| D0 跨股 Transformer | S2 多空＋台指期避險 | P3 | +1.16 | +0.66 | +0.11 | -7.8% | +12.6% | 3.30% | 50.0% |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P1 | +1.69 | +1.24 | +0.20 | -9.3% | +11.7% | 3.22% | 53.4% |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P2 | +2.10 | +3.78 | +0.10 | -6.2% | +9.4% | 3.22% | 52.8% |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P1+P2 | +1.81 | +2.01 | +0.17 | -9.3% | +11.0% | 3.22% | 53.3% |
| D1 Chronos-2 編碼器 | S2 多空＋台指期避險 | P3 | +1.03 | +0.66 | +0.02 | -10.0% | +12.6% | 3.22% | 54.7% |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P1 | +1.53 | +1.24 | +0.24 | -8.6% | +11.1% | 3.20% | 51.5% |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P2 | +1.91 | +3.78 | +0.14 | -7.4% | +10.8% | 3.20% | 57.6% |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P1+P2 | +1.64 | +2.01 | +0.20 | -8.6% | +11.0% | 3.20% | 53.3% |
| F0 基線 LightGBM（殘差標籤） | S2 多空＋台指期避險 | P3 | +0.60 | +0.66 | +0.05 | -7.0% | +15.5% | 3.20% | 54.7% |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P1 | +1.56 | +1.24 | +0.42 | -23.9% | +4.4% | 3.69% | 59.3% |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P2 | +3.36 | +3.78 | +0.51 | -9.5% | +4.6% | 3.69% | 64.2% |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P1+P2 | +2.16 | +2.01 | +0.45 | -23.9% | +4.4% | 3.69% | 60.7% |
| D1 Kronos 編碼器 | S-A 週頻純多頭 | P3 | +0.16 | +0.66 | +0.57 | -16.1% | +5.7% | 3.69% | 46.9% |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P1 | +1.54 | +1.24 | +0.44 | -25.1% | +5.0% | 4.42% | 57.6% |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P2 | +3.79 | +3.78 | +0.62 | -11.6% | +6.2% | 4.42% | 65.1% |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P1+P2 | +2.33 | +2.01 | +0.51 | -25.1% | +5.3% | 4.42% | 59.7% |
| F2 籌碼特徵 LightGBM （H-M 未通過） | S-A 週頻純多頭 | P3 | -0.33 | +0.66 | +0.75 | -24.6% | +7.8% | 4.42% | 45.3% |

## English

Sources: the 21-cell scenario suite `artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z` (grid.csv and daily.csv) and the strategy decision `strategy-decision-p3b-21cells-20261102T082429Z`. Every number is the institution cost profile, the combined TW market (TWSE and TPEX at 50% each), fractional capital, the v1 orthogonalized score and the primary slippage multiplier 1.0; they agree with the decision report.

### Definitions and units

- Periods: P1 = 2023-01-02 to 2025-07-01, P2 = 2025-07-01 to 2026-07-01, P1+P2 = both, P3 = 2026-07-01 to 2026-10-23 (out of sample; predictions end on 2026-10-01, so 64 open-to-open intervals). Window endpoints are the start dates of the first and last open-to-open interval in the period.
- Cumulative net return (%): the period's daily net returns compounded, minus one; not annualized.
- Annualized net return (%/yr): mean daily net return × 252; the 95% interval is the 21-day block bootstrap (1000 replicates) of the annualized net return.
- Annualized gross return (%/yr): mean daily gross return before costs × 252.
- Index: 50% TAIEX total-return index and 50% TPEX total-return index daily returns; 0050 is 0050.TW with dividends and the 2025-06-18 1:4 split corrected; the equal-weight universe is the tradeable universe's equal-weight daily return.
- Excess return: annualized excess = annualized net return − index annualized return, in percentage points per year; cumulative excess = cumulative net return − index cumulative return, in percentage points. The long-short strategies (S1, S2) are close to market neutral (beta about 0 to 0.2), so their excess over the index is clearly negative in a strong bull period such as P2 by construction, not because the signal failed; only the long-only strategy (S-A) is directly comparable with the index.
- Sharpe (annualized): mean daily net return ÷ standard deviation of daily net returns × √252, risk-free rate 0, the same formula for strategy and index.
- Beta to index: least-squares slope of the strategy's daily net return on the index's daily return in that period (unitless).
- Max drawdown (%): the largest peak-to-trough fall of the period's net equity. Cost drag (%/yr): annualized gross minus annualized net return. Average daily turnover (%/day): mean daily traded value as a share of portfolio value. Hit rate (%): share of trading days with a positive net return.
- Shortlist and pass: seven cells have net Sharpe ≥ 1.5 in each of P1, P2 and P1+P2 and a P1+P2 annualized-net lower bound above 0; six of them pass H-M (P3 net return > 0 and P3 max drawdown within 2× the P1+P2 max drawdown). F2 × S-A was shortlisted but its P3 net return is negative, so it fails and is listed for comparison only.

### Table 1: benchmark by period

| Period | Window (open to open) | Trading days | Index cumulative return (%) | Index annualized return (%/yr) | Index Sharpe (annualized) | Index max drawdown (%) | 0050 annualized return (%/yr) | 0050 Sharpe (annualized) | Equal-weight universe Sharpe (annualized) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 2023-01-04 → 2025-06-30 | 582 | +72.2% | +25.7% | +1.24 | -27.7% | +30.0% | +1.42 | +0.91 |
| P2 | 2025-07-01 → 2026-06-30 | 229 | +116.8% | +88.0% | +3.78 | -10.2% | +95.9% | +3.80 | +2.42 |
| P1+P2 | 2023-01-04 → 2026-06-30 | 811 | +273.2% | +43.3% | +2.01 | -27.7% | +48.6% | +2.17 | +1.38 |
| P3 | 2026-07-01 → 2026-10-01 | 64 | +4.0% | +19.9% | +0.66 | -14.6% | +14.8% | +0.53 | +0.17 |

### Table 2: returns and excess returns

| Model | Strategy | Period | Cumulative net return (%) | Annualized net return (%/yr) | 95% interval of annualized net (%/yr) | Annualized gross return (%/yr) | Index annualized return (%/yr) | Annualized excess (net − index, pp/yr) | Cumulative excess (net − index, pp) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P1 | +75.2% | +24.9% | [+12.1%, +38.5%] | +35.2% | +25.7% | -0.7 | +3.0 |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P2 | +46.7% | +43.4% | [+14.2%, +67.7%] | +52.9% | +88.0% | -44.6 | -70.0 |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P1+P2 | +157.0% | +30.2% | [+18.5%, +43.0%] | +40.2% | +43.3% | -13.1 | -116.2 |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P3 | +4.9% | +19.6% | [-38.2%, +87.8%] | +33.8% | +19.9% | -0.3 | +0.9 |
| F2 chip-feature LightGBM | S1 multi-day long-short | P1 | +58.4% | +20.5% | [+7.0%, +34.6%] | +30.7% | +25.7% | -5.2 | -13.8 |
| F2 chip-feature LightGBM | S1 multi-day long-short | P2 | +40.1% | +38.2% | [+9.2%, +65.9%] | +47.7% | +88.0% | -49.7 | -76.6 |
| F2 chip-feature LightGBM | S1 multi-day long-short | P1+P2 | +122.0% | +25.5% | [+13.4%, +39.2%] | +35.5% | +43.3% | -17.8 | -151.3 |
| F2 chip-feature LightGBM | S1 multi-day long-short | P3 | +3.4% | +14.2% | [-42.1%, +81.4%] | +28.5% | +19.9% | -5.7 | -0.6 |
| D0 cross-stock transformer | S2 long-short + TX hedge | P1 | +60.4% | +21.3% | [+8.2%, +35.8%] | +33.1% | +25.7% | -4.4 | -11.8 |
| D0 cross-stock transformer | S2 long-short + TX hedge | P2 | +33.0% | +32.5% | [+7.2%, +60.1%] | +42.0% | +88.0% | -55.5 | -83.8 |
| D0 cross-stock transformer | S2 long-short + TX hedge | P1+P2 | +113.3% | +24.5% | [+11.1%, +37.8%] | +35.6% | +43.3% | -18.8 | -159.9 |
| D0 cross-stock transformer | S2 long-short + TX hedge | P3 | +3.8% | +15.7% | [-38.2%, +77.9%] | +28.4% | +19.9% | -4.2 | -0.2 |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P1 | +62.3% | +21.8% | [+8.2%, +35.6%] | +33.5% | +25.7% | -3.9 | -9.9 |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P2 | +31.2% | +30.9% | [+2.0%, +62.7%] | +40.4% | +88.0% | -57.0 | -85.6 |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P1+P2 | +112.9% | +24.4% | [+11.2%, +39.0%] | +35.4% | +43.3% | -18.9 | -160.3 |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P3 | +3.5% | +14.5% | [-49.0%, +91.8%] | +27.1% | +19.9% | -5.5 | -0.5 |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P1 | +57.9% | +20.7% | [+6.8%, +35.1%] | +31.8% | +25.7% | -5.0 | -14.3 |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P2 | +26.1% | +26.5% | [-3.7%, +57.6%] | +37.2% | +88.0% | -61.5 | -90.7 |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P1+P2 | +99.1% | +22.3% | [+8.2%, +36.6%] | +33.4% | +43.3% | -20.9 | -174.1 |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P3 | +1.7% | +7.6% | [-33.4%, +52.6%] | +23.1% | +19.9% | -12.3 | -2.3 |
| D1 Kronos encoder | S-A weekly long-only | P1 | +85.6% | +28.5% | [+3.1%, +51.5%] | +32.8% | +25.7% | +2.8 | +13.4 |
| D1 Kronos encoder | S-A weekly long-only | P2 | +108.6% | +84.2% | [+42.8%, +131.6%] | +88.8% | +88.0% | -3.8 | -8.2 |
| D1 Kronos encoder | S-A weekly long-only | P1+P2 | +287.1% | +44.2% | [+22.4%, +67.7%] | +48.6% | +43.3% | +0.9 | +13.9 |
| D1 Kronos encoder | S-A weekly long-only | P3 | +0.1% | +4.8% | [-57.6%, +63.2%] | +10.5% | +19.9% | -15.1 | -3.9 |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P1 | +89.3% | +29.5% | [+4.8%, +52.4%] | +34.5% | +25.7% | +3.8 | +17.1 |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P2 | +169.9% | +114.0% | [+61.5%, +175.6%] | +120.2% | +88.0% | +26.0 | +53.1 |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P1+P2 | +410.8% | +53.4% | [+28.1%, +82.9%] | +58.7% | +43.3% | +10.1 | +137.6 |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P3 | -5.3% | -13.4% | [-120.3%, +75.8%] | -5.5% | +19.9% | -33.3 | -9.3 |

### Table 3: risk, Sharpe and beta

| Model | Strategy | Period | Strategy Sharpe (annualized) | Index Sharpe (annualized) | Beta to index (daily regression) | Max drawdown (%) | Cost drag (%/yr) | Average daily turnover (%/day) | Hit rate (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P1 | +2.17 | +1.24 | +0.18 | -8.1% | +10.3% | 2.79% | 55.0% |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P2 | +2.84 | +3.78 | +0.10 | -6.7% | +9.5% | 2.79% | 55.0% |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P1+P2 | +2.38 | +2.01 | +0.15 | -8.1% | +10.1% | 2.79% | 55.0% |
| F2 chip-feature LightGBM | S2 long-short + TX hedge | P3 | +1.48 | +0.66 | +0.04 | -8.0% | +14.3% | 2.79% | 54.7% |
| F2 chip-feature LightGBM | S1 multi-day long-short | P1 | +1.90 | +1.24 | -0.02 | -11.0% | +10.2% | 2.79% | 54.0% |
| F2 chip-feature LightGBM | S1 multi-day long-short | P2 | +2.58 | +3.78 | +0.07 | -6.8% | +9.4% | 2.79% | 54.6% |
| F2 chip-feature LightGBM | S1 multi-day long-short | P1+P2 | +2.11 | +2.01 | +0.02 | -11.0% | +10.0% | 2.79% | 54.1% |
| F2 chip-feature LightGBM | S1 multi-day long-short | P3 | +1.06 | +0.66 | +0.02 | -8.3% | +14.2% | 2.79% | 53.1% |
| D0 cross-stock transformer | S2 long-short + TX hedge | P1 | +1.62 | +1.24 | +0.22 | -8.2% | +11.8% | 3.30% | 52.7% |
| D0 cross-stock transformer | S2 long-short + TX hedge | P2 | +2.21 | +3.78 | +0.07 | -8.0% | +9.5% | 3.30% | 55.5% |
| D0 cross-stock transformer | S2 long-short + TX hedge | P1+P2 | +1.80 | +2.01 | +0.17 | -8.2% | +11.2% | 3.30% | 53.5% |
| D0 cross-stock transformer | S2 long-short + TX hedge | P3 | +1.16 | +0.66 | +0.11 | -7.8% | +12.6% | 3.30% | 50.0% |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P1 | +1.69 | +1.24 | +0.20 | -9.3% | +11.7% | 3.22% | 53.4% |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P2 | +2.10 | +3.78 | +0.10 | -6.2% | +9.4% | 3.22% | 52.8% |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P1+P2 | +1.81 | +2.01 | +0.17 | -9.3% | +11.0% | 3.22% | 53.3% |
| D1 Chronos-2 encoder | S2 long-short + TX hedge | P3 | +1.03 | +0.66 | +0.02 | -10.0% | +12.6% | 3.22% | 54.7% |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P1 | +1.53 | +1.24 | +0.24 | -8.6% | +11.1% | 3.20% | 51.5% |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P2 | +1.91 | +3.78 | +0.14 | -7.4% | +10.8% | 3.20% | 57.6% |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P1+P2 | +1.64 | +2.01 | +0.20 | -8.6% | +11.0% | 3.20% | 53.3% |
| F0 baseline LightGBM (residual label) | S2 long-short + TX hedge | P3 | +0.60 | +0.66 | +0.05 | -7.0% | +15.5% | 3.20% | 54.7% |
| D1 Kronos encoder | S-A weekly long-only | P1 | +1.56 | +1.24 | +0.42 | -23.9% | +4.4% | 3.69% | 59.3% |
| D1 Kronos encoder | S-A weekly long-only | P2 | +3.36 | +3.78 | +0.51 | -9.5% | +4.6% | 3.69% | 64.2% |
| D1 Kronos encoder | S-A weekly long-only | P1+P2 | +2.16 | +2.01 | +0.45 | -23.9% | +4.4% | 3.69% | 60.7% |
| D1 Kronos encoder | S-A weekly long-only | P3 | +0.16 | +0.66 | +0.57 | -16.1% | +5.7% | 3.69% | 46.9% |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P1 | +1.54 | +1.24 | +0.44 | -25.1% | +5.0% | 4.42% | 57.6% |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P2 | +3.79 | +3.78 | +0.62 | -11.6% | +6.2% | 4.42% | 65.1% |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P1+P2 | +2.33 | +2.01 | +0.51 | -25.1% | +5.3% | 4.42% | 59.7% |
| F2 chip-feature LightGBM (fails H-M) | S-A weekly long-only | P3 | -0.33 | +0.66 | +0.75 | -24.6% | +7.8% | 4.42% | 45.3% |
