# 與持有方式對齊的評估結果 / Holding-Aligned Evaluation Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.6](experiment_protocol.md) 的結果：以「次一交易日開盤買進、h 日後開盤賣出」的開盤到開盤報酬，重評現有 13 個台股訊號。本節只作描述，不設門檻；validation 用於描述，test 已在先前分析中多次使用，只作探索性判讀。

結論：

1. **多數訊號的前十分位在隔夜大幅落後。** 包括 loss 選出的 Kronos checkpoint、GRU、GBDT 與方向模型：
   - 10 日窗口中，前十分位相對等權的隔夜部分約 −80 到 −330 bps（GBDT 在 test 例外），抵銷了交易時段的領先。
   - test 的前十分位超額大多為負。
2. **以 Rank IC-IR 重選的 checkpoint 與 R23 不同。**
   - R22-IR 在上市 10 日的隔夜部分只有 −10.7 bps（validation）與 +18.6 bps（test）。
   - 上市前十分位超額在兩個期間都為正：validation +69.6 bps（95% 區間下界 +8.6），test +103.2 bps（下界 +57.3）。
   - 固定組合（K = 20、b = 2、主要成本情境）上市年化淨超額：validation +11.3% [−17.3%, +43.7%]，test +42.8% [+10.7%, +74.6%]。
3. **上櫃幾乎全部為負**，R23 是唯一例外。
4. **整體 Rank IC 與前十分位不一致。** 例如上市 10 日，R22 的 validation Rank IC 為 0.051，高於 R22-IR 的 0.036，但 R22 的前十分位超額為 −37.2 bps，R22-IR 為 +69.6 bps。
5. **以上都是描述性結果。** R22-IR 本身就是在 validation 上以 Rank IC-IR 選出的，validation 數值偏樂觀；13 個訊號 × 2 市場 × 3 天期有多重比較；test 已多次使用。是否可用，需要以未見資料確認。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 母體與分割 | 與 §13.4 相同；validation 112 日、test 103 日 |
| 記帳恆等式 | validation 96,166 組、test 93,402 組相鄰資料列全部成立（股票與指數） |
| 標籤 | `log(open(d+1+h) / open(d+1))`，由 `abs[d, h+1] − abs[d+h, 1]` 推得；指數同法；有標籤的比例：5 日約 93%–94%，10 日約 87%–89% |
| 天期 | 主要 5、10 日；輔助 3 日 |
| 十分位 | 每日前 10%（四捨五入，至少 1 檔）的開盤到開盤簡單報酬，減同日同市場有標籤股票的等權平均 |
| 隔夜拆解 | 前十分位超額 log 報酬拆成 h 個交易時段（各日開盤到收盤）與隔夜部分 |
| 固定組合 | §13.5 的引擎；s = h、K = 20、b = 2、折扣 0.6、滑價 1 倍、只做多 |
| 區間 | circular moving-block bootstrap，1,000 次、seed 11；h 日序列區塊 max(5, 2h)，組合每日序列區塊 10 |

### 3. 結果

#### 3.1 上市 10 日：前十分位超額與隔夜拆解（bps，每個 10 日窗口）

| 訊號 | validation 前十分位 | test 前十分位 | validation 交易時段 / 隔夜 | test 交易時段 / 隔夜 |
| --- | ---: | ---: | --- | --- |
| R22-IR | +69.6 | +103.2 | +78.5 / −10.7 | +92.5 / +18.6 |
| ENS-IR | +56.1 | +32.6 | +119.6 / −58.2 | +126.4 / −75.7 |
| R21-IR | +61.4 | −42.9 | +122.7 / −54.1 | +116.4 / −134.7 |
| R23 | +35.7 | +86.4 | +113.3 / −73.1 | +143.4 / −47.5 |
| R24 datemix | −4.3 | −137.6 | +86.8 / −79.1 | +214.1 / −307.2 |
| R22 | −37.2 | −113.7 | +198.0 / −213.8 | +239.0 / −309.1 |
| R21 | −84.8 | −191.2 | +159.0 / −220.9 | +136.1 / −283.6 |
| ENS-LOSS | −40.3 | −111.4 | +198.4 / −216.8 | +226.7 / −295.6 |
| GRU | −39.7 | −133.6 | +214.4 / −230.3 | +232.2 / −318.9 |
| GBDT | −46.4 | +52.1 | +134.7 / −162.3 | +34.6 / +15.4 |
| 方向集成 | −13.2 | −51.5 | +161.1 / −157.3 | +210.6 / −229.8 |
| GRU 方向 | −49.4 | −198.2 | +192.5 / −217.8 | +184.3 / −333.7 |
| GBDT 方向 | −57.0 | −159.1 | +154.4 / −190.6 | +102.6 / −222.6 |

隔夜拖累小的四個訊號（R22-IR、ENS-IR、R21-IR、R23），都使用訓練較後期的 checkpoint（步數 018399–022079）；loss 選出的 R21、R22 與 R24 使用較早的 checkpoint（003680–011040）。這是觀察，原因尚未檢驗：可能來自訓練長度，也可能來自 Rank IC-IR 選點本身。

#### 3.2 上櫃

- 10 日 test 的前十分位超額只有 R23（+63.5 bps）與 GBDT（+43.3 bps）為正，其餘介於 −29.7 與 −300.8 bps。
- R22-IR 在 validation 為 +116.8 bps（下界 +42.9），在 test 轉為 −49.3 bps。

#### 3.3 固定組合（K = 20、b = 2、主要成本情境，年化淨超額 [95% 區間]）

| 市場 | h | 訊號 | validation | test | test 每日換手 | test 平均持有（交易日） |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 上市 | 10 | R22-IR | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | 6.9% | 12.7 |
| 上市 | 5 | R22-IR | +7.2% [−18.5%, +36.5%] | +30.1% [+2.6%, +58.3%] | 6.4% | 13.7 |
| 上市 | 10 | ENS-IR | +13.1% [−8.3%, +34.3%] | +0.9% [−28.3%, +30.7%] | 4.6% | 18.0 |
| 上市 | 10 | R23 | −3.0% [−26.1%, +19.8%] | +11.7% [−25.2%, +46.0%] | 4.0% | 20.4 |
| 上市 | 10 | R22 | −18.1% [−44.4%, +7.6%] | −29.3% [−67.1%, +12.4%] | 1.8% | 36.4 |
| 上市 | 10 | GRU | −10.2% [−39.8%, +19.4%] | −35.2% [−75.5%, +7.4%] | 5.7% | 15.0 |
| 合併 | 10 | R22-IR | +10.1% [−19.3%, +41.0%] | +38.1% [−1.6%, +83.2%] | 8.0% | 11.2 |
| 上櫃 | 10 | R23 | +3.0% [−32.4%, +42.5%] | +16.4% [−24.4%, +56.1%] | 5.2% | 16.3 |

#### 3.4 可留意清單（主要天期、validation 前十分位下界大於 0）

| 市場 | h | 訊號 | validation 前十分位 [下界] | test 前十分位 [下界] |
| --- | ---: | --- | --- | --- |
| 上市 | 10 | R22-IR | +69.6 [+8.6] | +103.2 [+57.3] |
| 上市 | 10 | ENS-IR | +56.1 [+9.3] | +32.6 [−30.2] |
| 上市 | 5 | ENS-IR | +31.4 [+2.0] | +25.1 [−30.4] |
| 上市 | 10 | R21-IR | +61.4 [+7.7] | −42.9 [−151.2] |
| 上櫃 | 10 | R22-IR | +116.8 [+42.9] | −49.3 [−166.9] |

5 個項目中，只有上市 10 日的 R22-IR 在 test 也維持下界大於 0。ENS-IR 包含 R22-IR，兩者並不獨立。

### 4. 意涵

- 「選哪個 checkpoint」對隔夜行為的影響，與「用哪種標籤」一樣值得檢驗：同一個 R22 run，loss 選出的 checkpoint 隔夜拖累 −213.8 bps，Rank IC-IR 選出的只有 −10.7 bps。以與持有方式對齊的 validation 指標選 checkpoint，可能是比重訓更便宜的改善方向（假說）。
- 上市 10 日的 R22-IR 是目前唯一在兩個期間都為正的候選，但它是事後找到的。要把它當成可用訊號，必須先事先登記，再用 2026-06-01 之後的未見資料檢驗。
- B（GBDT 改用開盤到開盤標籤訓練）仍可回答「標籤本身是否有幫助」，但 A 的結果顯示 checkpoint 選擇可能是更直接的槓桿。

### 5. 限制

- 13 個訊號 × 2 市場 × 3 天期的多重比較；h 日標籤互相重疊。
- R22-IR、R21-IR、ENS-IR 的 checkpoint 是在 validation 上選的，validation 數值偏樂觀。
- test 已在先前分析中多次使用；validation 與 test 各約 5 個月。
- 未模擬成交量與流動性、每筆最低手續費、整股、股利稅；存活者偏誤同 §13.4。
- 分析在本機沙箱內以單一 worker 執行；多程序的一致性測試在沙箱中被略過。

### 6. 重現方式

```bash
cd backtest
uv run python -m twbacktest.holding_eval
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `3949fb6` |
| 程式 commit | `7e7d925` |
| 設定檔 | `backtest/config/holding_eval.json` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/holding-eval-20261019T064548Z-cb9a4149/`（`report.json`、`ranking.csv`、`portfolio.csv`、`summary.md`） |

## English

### 1. Purpose and conclusions

This report records the results of [experiment protocol §13.6](experiment_protocol.md): the 13 existing Taiwan signals re-evaluated on open-to-open returns, buying at the next session's open and selling at the open h sessions later. The section is descriptive only, with no gate; validation is descriptive, and test, consulted repeatedly in earlier analyses, is exploratory only.

Conclusions:

1. **For most signals the top decile lags heavily overnight.** This covers the loss-selected Kronos checkpoints, GRU, GBDT and the direction models:
   - Within 10-session windows, the top decile's overnight part relative to equal weight is about −80 to −330 bps (GBDT on test excepted), cancelling its session lead.
   - Most test top-decile excesses are negative.
2. **The checkpoints re-selected by Rank IC-IR and R23 behave differently.**
   - R22-IR's 10-session TWSE overnight part is only −10.7 bps (validation) and +18.6 bps (test).
   - Its TWSE top-decile excess is positive in both periods: validation +69.6 bps (95% lower bound +8.6) and test +103.2 bps (lower bound +57.3).
   - The fixed portfolio (K = 20, b = 2, primary costs) earns TWSE annual net excess of +11.3% [−17.3%, +43.7%] on validation and +42.8% [+10.7%, +74.6%] on test.
3. **TPEX is negative almost everywhere**, with R23 the only exception.
4. **Overall Rank IC and the top decile disagree.** For example, at 10 sessions on TWSE, R22's validation Rank IC of 0.051 is above R22-IR's 0.036, yet R22's top-decile excess is −37.2 bps against R22-IR's +69.6 bps.
5. **All of this is descriptive.** R22-IR itself was selected on validation by Rank IC-IR, so its validation values are optimistic; 13 signals × 2 markets × 3 horizons create multiple comparisons; and test has been consulted repeatedly. Usability must be confirmed on unseen data.

### 2. Data and method

| Item | Content |
| --- | --- |
| Population and splits | As in §13.4; 112 validation and 103 test dates |
| Accounting identity | All 96,166 adjacent pairs in validation and 93,402 in test satisfy it (stocks and index) |
| Labels | `log(open(d+1+h) / open(d+1))`, derived as `abs[d, h+1] − abs[d+h, 1]`; the index likewise; label coverage about 93%–94% at 5 sessions and 87%–89% at 10 |
| Horizons | 5 and 10 sessions primary; 3 auxiliary |
| Deciles | The daily top 10% (rounded, at least 1 stock) open-to-open simple return minus the same-date, same-market equal weight of stocks with labels |
| Overnight split | The top decile's excess log return split into the h sessions (each day's open to close) and the overnight part |
| Fixed portfolio | The §13.5 engine; s = h, K = 20, b = 2, discount 0.6, slippage ×1, long-only |
| Intervals | Circular moving-block bootstrap, 1,000 replicates, seed 11; block max(5, 2h) for h-session series and 10 for portfolio daily series |

### 3. Results

#### 3.1 TWSE, 10 sessions: top-decile excess and overnight split (bps per 10-session window)

| Signal | Validation top decile | Test top decile | Validation session / overnight | Test session / overnight |
| --- | ---: | ---: | --- | --- |
| R22-IR | +69.6 | +103.2 | +78.5 / −10.7 | +92.5 / +18.6 |
| ENS-IR | +56.1 | +32.6 | +119.6 / −58.2 | +126.4 / −75.7 |
| R21-IR | +61.4 | −42.9 | +122.7 / −54.1 | +116.4 / −134.7 |
| R23 | +35.7 | +86.4 | +113.3 / −73.1 | +143.4 / −47.5 |
| R24 datemix | −4.3 | −137.6 | +86.8 / −79.1 | +214.1 / −307.2 |
| R22 | −37.2 | −113.7 | +198.0 / −213.8 | +239.0 / −309.1 |
| R21 | −84.8 | −191.2 | +159.0 / −220.9 | +136.1 / −283.6 |
| ENS-LOSS | −40.3 | −111.4 | +198.4 / −216.8 | +226.7 / −295.6 |
| GRU | −39.7 | −133.6 | +214.4 / −230.3 | +232.2 / −318.9 |
| GBDT | −46.4 | +52.1 | +134.7 / −162.3 | +34.6 / +15.4 |
| Direction ensemble | −13.2 | −51.5 | +161.1 / −157.3 | +210.6 / −229.8 |
| GRU direction | −49.4 | −198.2 | +192.5 / −217.8 | +184.3 / −333.7 |
| GBDT direction | −57.0 | −159.1 | +154.4 / −190.6 | +102.6 / −222.6 |

The four signals with little overnight drag (R22-IR, ENS-IR, R21-IR, R23) all use late checkpoints (steps 018399–022079); the loss-selected R21, R22 and R24 use earlier ones (003680–011040). This is an observation whose cause is untested: training length or the Rank IC-IR selection itself.

#### 3.2 TPEX

- At 10 sessions on test, only R23 (+63.5 bps) and GBDT (+43.3 bps) have a positive top-decile excess; the others range from −29.7 to −300.8 bps.
- R22-IR earns +116.8 bps on validation (lower bound +42.9) and turns to −49.3 bps on test.

#### 3.3 Fixed portfolio (K = 20, b = 2, primary costs; annual net excess [95% CI])

| Market | h | Signal | Validation | Test | Test daily turnover | Test mean holding (sessions) |
| --- | ---: | --- | --- | --- | ---: | ---: |
| TWSE | 10 | R22-IR | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | 6.9% | 12.7 |
| TWSE | 5 | R22-IR | +7.2% [−18.5%, +36.5%] | +30.1% [+2.6%, +58.3%] | 6.4% | 13.7 |
| TWSE | 10 | ENS-IR | +13.1% [−8.3%, +34.3%] | +0.9% [−28.3%, +30.7%] | 4.6% | 18.0 |
| TWSE | 10 | R23 | −3.0% [−26.1%, +19.8%] | +11.7% [−25.2%, +46.0%] | 4.0% | 20.4 |
| TWSE | 10 | R22 | −18.1% [−44.4%, +7.6%] | −29.3% [−67.1%, +12.4%] | 1.8% | 36.4 |
| TWSE | 10 | GRU | −10.2% [−39.8%, +19.4%] | −35.2% [−75.5%, +7.4%] | 5.7% | 15.0 |
| TW | 10 | R22-IR | +10.1% [−19.3%, +41.0%] | +38.1% [−1.6%, +83.2%] | 8.0% | 11.2 |
| TPEX | 10 | R23 | +3.0% [−32.4%, +42.5%] | +16.4% [−24.4%, +56.1%] | 5.2% | 16.3 |

#### 3.4 Watch list (primary horizons, validation top-decile lower bound above 0)

| Market | h | Signal | Validation top decile [lower bound] | Test top decile [lower bound] |
| --- | ---: | --- | --- | --- |
| TWSE | 10 | R22-IR | +69.6 [+8.6] | +103.2 [+57.3] |
| TWSE | 10 | ENS-IR | +56.1 [+9.3] | +32.6 [−30.2] |
| TWSE | 5 | ENS-IR | +31.4 [+2.0] | +25.1 [−30.4] |
| TWSE | 10 | R21-IR | +61.4 [+7.7] | −42.9 [−151.2] |
| TPEX | 10 | R22-IR | +116.8 [+42.9] | −49.3 [−166.9] |

Of the 5 entries, only TWSE 10-session R22-IR keeps a lower bound above 0 on test. ENS-IR contains R22-IR, so the two are not independent.

### 4. Implications

- Checkpoint choice deserves testing as much as the label: within the same R22 run, the loss-selected checkpoint drags −213.8 bps overnight and the Rank IC-IR checkpoint only −10.7 bps. Selecting checkpoints with a holding-aligned validation metric may be a cheaper improvement than retraining (a hypothesis).
- TWSE 10-session R22-IR is the only candidate positive in both periods, but it was found after the fact. Treating it as a usable signal requires pre-registration first and a test on unseen data from 2026-06-01 onward.
- B (GBDT trained on open-to-open labels) can still answer whether the label itself helps, but A suggests checkpoint selection may be the more direct lever.

### 5. Limits

- Multiple comparisons over 13 signals × 2 markets × 3 horizons; h-session labels overlap.
- The R22-IR, R21-IR and ENS-IR checkpoints were chosen on validation, so validation values are optimistic.
- Test has been consulted repeatedly; validation and test each cover about five months.
- Volume and liquidity, minimum commissions, board lots and dividend taxes are not simulated; survivorship bias as in §13.4.
- The analysis ran with one worker inside the local sandbox; the multi-process consistency tests were skipped there.

### 6. Reproduction

```bash
cd backtest
uv run python -m twbacktest.holding_eval
```

| Item | Content |
| --- | --- |
| Pre-registration commit | `3949fb6` |
| Code commit | `7e7d925` |
| Configuration | `backtest/config/holding_eval.json` |
| Output (not tracked by Git) | `artifacts/diagnostics/backtest/holding-eval-20261019T064548Z-cb9a4149/` (`report.json`, `ranking.csv`, `portfolio.csv`, `summary.md`) |
