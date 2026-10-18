# 台股回測第一階段與 1b 結果 / Taiwan Backtest Stage 1 and 1b Results

## 中文

### 1. 目的與結論

本報告記錄兩次事先登記的本機回測：

- **第一階段**（[實驗協定 §13.4](experiment_protocol.md)）：固定持有天數。
- **第一階段 1b**（[實驗協定 §13.5](experiment_protocol.md)）：持有緩衝，逐部位記帳。

兩者都在檢驗：已訓練模型的台股截面排序訊號，扣除台股實際交易成本後，能否穩定勝過同市場等權持有。回測不重訓模型、不使用 RunPod，程式在 `backtest/`（獨立的 uv 環境）。

結論：

1. **兩次回測在上市、上櫃與合併市場都未通過事先固定的門檻。**
   - 第一階段：validation 選出的主要設定在 test 的年化淨超額為 −37% 至 −44%。
   - 1b：validation 選出的主要設定在 test 為 −13% 至 −40%；1b 的 test 只作探索性判讀。
2. **前 K 名的優勢只出現在交易時段。** 事後診斷顯示：
   - 從次一交易日開盤到收盤，前 10 名比等權多約 20–31 bps（上市）或 13–29 bps（上櫃）；上市在兩個期間都顯著，上櫃較弱。
   - 從收盤到再下一個開盤，又落後約 5–30 bps（GRU 在 test 例外）。
   - 持有過夜會把優勢大致抵銷；只在交易時段持有則必須當沖，成本高於優勢。
3. **長天期的 Rank IC 無法轉成只做多的前 K 名報酬。** 10 日持有時，test 的預測最差十分位反而大幅勝過最佳十分位。
4. **降低換手只省下成本，換不回優勢。** 持有緩衝把每日換手由 12%–24% 降到 1%–7%、成本拖累降到約四分之一，但開盤到開盤的毛超額本來就所剩無幾。
5. **依事先登記，停止「降低換手」路線，回到模型本身。** 確認性檢驗仍需第三階段的未見資料（2026-06-01 之後）。

### 2. 資料與共同設定

| 項目 | 內容 |
| --- | --- |
| 母體 | P0 評估母體中的上市、上櫃列；列順序以樣本雜湊與各匯出檔核對一致 |
| Validation | 2025-06-02 至 2025-11-10，112 個訊號日；上市 65,517 列、上櫃 31,846 列 |
| Test | 2025-12-01 至 2026-05-11，103 個訊號日；上市 63,313 列、上櫃 31,285 列 |
| 訊號（13 個） | Kronos alpha 8 個（R21、R21-IR、R22、R22-IR、R23、R24 datemix、LOSS 與 IR 三 seed 集成）、GRU／GBDT alpha、3 個方向模型（訊號為 P(漲) − P(跌)） |
| 報酬 | 還原價 log 報酬 `absolute_targets`：第 t+1 日開盤到第 t+h 日收盤 |
| 基準 | 同日、同市場全部股票等權，不計成本 |
| 成本 | 手續費 0.1425% × 折扣；證交稅股票 0.3%、ETF 0.1%；當沖股票 0.15%；滑價每邊上市 10 bps、上櫃 20 bps |
| 主要成本情境 | 折扣 0.6、滑價 1 倍 |

主要成本情境下，一次來回的成本：

| 市場 | 當沖（同日買賣） | 持有過夜 |
| --- | ---: | ---: |
| 上市 | 0.52% | 0.67% |
| 上櫃 | 0.72% | 0.87% |
| 上市，折扣 0.28、無滑價 | 0.23% | 0.38% |

### 3. 第一階段：固定持有天數（§13.4）

規則：第 t 日收盤後排序，第 t+1 日開盤等權買進前 K 檔，第 t+h 日收盤賣出；h ∈ {1, 2, 3, 5, 10}、K ∈ {10, 20, 50}，共 42,120 組設定。

#### 3.1 主要設定與門檻

| 市場 | 訊號 | h | K | validation 年化淨超額 [95% 區間] | test 年化淨超額 [95% 區間] | test IR | 通過 |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| 上市 | R21-IR | 10 | 10 | +19.4% [−3.5%, +49.1%] | −36.8% [−51.0%, −19.1%] | −2.70 | 否 |
| 上櫃 | R22-IR | 10 | 20 | −0.2% [−18.5%, +19.3%] | −42.7% [−63.4%, −23.5%] | −3.43 | 否 |
| 合併 | R21-IR | 10 | 50 | +7.4% [−3.1%, +18.7%] | −43.8% [−66.8%, −24.0%] | −4.48 | 否 |

三個市場都選中 10 日持有（持有越久，成本分攤到越多天），但 validation 的優勢沒有延續到 test。

#### 3.2 1 日排序有毛優勢，但被成本吃掉

- h = 1 時，各訊號在最佳 K 下，前 K 名每批比等權多 11–31 bps（上市）、3–27 bps（上櫃），validation 與 test 方向一致。
- 這個持有窗口是同一交易日的開盤到收盤，屬於當沖；主要成本情境來回需 0.52%（上市）。
- 即使折扣 0.28、無滑價，h = 1 的 117 組設定在兩個期間都沒有任何一組的淨超額區間下界大於 0。

#### 3.3 長天期的尾端反轉

| 期間 | 市場 | 訊號 | h | Rank IC | 最佳十分位 − 等權 | 最差十分位 − 等權 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Validation | 上櫃 | R22-IR | 10 | +0.074 | +95.0 bps | −91.6 bps |
| Test | 上櫃 | R22-IR | 10 | +0.005 | −46.6 bps | +289.8 bps |
| Test | 上市 | R21-IR | 10 | +0.024 | −30.7 bps | +189.0 bps |
| Test | 上市 | R22-IR | 10 | +0.068 | +104.6 bps | +131.3 bps |

在 test，預測最差的十分位常大幅勝過最佳十分位，即使整體 Rank IC 仍為正。長天期的 Rank IC 主要來自中段排序，只做多的前 K 名無法利用。

#### 3.4 大盤與等權的落差

以不重疊的 10 日窗口連乘（窗口之間的隔夜不計）：

- Test：加權指數 +47.5%，上市等權 +17.3%；櫃買指數 +56.0%，上櫃等權 +28.4%。
- Validation：加權指數 +20.5%，上市等權 +1.8%。

大型權值股主導了這段期間，因此以指數避險的版本全部大幅為負。這是規模效果，不是模型的選股能力。

#### 3.5 為何需要 1b

第一階段的「樂觀換手」敏感度只讓續抱股票不重複付費，但有兩個偏誤：

- 漏算續抱股票從收盤到下一個開盤的隔夜跳空。
- 仍以當沖稅率計稅。

因此它不能證明降低換手有效。1b 以正確的逐部位記帳重新檢驗。

### 4. 第一階段 1b：持有緩衝與逐部位記帳（§13.5）

規則：

- 資金分成 K 個等額位置。持股排名仍在前 b × K 就續抱，空出的位置補入排名最高的未持有股票。
- 所有買賣在下一個交易日開盤成交，以開盤到開盤的報酬記帳。
- 開盤跳空達 ±9.5% 視為無法成交，不設持有上限。
- 網格：預測天數 s ∈ {1, 3, 5}、K ∈ {10, 20, 50}、b ∈ {1, 2, 4}，共 37,908 組設定。
- 寫入協定時已聲明：test 已在第一階段事後分析中看過，1b 的 test 結果只作探索性判讀。

#### 4.1 記帳恆等式

同一檔股票相鄰兩日的資料列滿足 `abs[d, h+1] − abs[d+1, h] = log(open(d+2) / open(d+1))`：

- Validation 96,166 組、test 93,402 組相鄰資料列全部成立，股票與指數皆然（誤差 ≤ 10⁻⁵ 的比例為 100%）。
- 缺下一日資料列的股票-日期：validation 273 筆、test 203 筆。

#### 4.2 主要設定與門檻

| 市場 | 訊號 | s | K | b | validation 年化淨超額 [95% 區間] | test 年化淨超額 [95% 區間] | test IR | test 每日換手 | test 平均持有（交易日） | 通過 |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 上市 | ENS-IR | 5 | 10 | 2 | +30.5% [−6.9%, +64.9%] | −12.8% [−48.3%, +26.4%] | −1.01 | 5.4% | 15.7 | 否 |
| 上櫃 | R22-IR | 5 | 50 | 4 | +18.9% [+1.0%, +37.9%] | −17.9% [−47.9%, +13.7%] | −1.94 | 1.8% | 36.3 | 否 |
| 合併 | R21-IR | 5 | 10 | 4 | +42.4% [+13.2%, +72.2%] | −39.7% [−92.7%, +2.9%] | −2.67 | 7.2% | 12.3 | 否 |

上櫃與合併市場的 validation 區間下界雖大於 0，但它們是從 351 組候選中取最大值。主要成本情境、只做多的 1,053 組 validation 設定中，只有 3 組的區間下界大於 0。

#### 4.3 緩衝的效果

以下為 10 個 alpha 訊號與 3 個預測天期的平均，K = 10、主要成本情境、只做多（年化）：

| 期間 | 市場 | b | 毛超額 | 成本拖累 | 淨超額 | 每日換手 | 平均持有（交易日） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 上市 | 1 | +9.2% | 36.1% | −26.9% | 21.4% | 5.0 |
| Validation | 上市 | 2 | +4.6% | 13.9% | −9.4% | 7.6% | 14.5 |
| Validation | 上市 | 4 | +4.0% | 8.0% | −4.0% | 4.0% | 25.6 |
| Test | 上市 | 1 | +0.5% | 33.1% | −32.6% | 19.7% | 5.3 |
| Test | 上市 | 4 | −3.9% | 8.3% | −12.2% | 4.2% | 29.2 |
| Validation | 上櫃 | 1 | −2.5% | 39.8% | −42.3% | 17.4% | 5.8 |
| Validation | 上櫃 | 4 | −8.5% | 9.3% | −17.8% | 3.2% | 28.0 |
| Test | 上櫃 | 1 | −34.7% | 39.5% | −74.3% | 17.0% | 6.0 |
| Test | 上櫃 | 4 | −37.2% | 11.2% | −48.4% | 4.1% | 28.6 |

每天換成前 K 名（b = 1）時，改成持有過夜後的開盤到開盤毛超額只剩年化 +9.2%（validation）與 +0.5%（test），遠低於第一階段 h = 1 的交易時段優勢（每日約 20–30 bps）。緩衝省下約四分之三的成本，但沒有可保留的毛優勢。

#### 4.4 GRU baseline 的 test 結果不採信

GRU 在 test 的部分緩衝設定表現很好，例如合併市場 s = 1、K = 20、b = 4，test 年化淨超額 +118%（IR 3.5）。但同一設定在 validation 只有 +3.4%（IR 0.25），未被選中。第 5 節也顯示，GRU 前 10 名的隔夜報酬在 validation 為負、在 test 轉正，屬於期間特有的現象，不能作為可用訊號的證據。

### 5. 事後診斷：交易時段與隔夜的拆解

方法：

- 以 1 日預測（s = 1）每天選前 10 名，將相對等權的超額報酬拆成三段：
  - 次一交易日開盤到收盤（`abs[d, 1]`）；
  - 收盤到再下一個開盤（開盤到開盤報酬減前一段）；
  - 第二個交易日的開盤到收盤。
- 數值是每日超額的平均（bps），括號內為 t 值。
- 這是看過結果後的描述性分析，以唯讀腳本計算，未納入 `backtest/` 套件。

| 期間 | 市場 | 訊號 | 交易時段 | 隔夜 | 第二日交易時段 |
| --- | --- | --- | ---: | ---: | ---: |
| Validation | 上市 | ENS-IR | +29.0 (4.1) | −19.7 (−5.8) | +25.4 (3.6) |
| Validation | 上市 | R22-IR | +26.2 (3.8) | −16.3 (−5.5) | +24.2 (3.6) |
| Validation | 上市 | GBDT | +20.1 (3.3) | −15.2 (−4.8) | +19.4 (3.1) |
| Validation | 上市 | GRU | +30.9 (3.9) | −15.1 (−3.2) | +16.9 (2.0) |
| Validation | 上櫃 | ENS-IR | +15.8 (2.1) | −6.6 (−1.4) | +14.1 (1.8) |
| Validation | 上櫃 | GBDT | +16.2 (2.0) | −27.0 (−5.4) | +13.5 (1.7) |
| Test | 上市 | ENS-IR | +31.0 (4.1) | −22.9 (−6.0) | +26.5 (3.4) |
| Test | 上市 | R22-IR | +28.2 (3.8) | −5.1 (−1.1) | +25.9 (3.4) |
| Test | 上市 | GBDT | +25.5 (2.2) | −26.0 (−3.9) | +34.3 (3.4) |
| Test | 上市 | GRU | +25.0 (1.6) | +22.3 (1.6) | +26.1 (1.7) |
| Test | 上櫃 | ENS-IR | +12.7 (1.3) | −27.2 (−4.1) | +13.3 (1.4) |
| Test | 上櫃 | GBDT | +29.3 (2.6) | −29.6 (−3.5) | +18.8 (1.6) |

解讀：

- 被選中的股票在交易時段領先、在隔夜落後，第二個交易日的交易時段仍然領先。Kronos 與 GBDT 皆然；GRU 在 test 的隔夜轉正，與 validation 相反。
- 1 日 alpha 標籤正是次一交易日的開盤到收盤，模型學到的就是這段交易時段報酬。
- 只在交易時段持有就是當沖，需付 0.52%（上市，主要成本情境）的來回成本，高於約 0.2%–0.3% 的優勢。
- 持有過夜則被隔夜反轉抵銷。

### 6. 對模型開發的意涵

- 現有訊號不適合「收盤後產生訊號、隔日開盤進場、持有數天到兩週」的交易方式。
- 以「次一交易日開盤到收盤」或「開盤到第 h 日收盤」的 Rank IC 評估模型，會高估可持有過夜的優勢。評估需要與實際持有方式一致的報酬口徑，例如開盤到開盤，以及扣成本後的前 K 名表現。
- 長天期的整體 Rank IC 可能為正，但前 K 名的尾端反轉；評估需要直接看尾端（前 K 名、十分位），不能只看整體相關。
- Validation 選出的回測設定在 test 反轉，與先前觀察到的「台股 validation 排名無法預測 test 排名」一致。

### 7. 限制

- Validation 與 test 各約 5 個月；test 已多次用於研究比較，1b 的 test 只作探索性判讀。
- 未模擬成交量與流動性、每筆最低手續費、整股交易單位、股利所得稅與補充保費；母體可能排除了之後停牌或下市的股票。
- 漲跌停以還原價的開盤跳空近似；1b 的位置不再平衡，權重會漂移。
- 回測在本機沙箱內以單一 worker 執行；多程序路徑的一致性測試在沙箱中被略過，尚未在沙箱外驗證。
- R09／R10（LoRA32／64，1–14 日平均 Rank IC 最穩定）沒有逐樣本匯出，未納入回測。

### 8. 重現方式

```bash
cd backtest
uv sync
uv run python -m twbacktest.stage1
uv run python -m twbacktest.stage1b
```

| 項目 | 第一階段 | 1b |
| --- | --- | --- |
| 事先登記 commit | `5ddeff6` | `c4ee514` |
| 程式 commit | `a292a3b` | `4a5bdbb` |
| 設定檔 | `backtest/config/stage1.json` | `backtest/config/stage1b.json` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/stage1-20261018T221138Z-a2a9d283/` | `artifacts/diagnostics/backtest/stage1b-20261018T224557Z-f9cd85b4/` |

輸出目錄包含 `report.json`、`grid.csv`、`summary.md`；1b 另有 `daily.csv`。

## English

### 1. Purpose and conclusions

This report records two pre-registered local backtests:

- **Stage 1** ([experiment protocol §13.4](experiment_protocol.md)): fixed holding periods.
- **Stage 1b** ([experiment protocol §13.5](experiment_protocol.md)): a hold buffer with per-position accounting.

Both test whether the trained models' Taiwan cross-sectional ranking signals beat same-market equal-weight holding after real Taiwan trading costs. The backtests do not retrain models or use RunPod; the code is in `backtest/`, with its own uv environment.

Conclusions:

1. **Neither backtest passed its pre-registered gate on TWSE, TPEX or the combined market.**
   - Stage 1: the primary configurations chosen on validation lost 37% to 44% a year in net excess on test.
   - Stage 1b: the primary configurations lost 13% to 40% on test; stage 1b test results are exploratory only.
2. **The top-K edge exists only during trading sessions.** Post-hoc diagnostics show:
   - From the next session's open to its close, the top 10 beat equal weight by about 20–31 bps on TWSE or 13–29 bps on TPEX; TWSE is significant in both periods, TPEX is weaker.
   - From that close to the following open they trail by about 5–30 bps (GRU on test is the exception).
   - Holding overnight roughly cancels the edge; holding only during the session means day trading, whose costs exceed the edge.
3. **Long-horizon Rank IC does not turn into long-only top-K returns.** At a 10-session holding, the test bottom decile often beats the top decile by a wide margin.
4. **Lower turnover saves costs but cannot recover the edge.** The hold buffer cuts daily turnover from 12%–24% to 1%–7% and the cost drag to about a quarter, but little open-to-open gross excess was there to begin with.
5. **As pre-registered, the lower-turnover route stops and work returns to the models.** Confirmation still requires unseen stage 3 data (from 2026-06-01 onward).

### 2. Data and shared settings

| Item | Content |
| --- | --- |
| Population | TWSE and TPEX rows of the P0 evaluation population; row order verified against every export by sample digest |
| Validation | 2025-06-02 to 2025-11-10, 112 signal dates; 65,517 TWSE and 31,846 TPEX rows |
| Test | 2025-12-01 to 2026-05-11, 103 signal dates; 63,313 TWSE and 31,285 TPEX rows |
| Signals (13) | 8 Kronos alpha (R21, R21-IR, R22, R22-IR, R23, R24 datemix, and the LOSS and IR three-seed ensembles), GRU/GBDT alpha, and 3 direction models (signal P(up) − P(down)) |
| Returns | Adjusted log returns `absolute_targets`: open of session t+1 to close of session t+h |
| Benchmark | Same-date, same-market equal weight of all stocks, cost-free |
| Costs | Commission 0.1425% × discount; transaction tax 0.3% for stocks and 0.1% for ETFs; 0.15% for stock day trades; slippage 10 bps per side on TWSE and 20 bps on TPEX |
| Primary costs | Discount 0.6, slippage ×1 |

Round-trip costs:

| Market | Day trade (same-day round trip) | Held overnight |
| --- | ---: | ---: |
| TWSE | 0.52% | 0.67% |
| TPEX | 0.72% | 0.87% |
| TWSE, discount 0.28, no slippage | 0.23% | 0.38% |

### 3. Stage 1: fixed holding periods (§13.4)

Rule: rank after the close of session t, buy the top K equally weighted at the open of t+1 and sell at the close of t+h; h ∈ {1, 2, 3, 5, 10}, K ∈ {10, 20, 50}; 42,120 configurations.

#### 3.1 Primary configurations and gate

| Market | Signal | h | K | Validation annual net excess [95% CI] | Test annual net excess [95% CI] | Test IR | Passed |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| TWSE | R21-IR | 10 | 10 | +19.4% [−3.5%, +49.1%] | −36.8% [−51.0%, −19.1%] | −2.70 | no |
| TPEX | R22-IR | 10 | 20 | −0.2% [−18.5%, +19.3%] | −42.7% [−63.4%, −23.5%] | −3.43 | no |
| TW | R21-IR | 10 | 50 | +7.4% [−3.1%, +18.7%] | −43.8% [−66.8%, −24.0%] | −4.48 | no |

All three markets chose 10-session holding (a longer hold spreads costs over more days), but the validation advantage did not carry over to test.

#### 3.2 One-day rankings have a gross edge that costs consume

- At h = 1, each signal's top K at its best K beats equal weight by 11–31 bps per batch on TWSE and 3–27 bps on TPEX, in the same direction on validation and test.
- That holding window runs from one session's open to its close, a day trade; a round trip costs 0.52% on TWSE under the primary costs.
- Even with discount 0.28 and no slippage, none of the 117 h = 1 configurations has a net-excess interval lower bound above 0 in either period.

#### 3.3 Long-horizon tails invert

| Period | Market | Signal | h | Rank IC | Top decile − EW | Bottom decile − EW |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Validation | TPEX | R22-IR | 10 | +0.074 | +95.0 bps | −91.6 bps |
| Test | TPEX | R22-IR | 10 | +0.005 | −46.6 bps | +289.8 bps |
| Test | TWSE | R21-IR | 10 | +0.024 | −30.7 bps | +189.0 bps |
| Test | TWSE | R22-IR | 10 | +0.068 | +104.6 bps | +131.3 bps |

On test, the bottom-ranked decile often beats the top decile by a wide margin even when overall Rank IC stays positive. Long-horizon Rank IC comes mostly from the middle of the ranking, which long-only top-K cannot use.

#### 3.4 Index versus equal weight

Compounding non-overlapping 10-session windows (overnight gaps between windows excluded):

- Test: TAIEX +47.5% versus TWSE equal weight +17.3%; the TPEx index +56.0% versus TPEX equal weight +28.4%.
- Validation: TAIEX +20.5% versus TWSE equal weight +1.8%.

Large-cap stocks dominated the period, so every index-hedged version is deeply negative. This is a size effect, not the models' stock selection.

#### 3.5 Why stage 1b was needed

Stage 1's "optimistic turnover" sensitivity only stopped charging held names again, and it carried two biases:

- It omitted the overnight gap of held names from the close to the next open.
- It still applied the day-trade tax.

It therefore cannot show that lower turnover works. Stage 1b re-tests this with correct per-position accounting.

### 4. Stage 1b: hold buffer with per-position accounting (§13.5)

Rule:

- Capital is split into K equal slots. Held stocks still ranked within the top b × K are kept, and empty slots take the best-ranked stocks not yet held.
- All trades execute at the next session's open and are accounted for with open-to-open returns.
- Opening gaps of ±9.5% count as unfillable; there is no maximum holding period.
- Grid: score horizon s ∈ {1, 3, 5}, K ∈ {10, 20, 50}, b ∈ {1, 2, 4}; 37,908 configurations.
- When it was pre-registered, the protocol declared that test had already been seen in the stage 1 post-hoc analysis, so stage 1b test results are exploratory.

#### 4.1 Accounting identity

One stock's rows on adjacent dates satisfy `abs[d, h+1] − abs[d+1, h] = log(open(d+2) / open(d+1))`:

- All 96,166 adjacent pairs in validation and 93,402 in test satisfy it, for stocks and for the index (100% within 10⁻⁵).
- Stock-dates without a next-date row: 273 in validation and 203 in test.

#### 4.2 Primary configurations and gate

| Market | Signal | s | K | b | Validation annual net excess [95% CI] | Test annual net excess [95% CI] | Test IR | Test daily turnover | Test mean holding (sessions) | Passed |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| TWSE | ENS-IR | 5 | 10 | 2 | +30.5% [−6.9%, +64.9%] | −12.8% [−48.3%, +26.4%] | −1.01 | 5.4% | 15.7 | no |
| TPEX | R22-IR | 5 | 50 | 4 | +18.9% [+1.0%, +37.9%] | −17.9% [−47.9%, +13.7%] | −1.94 | 1.8% | 36.3 | no |
| TW | R21-IR | 5 | 10 | 4 | +42.4% [+13.2%, +72.2%] | −39.7% [−92.7%, +2.9%] | −2.67 | 7.2% | 12.3 | no |

The TPEX and TW validation lower bounds are above 0, but each is the maximum of 351 candidates. Of the 1,053 long-only validation configurations under the primary costs, only 3 have a lower bound above 0.

#### 4.3 Effect of the buffer

Averages over the 10 alpha signals and 3 score horizons, K = 10, primary costs, long-only (annualized):

| Period | Market | b | Gross excess | Cost drag | Net excess | Daily turnover | Mean holding (sessions) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | TWSE | 1 | +9.2% | 36.1% | −26.9% | 21.4% | 5.0 |
| Validation | TWSE | 2 | +4.6% | 13.9% | −9.4% | 7.6% | 14.5 |
| Validation | TWSE | 4 | +4.0% | 8.0% | −4.0% | 4.0% | 25.6 |
| Test | TWSE | 1 | +0.5% | 33.1% | −32.6% | 19.7% | 5.3 |
| Test | TWSE | 4 | −3.9% | 8.3% | −12.2% | 4.2% | 29.2 |
| Validation | TPEX | 1 | −2.5% | 39.8% | −42.3% | 17.4% | 5.8 |
| Validation | TPEX | 4 | −8.5% | 9.3% | −17.8% | 3.2% | 28.0 |
| Test | TPEX | 1 | −34.7% | 39.5% | −74.3% | 17.0% | 6.0 |
| Test | TPEX | 4 | −37.2% | 11.2% | −48.4% | 4.1% | 28.6 |

When the portfolio is replaced by the top K every day (b = 1), holding overnight leaves an open-to-open gross excess of only +9.2% (validation) and +0.5% (test) a year, far below stage 1's h = 1 session edge (about 20–30 bps a day). The buffer saves about three quarters of the costs, but there is no gross edge left to keep.

#### 4.4 GRU baseline test results are not adopted

Some GRU buffer configurations did very well on test; for example TW with s = 1, K = 20, b = 4 earned +118% annual net excess (IR 3.5). On validation the same configuration earned only +3.4% (IR 0.25) and was not selected. Section 5 also shows that GRU's top 10 had negative overnight returns on validation and positive ones on test. This is specific to the period and is not evidence of a usable signal.

### 5. Post-hoc diagnostic: session versus overnight

Method:

- With the 1-day prediction (s = 1), select the top 10 each day and split their excess over equal weight into three parts:
  - the next session from open to close (`abs[d, 1]`);
  - that close to the following open (the open-to-open return minus the first part);
  - the second session from open to close.
- Values are mean daily excess in bps, with t-statistics in parentheses.
- This descriptive analysis was done after seeing the results, with read-only scripts that are not part of the `backtest/` package.

| Period | Market | Signal | Session | Overnight | Second session |
| --- | --- | --- | ---: | ---: | ---: |
| Validation | TWSE | ENS-IR | +29.0 (4.1) | −19.7 (−5.8) | +25.4 (3.6) |
| Validation | TWSE | R22-IR | +26.2 (3.8) | −16.3 (−5.5) | +24.2 (3.6) |
| Validation | TWSE | GBDT | +20.1 (3.3) | −15.2 (−4.8) | +19.4 (3.1) |
| Validation | TWSE | GRU | +30.9 (3.9) | −15.1 (−3.2) | +16.9 (2.0) |
| Validation | TPEX | ENS-IR | +15.8 (2.1) | −6.6 (−1.4) | +14.1 (1.8) |
| Validation | TPEX | GBDT | +16.2 (2.0) | −27.0 (−5.4) | +13.5 (1.7) |
| Test | TWSE | ENS-IR | +31.0 (4.1) | −22.9 (−6.0) | +26.5 (3.4) |
| Test | TWSE | R22-IR | +28.2 (3.8) | −5.1 (−1.1) | +25.9 (3.4) |
| Test | TWSE | GBDT | +25.5 (2.2) | −26.0 (−3.9) | +34.3 (3.4) |
| Test | TWSE | GRU | +25.0 (1.6) | +22.3 (1.6) | +26.1 (1.7) |
| Test | TPEX | ENS-IR | +12.7 (1.3) | −27.2 (−4.1) | +13.3 (1.4) |
| Test | TPEX | GBDT | +29.3 (2.6) | −29.6 (−3.5) | +18.8 (1.6) |

Interpretation:

- The selected stocks lead during the session, lag overnight, and still lead during the second session. This holds for Kronos and GBDT alike; GRU's overnight turns positive on test, the opposite of validation.
- The 1-day alpha label is exactly the next session's open-to-close return, so that session return is what the models learned.
- Holding only during the session is a day trade with a 0.52% round trip (TWSE, primary costs), more than the edge of about 0.2%–0.3%.
- Holding overnight gives the edge back.

### 6. Implications for model development

- The current signals do not suit the intended trading style: signals after the close, entry at the next open, holding for days to two weeks.
- Evaluating models by Rank IC on next-session open-to-close or open-to-close-of-h returns overstates any edge that survives overnight. Evaluation needs returns that match the actual holding, such as open-to-open, and top-K performance after costs.
- Overall long-horizon Rank IC can be positive while the top-K tail inverts; evaluation must look at the tails directly (top K, deciles), not only at overall correlation.
- Backtest configurations chosen on validation reverse on test, consistent with the earlier finding that Taiwan validation rankings do not predict test rankings.

### 7. Limits

- Validation and test each cover about five months; test has been consulted repeatedly in research, and the stage 1b test is exploratory.
- Volume and liquidity, minimum commissions, board lots, and dividend income tax and the supplementary health premium are not simulated; the population may exclude stocks later suspended or delisted.
- Price limits are approximated from adjusted opening gaps; stage 1b slots are not rebalanced, so weights drift.
- The backtests ran with one worker inside the local sandbox; the multi-process consistency tests were skipped there and have not yet been verified outside it.
- R09/R10 (LoRA32/64, the most stable mean 1–14 day Rank IC) have no per-sample exports and are not included.

### 8. Reproduction

```bash
cd backtest
uv sync
uv run python -m twbacktest.stage1
uv run python -m twbacktest.stage1b
```

| Item | Stage 1 | Stage 1b |
| --- | --- | --- |
| Pre-registration commit | `5ddeff6` | `c4ee514` |
| Code commit | `a292a3b` | `4a5bdbb` |
| Configuration | `backtest/config/stage1.json` | `backtest/config/stage1b.json` |
| Output (not tracked by Git) | `artifacts/diagnostics/backtest/stage1-20261018T221138Z-a2a9d283/` | `artifacts/diagnostics/backtest/stage1b-20261018T224557Z-f9cd85b4/` |

Each output directory contains `report.json`, `grid.csv` and `summary.md`; stage 1b also writes `daily.csv`.
