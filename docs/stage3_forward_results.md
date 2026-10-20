# 第三階段前瞻檢驗結果 / Stage 3 Forward Test Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.8](experiment_protocol.md) 及其修訂、修訂補充的前瞻檢驗：以 2026-06-01 之後從未用過的台股資料，檢驗上市 10 日的 R22-IR（R22 checkpoint-018399）。主要假說、判準與次要描述都在取得前瞻資料前固定；評分只執行一次。

結論：

1. **所有資料檢查與管線一致性檢查都通過，評分有效。**
2. **主要假說未通過。**
   - 前十分位開盤到開盤超額：+57.1 bps，95% 區間 [−10.5, +124.3]。下界小於 0，判準 1 不成立。
   - 固定組合年化淨超額：+7.4%（[−36.6%, +56.0%]）。平均大於 0，判準 2 成立。
   - 兩項都要成立才通過。依 §13.8 第 6 點：**不採用，回到模型本身的改善。**
3. **未通過不等於證實沒有優勢。** 前瞻的點估計（+57.1 bps）與 validation（+69.6 bps）相近，只是區間太寬。前瞻期間只有 82 個訊號日，10 日窗口互相重疊，互不重疊的窗口約 7 段。
4. **以持有指標選 checkpoint，在前瞻資料上沒有顯現優勢。** 上市 10 日的 6 個訊號中，R22-IR 的前十分位超額只高於 R21-loss（+42.4 bps）；loss 選出的 R22 checkpoint（011040）點估計反而是 +128.7 bps。所有區間都大幅重疊。
5. **上市的排序資訊延續到前瞻期間，但不足以支持交易。** 6 個訊號的上市 10 日開盤到開盤 Rank IC 都為正（0.074–0.149）。後十分位都低於等權平均，前後十分位差的點估計都為正（+78 到 +213 bps）。但沒有一個前後差、也沒有一個上市固定組合的區間排除 0。
6. **多數訊號的前十分位仍是「交易時段賺、隔夜賠」。** 上市 10 日，6 個訊號中有 5 個的隔夜部分為負（−42 到 −185 bps）；只有 R22-IR 例外（交易時段 +9.4、隔夜 +49.6 bps）。
7. **上櫃仍然偏弱。** R23 與 ENS-IR 的 5 日組合、R23 的 10 日組合，年化淨超額的區間都低於 0。
8. **ENS-IR 與 R23 的上市 10 日前十分位區間下界大於 0**（+70.3、+46.8 bps），但這是 24 個排序格子中的事後觀察，依協定只作描述，不能據此採用。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 前瞻資料 | 獨立前瞻管線（§13.8 修訂補充）：證交所與櫃買中心官方資料，2025-03-01 至 2026-10-20（不含）；不建立訓練資料組 |
| 輸入包 | `b9d633c5916f91e678ac532fc27f30cc65ee251bebb7f2dbf41543bbfd637576`；`overlap` 94,598 列（訊號日 2025-12-01 至 2026-05-11，103 日）、`forward` 86,830 列（訊號日 2026-06-01 至 2026-09-24，82 日） |
| 休市日補充 | 只有一筆：XTAI 2026-07-10 列為休市（颱風停市，固定日曆仍列為交易日）。只套用在這個輸入包，訓練與評估日曆不變 |
| 推論 | RunPod RTX 5090，CUDA bf16（`export-signal-predictions --forward-package`），2026-10-20 一個工作內完成 6 個 checkpoint 的 `overlap` 與 `forward` |
| 標籤與指標 | 與 §13.6 相同：開盤到開盤，前十分位減同日同市場等權平均；circular moving-block bootstrap，區塊 20（h = 10）、重抽 1,000 次、seed 11 |
| 固定組合 | s = h、K = 20、b = 2、主要成本情境（折扣 0.6、滑價 1 倍）、只做多 |
| 判定 | 只看主要假說（R22-IR、上市、h = 10）；其餘只作描述 |

### 3. 結果

#### 3.1 評分前的檢查（全部通過）

| 檢查 | 結果 |
| --- | --- |
| 資料檢查 1：輸入包 | 519 個封存檔案的雜湊全部相符 |
| 資料檢查 2：記帳恆等式 | 85,618 組相鄰資料列，股票與指數 100% 成立 |
| 資料檢查 3：`overlap` 標籤 | 94,598 列全部對到共用評估快照，最大差距 4.4×10⁻¹⁶ |
| 資料檢查 4：模型輸入 | 94,598 列全部對到，最大相對差距 6.3×10⁻¹⁶ |
| 管線一致性 | 與 2026-10-19 的 test 匯出對到 94,598 列；預測不是逐位元相同，相關係數 0.9999999999996；重算的前十分位超額 +103.23 bps，與選點報告的 +103.2 bps 相同 |

#### 3.2 主要假說：R22-IR，上市，10 日

| 指標 | validation | test | 前瞻 |
| --- | --- | --- | --- |
| 前十分位超額（bps） | +69.6 [+8.6, +139.0] | +103.2 [+57.3, +147.8] | **+57.1 [−10.5, +124.3]** |
| 後十分位超額（bps） | −35.6 | +168.9 | −21.1 |
| 前後十分位差（bps） | +105.2 | −65.7 | +78.2 [−126.2, +279.2] |
| 交易時段 / 隔夜（bps） | +78.5 / −10.7 | +92.5 / +18.6 | +9.4 / +49.6 |
| 開盤到開盤 Rank IC（IR） | 0.036（0.32） | 0.071（0.58） | 0.074（0.64） |
| 固定組合年化淨超額 | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | **+7.4% [−36.6%, +56.0%]** |

前瞻期間的固定組合：年化毛超額 +27.4%，成本拖累 +19.9%，每日換手 10.5%，最大回撤 −24.4%。成本吃掉了大部分的毛超額。

#### 3.3 次要描述：上市 10 日（bps，[95% 區間]）

| 訊號 | 前十分位 | 後十分位 | 前後差 | 交易時段 / 隔夜 | Rank IC（IR） |
| --- | --- | ---: | --- | --- | --- |
| R22-IR（主要） | +57.1 [−10.5, +124.3] | −21.1 | +78.2 [−126.2, +279.2] | +9.4 / +49.6 | 0.074（0.64） |
| ENS-IR | +142.9 [+70.3, +223.9] | −60.3 | +203.1 [−97.8, +566.0] | +203.4 / −42.4 | 0.125（0.85） |
| R23 | +122.1 [+46.8, +194.7] | −90.6 | +212.7 [−39.4, +500.5] | +208.5 / −73.0 | 0.131（0.89） |
| R21-018399 | +107.4 [−40.8, +287.7] | −29.1 | +136.5 [−260.2, +612.5] | +296.6 / −154.8 | 0.129（0.57） |
| R21-loss（003680） | +42.4 [−121.1, +264.1] | −103.2 | +145.5 [−324.6, +757.8] | +276.5 / −185.3 | 0.130（0.51） |
| R22-loss（011040） | +128.7 [−17.4, +315.3] | −70.2 | +198.9 [−188.8, +719.9] | +348.0 / −177.8 | 0.149（0.56） |

#### 3.4 次要描述：上市 5 日（bps，[95% 區間]）

| 訊號 | 前十分位 | 後十分位 | 前後差 | 交易時段 / 隔夜 | Rank IC（IR） |
| --- | --- | ---: | --- | --- | --- |
| R22-IR | +29.7 [−21.5, +83.3] | +5.9 | +23.9 [−85.0, +126.4] | +37.6 / −4.7 | 0.067（0.40） |
| ENS-IR | +60.3 [−2.6, +127.7] | −10.2 | +70.5 [−90.0, +243.6] | +117.9 / −45.4 | 0.093（0.46） |
| R23 | +36.1 [−10.5, +87.6] | −24.4 | +60.5 [−66.3, +187.4] | +97.8 / −55.1 | 0.091（0.49） |
| R21-018399 | +54.5 [−52.0, +164.8] | −14.5 | +69.0 [−143.4, +297.2] | +159.0 / −82.3 | 0.101（0.36） |
| R21-loss（003680） | +21.3 [−78.4, +126.1] | −38.2 | +59.5 [−186.7, +312.7] | +131.5 / −83.6 | 0.102（0.36） |
| R22-loss（011040） | +55.2 [−55.2, +168.4] | −19.0 | +74.3 [−154.2, +315.4] | +172.5 / −92.0 | 0.114（0.36） |

#### 3.5 次要描述：上櫃 10 日（bps，[95% 區間]）

| 訊號 | 前十分位 | 後十分位 | 前後差 | 交易時段 / 隔夜 | Rank IC（IR） |
| --- | --- | ---: | --- | --- | --- |
| R22-IR | +18.4 [−19.0, +55.5] | +44.8 | −26.3 [−253.9, +194.9] | +3.2 / +31.4 | 0.028（0.31） |
| ENS-IR | −17.1 [−106.7, +76.0] | +95.8 | −112.9 [−450.8, +242.0] | +40.7 / −27.1 | 0.049（0.41） |
| R23 | −49.1 [−104.2, −1.9] | +136.1 | −185.2 [−462.2, +92.5] | −22.9 / −15.5 | 0.033（0.31） |
| R21-018399 | +38.2 [−171.5, +271.4] | +108.6 | −70.4 [−506.0, +408.7] | +186.0 / −104.3 | 0.062（0.33） |
| R21-loss（003680） | +95.0 [−137.8, +359.3] | +86.3 | +8.7 [−525.1, +583.4] | +281.0 / −135.6 | 0.100（0.50） |
| R22-loss（011040） | +57.8 [−176.4, +319.1] | +69.4 | −11.5 [−507.1, +526.0] | +249.4 / −143.4 | 0.089（0.43） |

R22-IR 的上櫃 5 日：前十分位 −25.5 bps [−70.7, +16.3]，後十分位 +42.9，前後差 −68.4 bps。

#### 3.6 次要描述：固定組合（年化淨超額 [95% 區間]）

| 訊號 | 上市 10 日 | 上市 5 日 | 上櫃 10 日 | 上櫃 5 日 | 全台 10 日 |
| --- | --- | --- | --- | --- | --- |
| R22-IR（主要） | +7.4% [−36.6%, +56.0%] | +5.3% [−34.5%, +47.2%] | −24.4% [−50.1%, +1.3%] | −5.2% [−41.1%, +32.4%] | +14.3% [−25.4%, +59.8%] |
| ENS-IR | +26.3% [−15.8%, +68.6%] | +34.4% [−19.0%, +86.5%] | −20.0% [−58.0%, +12.8%] | −49.4% [−78.0%, −24.2%] | +0.5% [−41.6%, +46.1%] |
| R23 | +28.5% [−20.9%, +77.5%] | +16.5% [−32.6%, +66.0%] | −52.6% [−81.6%, −25.0%] | −64.8% [−96.5%, −30.9%] | +3.7% [−41.5%, +49.7%] |
| R21-018399 | +9.1% [−57.7%, +80.4%] | +14.7% [−51.3%, +84.2%] | −6.6% [−73.8%, +63.0%] | +6.0% [−75.9%, +89.4%] | +7.2% [−62.4%, +81.0%] |
| R21-loss（003680） | +7.4% [−61.7%, +83.7%] | +6.7% [−64.8%, +84.2%] | +24.3% [−45.9%, +99.8%] | +20.4% [−51.2%, +98.1%] | +17.0% [−60.1%, +103.7%] |
| R22-loss（011040） | +20.6% [−52.6%, +97.3%] | +25.3% [−47.6%, +102.7%] | +22.0% [−55.4%, +109.8%] | +9.5% [−68.4%, +101.5%] | +23.7% [−54.5%, +107.5%] |

上市組合沒有一個區間排除 0；上櫃有三個區間全部低於 0。

### 4. 意涵

- **依協定，R22-IR 不列為交易規則的候選訊號**，回到模型本身的改善。
- 這次的未通過主要是檢定力不足：點估計與 validation 相近，區間寬度約 ±67 bps。要在這個效果大小下得到排除 0 的區間，需要明顯更長的前瞻期間。
- 選點沒有帶來可確認的好處：R22-IR 的上市 10 日前十分位在 6 個訊號中排第 5；loss 選點與 Rank IC-IR 選點的點估計沒有一致的高低順序。「checkpoint 選擇和標籤一樣重要」的假設，在前瞻資料上沒有得到支持。
- 上市的排序資訊（Rank IC、前後差的點估計）在三個期間都存在，但每一段都太短、太吵，無法轉換成扣成本後可確認的超額。成本拖累每年約 6%–20%，在前瞻期間吃掉了 R22-IR 大部分的毛超額。
- 多數訊號仍有隔夜拖累，和回測 1b 的發現一致：優勢偏向交易時段，持有多日時要付出隔夜的代價。
- 次要訊號中 ENS-IR 與 R23 的前十分位下界大於 0，是之後若要繼續前瞻監測時，可以「事先登記」的候選，不是現在可以採用的結果。

### 5. 限制

- 訊號日 82 個（10 日標籤可用 72 個），10 日窗口重疊；互不重疊的窗口約 7 段，區間很寬。
- 只涵蓋 2026 年 6 月到 9 月這一段行情。
- 標的清單來自官方下載內容，可能缺少期間內下市的標的（存活者偏誤）。
- 未模擬成交量與流動性、每筆最低手續費、整股、股利稅。
- 次要描述共 24 個排序格子與 36 個組合格子，個別格子的顯著不能當成確認。
- 推論預測與 10-19 的匯出不是逐位元相同（相關係數 0.9999999999996），排序指標相同，不影響結論。

### 6. 重現方式

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --forward 2026-10-20 <6 個 RUN_ID/checkpoint-NNNNNN>
cd backtest
uv run python -m twbacktest.stage3
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `e4d823c`（§13.8）、`c19bedc`（修訂與 `stage3.json`）、`5afcae9`（修訂補充） |
| 程式 commit | `65bba60`、`b7d8561`（前瞻管線與匯出）、`cc57971`（評分）、`71ac85d`、`41278bc`（休市日只套用在輸入包）、`e93705c`（匯出修正） |
| 設定檔 | `backtest/config/stage3.json`（SHA-256 `87adfaa3…`） |
| 前瞻匯出（不受 Git 追蹤） | `artifacts/diagnostics/forward-predictions/2026-10-20/` 下 6 個 checkpoint 的 `export-20261020T1205…`–`T1251…` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/stage3-20261020T134953Z-6f781e13/`（`report.json`、`ranking.csv`、`portfolio.csv`、`summary.md`） |

## English

### 1. Purpose and conclusions

This report records the forward test of [experiment protocol §13.8](experiment_protocol.md), its amendment and its supplement: TWSE 10-session R22-IR (R22 checkpoint-018399) tested on Taiwan data from 2026-06-01 onward that had never been used. The primary hypothesis, criteria and secondary descriptions were fixed before the forward data were obtained, and scoring ran once.

Conclusions:

1. **Every data check and the pipeline consistency check passed, so the scoring is valid.**
2. **The primary hypothesis failed.**
   - Top-decile open-to-open excess: +57.1 bps, 95% interval [−10.5, +124.3]. The lower bound is below 0, so criterion 1 fails.
   - Fixed-portfolio annual net excess: +7.4% ([−36.6%, +56.0%]). The mean is above 0, so criterion 2 holds.
   - Both must hold. Under §13.8 item 6: **not adopted; work returns to improving the model itself.**
3. **Failing is not proof that there is no edge.** The forward point estimate (+57.1 bps) is close to validation (+69.6 bps); the interval is simply too wide. The forward period has only 82 signal dates with overlapping 10-session windows, about 7 non-overlapping windows.
4. **Checkpoint selection by a holding metric shows no advantage on forward data.** Among the six TWSE 10-session signals, R22-IR's top-decile excess is above only R21-loss (+42.4 bps); the loss-selected R22 checkpoint (011040) has a point estimate of +128.7 bps. All intervals overlap widely.
5. **TWSE ranking information carried into the forward period, but not enough to support trading.** All six signals have positive TWSE 10-session open-to-open Rank IC (0.074–0.149). Every bottom decile is below the equal-weight average and every top-minus-bottom point estimate is positive (+78 to +213 bps). No spread interval and no TWSE fixed-portfolio interval excludes 0.
6. **For most signals the top decile still earns in sessions and loses overnight.** At TWSE 10 sessions, 5 of 6 signals have a negative overnight part (−42 to −185 bps); R22-IR is the exception (session +9.4, overnight +49.6 bps).
7. **TPEX remains weak.** The 5-session portfolios of R23 and ENS-IR and the 10-session portfolio of R23 have annual net excess intervals entirely below 0.
8. **ENS-IR and R23 have TWSE 10-session top-decile lower bounds above 0** (+70.3 and +46.8 bps), but this is a post-hoc observation among 24 ranking cells; under the protocol it is descriptive only and cannot justify adoption.

### 2. Data and method

| Item | Content |
| --- | --- |
| Forward data | Independent forward pipeline (§13.8 supplement): official TWSE and TPEx data, 2025-03-01 to 2026-10-20 (exclusive); no training dataset is created |
| Input package | `b9d633c5916f91e678ac532fc27f30cc65ee251bebb7f2dbf41543bbfd637576`; `overlap` 94,598 rows (signal dates 2025-12-01 to 2026-05-11, 103 dates), `forward` 86,830 rows (signal dates 2026-06-01 to 2026-09-24, 82 dates) |
| Session additions | One only: XTAI 2026-07-10 marked closed (a typhoon closure the pinned calendar lists as a session). Applied to this package only; the training and evaluation calendar is unchanged |
| Inference | RunPod RTX 5090, CUDA bf16 (`export-signal-predictions --forward-package`); one job on 2026-10-20 covered `overlap` and `forward` for 6 checkpoints |
| Labels and metrics | As in §13.6: open to open, top decile minus the same-day, same-market equal-weight average; circular moving-block bootstrap, block 20 (h = 10), 1,000 replicates, seed 11 |
| Fixed portfolio | s = h, K = 20, b = 2, primary cost scenario (discount 0.6, slippage ×1), long only |
| Decision | Primary hypothesis only (R22-IR, TWSE, h = 10); everything else is descriptive |

### 3. Results

#### 3.1 Checks before scoring (all passed)

| Check | Result |
| --- | --- |
| Data check 1: input package | All 519 sealed files match their hashes |
| Data check 2: accounting identity | 85,618 adjacent row pairs; 100% hold for stocks and indices |
| Data check 3: `overlap` labels | All 94,598 rows match the shared evaluation snapshot; maximum difference 4.4×10⁻¹⁶ |
| Data check 4: model inputs | All 94,598 rows matched; maximum relative difference 6.3×10⁻¹⁶ |
| Pipeline consistency | 94,598 rows matched against the 2026-10-19 test export; predictions are not bit-identical, correlation 0.9999999999996; the recomputed top-decile excess is +103.23 bps, equal to the selection report's +103.2 bps |

#### 3.2 Primary hypothesis: R22-IR, TWSE, 10 sessions

| Metric | Validation | Test | Forward |
| --- | --- | --- | --- |
| Top-decile excess (bps) | +69.6 [+8.6, +139.0] | +103.2 [+57.3, +147.8] | **+57.1 [−10.5, +124.3]** |
| Bottom-decile excess (bps) | −35.6 | +168.9 | −21.1 |
| Top minus bottom (bps) | +105.2 | −65.7 | +78.2 [−126.2, +279.2] |
| Session / overnight (bps) | +78.5 / −10.7 | +92.5 / +18.6 | +9.4 / +49.6 |
| Open-to-open Rank IC (IR) | 0.036 (0.32) | 0.071 (0.58) | 0.074 (0.64) |
| Fixed-portfolio annual net excess | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | **+7.4% [−36.6%, +56.0%]** |

Forward fixed portfolio: annual gross excess +27.4%, cost drag +19.9%, daily turnover 10.5%, maximum drawdown −24.4%. Costs take most of the gross excess.

#### 3.3 Secondary description: TWSE, 10 sessions (bps, [95% interval])

| Signal | Top decile | Bottom decile | Top minus bottom | Session / overnight | Rank IC (IR) |
| --- | --- | ---: | --- | --- | --- |
| R22-IR (primary) | +57.1 [−10.5, +124.3] | −21.1 | +78.2 [−126.2, +279.2] | +9.4 / +49.6 | 0.074 (0.64) |
| ENS-IR | +142.9 [+70.3, +223.9] | −60.3 | +203.1 [−97.8, +566.0] | +203.4 / −42.4 | 0.125 (0.85) |
| R23 | +122.1 [+46.8, +194.7] | −90.6 | +212.7 [−39.4, +500.5] | +208.5 / −73.0 | 0.131 (0.89) |
| R21-018399 | +107.4 [−40.8, +287.7] | −29.1 | +136.5 [−260.2, +612.5] | +296.6 / −154.8 | 0.129 (0.57) |
| R21-loss (003680) | +42.4 [−121.1, +264.1] | −103.2 | +145.5 [−324.6, +757.8] | +276.5 / −185.3 | 0.130 (0.51) |
| R22-loss (011040) | +128.7 [−17.4, +315.3] | −70.2 | +198.9 [−188.8, +719.9] | +348.0 / −177.8 | 0.149 (0.56) |

#### 3.4 Secondary description: TWSE, 5 sessions (bps, [95% interval])

| Signal | Top decile | Bottom decile | Top minus bottom | Session / overnight | Rank IC (IR) |
| --- | --- | ---: | --- | --- | --- |
| R22-IR | +29.7 [−21.5, +83.3] | +5.9 | +23.9 [−85.0, +126.4] | +37.6 / −4.7 | 0.067 (0.40) |
| ENS-IR | +60.3 [−2.6, +127.7] | −10.2 | +70.5 [−90.0, +243.6] | +117.9 / −45.4 | 0.093 (0.46) |
| R23 | +36.1 [−10.5, +87.6] | −24.4 | +60.5 [−66.3, +187.4] | +97.8 / −55.1 | 0.091 (0.49) |
| R21-018399 | +54.5 [−52.0, +164.8] | −14.5 | +69.0 [−143.4, +297.2] | +159.0 / −82.3 | 0.101 (0.36) |
| R21-loss (003680) | +21.3 [−78.4, +126.1] | −38.2 | +59.5 [−186.7, +312.7] | +131.5 / −83.6 | 0.102 (0.36) |
| R22-loss (011040) | +55.2 [−55.2, +168.4] | −19.0 | +74.3 [−154.2, +315.4] | +172.5 / −92.0 | 0.114 (0.36) |

#### 3.5 Secondary description: TPEX, 10 sessions (bps, [95% interval])

| Signal | Top decile | Bottom decile | Top minus bottom | Session / overnight | Rank IC (IR) |
| --- | --- | ---: | --- | --- | --- |
| R22-IR | +18.4 [−19.0, +55.5] | +44.8 | −26.3 [−253.9, +194.9] | +3.2 / +31.4 | 0.028 (0.31) |
| ENS-IR | −17.1 [−106.7, +76.0] | +95.8 | −112.9 [−450.8, +242.0] | +40.7 / −27.1 | 0.049 (0.41) |
| R23 | −49.1 [−104.2, −1.9] | +136.1 | −185.2 [−462.2, +92.5] | −22.9 / −15.5 | 0.033 (0.31) |
| R21-018399 | +38.2 [−171.5, +271.4] | +108.6 | −70.4 [−506.0, +408.7] | +186.0 / −104.3 | 0.062 (0.33) |
| R21-loss (003680) | +95.0 [−137.8, +359.3] | +86.3 | +8.7 [−525.1, +583.4] | +281.0 / −135.6 | 0.100 (0.50) |
| R22-loss (011040) | +57.8 [−176.4, +319.1] | +69.4 | −11.5 [−507.1, +526.0] | +249.4 / −143.4 | 0.089 (0.43) |

R22-IR at TPEX 5 sessions: top decile −25.5 bps [−70.7, +16.3], bottom decile +42.9, top minus bottom −68.4 bps.

#### 3.6 Secondary description: fixed portfolios (annual net excess [95% interval])

| Signal | TWSE 10 | TWSE 5 | TPEX 10 | TPEX 5 | All Taiwan 10 |
| --- | --- | --- | --- | --- | --- |
| R22-IR (primary) | +7.4% [−36.6%, +56.0%] | +5.3% [−34.5%, +47.2%] | −24.4% [−50.1%, +1.3%] | −5.2% [−41.1%, +32.4%] | +14.3% [−25.4%, +59.8%] |
| ENS-IR | +26.3% [−15.8%, +68.6%] | +34.4% [−19.0%, +86.5%] | −20.0% [−58.0%, +12.8%] | −49.4% [−78.0%, −24.2%] | +0.5% [−41.6%, +46.1%] |
| R23 | +28.5% [−20.9%, +77.5%] | +16.5% [−32.6%, +66.0%] | −52.6% [−81.6%, −25.0%] | −64.8% [−96.5%, −30.9%] | +3.7% [−41.5%, +49.7%] |
| R21-018399 | +9.1% [−57.7%, +80.4%] | +14.7% [−51.3%, +84.2%] | −6.6% [−73.8%, +63.0%] | +6.0% [−75.9%, +89.4%] | +7.2% [−62.4%, +81.0%] |
| R21-loss (003680) | +7.4% [−61.7%, +83.7%] | +6.7% [−64.8%, +84.2%] | +24.3% [−45.9%, +99.8%] | +20.4% [−51.2%, +98.1%] | +17.0% [−60.1%, +103.7%] |
| R22-loss (011040) | +20.6% [−52.6%, +97.3%] | +25.3% [−47.6%, +102.7%] | +22.0% [−55.4%, +109.8%] | +9.5% [−68.4%, +101.5%] | +23.7% [−54.5%, +107.5%] |

No TWSE portfolio interval excludes 0; three TPEX intervals lie entirely below 0.

### 4. Implications

- **Under the protocol, R22-IR does not become a candidate signal for the trading rules**, and work returns to improving the model itself.
- This failure is mainly a lack of power: the point estimate is close to validation, and the interval is about ±67 bps wide. An interval excluding 0 at this effect size needs a clearly longer forward period.
- Checkpoint selection brought no confirmable benefit: R22-IR's TWSE 10-session top decile ranks fifth of the six signals, and loss-selected and Rank IC-IR-selected checkpoints show no consistent ordering. The hypothesis that checkpoint choice matters as much as labels is not supported on forward data.
- TWSE ranking information (Rank IC, top-minus-bottom point estimates) is present in all three periods, but each period is too short and noisy to turn into a confirmable net excess. Cost drag runs about 6%–20% a year and took most of R22-IR's gross excess in the forward period.
- Most signals still carry an overnight drag, consistent with backtest stage 1b: the edge leans toward the sessions, and multi-day holding pays for the overnight part.
- ENS-IR and R23, whose secondary lower bounds lie above 0, are candidates that could be pre-registered if forward monitoring continues; they are not results that can be adopted now.

### 5. Limits

- 82 signal dates (72 with 10-session labels) and overlapping 10-session windows; about 7 non-overlapping windows, so intervals are wide.
- Covers only June to September 2026.
- The symbol list comes from the official downloads and may miss symbols delisted during the period (survivorship bias).
- Volume and liquidity, the minimum fee per trade, round lots and dividend tax are not simulated.
- The secondary description has 24 ranking cells and 36 portfolio cells; significance in one cell is not a confirmation.
- Forward predictions are not bit-identical to the 10-19 export (correlation 0.9999999999996); ranking metrics are identical and the conclusion is unaffected.

### 6. Reproduction

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --forward 2026-10-20 <6 RUN_ID/checkpoint-NNNNNN selectors>
cd backtest
uv run python -m twbacktest.stage3
```

| Item | Content |
| --- | --- |
| Pre-registration commits | `e4d823c` (§13.8), `c19bedc` (amendment and `stage3.json`), `5afcae9` (supplement) |
| Code commits | `65bba60`, `b7d8561` (forward pipeline and export), `cc57971` (scoring), `71ac85d`, `41278bc` (session additions applied to the package only), `e93705c` (export fix) |
| Configuration | `backtest/config/stage3.json` (SHA-256 `87adfaa3…`) |
| Forward exports (untracked) | `export-20261020T1205…` to `T1251…` for the 6 checkpoints under `artifacts/diagnostics/forward-predictions/2026-10-20/` |
| Outputs (untracked) | `artifacts/diagnostics/backtest/stage3-20261020T134953Z-6f781e13/` (`report.json`, `ranking.csv`, `portfolio.csv`, `summary.md`) |
