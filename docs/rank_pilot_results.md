# Kronos 排序模型試行結果 / Kronos Rank Model Pilot Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.11](experiment_protocol.md) 的判定：Kronos 橫斷面排序模型 `a-rank64`（seed 42）與 §13.10 的 rank baseline 比較台股的排序能力。判定用 test；validation 只用於選點。§13.11 的分組在看過 §13.10 的 test 後才決定，所以本報告的判定屬於試行，最終確認以 2026-10-20 之後的前瞻資料為準。

結論：

1. **判定：H1 通過，H2、H3 未通過。**
   - H1（台股 1 日 Rank IC 大於 0）：0.0642 [0.0271, 0.1015]，通過。
   - H2（台股 1 日勝過 PatchTST）：配對差 −0.0022 [−0.0211, 0.0182]，未通過。
   - H3（台股 6–14 日 Rank IC 大於 0）：0.0082 [−0.0440, 0.0637]，未通過。
2. **依 §13.11 第 6 點的處置：**
   - 台股 1 日有排序能力，但 Kronos 沒有勝過 PatchTST。前瞻確認時兩者一起比較，不預設採用 Kronos。
   - 只有 H1、H2 都通過才加跑 seed 43、44，因此不加跑。
   - 台股長天期不列為候選訊號。
3. **Kronos 與 PatchTST 幾乎相同。**
   - test 上，台股與 US 共六格的配對差都在 −0.002 到 −0.007 之間，區間全部跨 0。
   - validation 上 Kronos 在台股 1 日曾領先 0.0160 [0.0040, 0.0278]，到 test 消失。
4. **台股 1 日的排序能力是所有學習型模型共有的**，不是 Kronos 特有。test 上 GBDT（0.0717）、GRU（0.0711）的點估計略高於 Kronos。
5. **台股長天期在 validation 與 test 之間反轉。**
   - 學習型模型從 validation 的 0.043–0.076 降到 test 的 −0.037 到 0.012。
   - 均線交叉反而從 0.030 升到 0.063 [0.014, 0.118]。
   - 這和 test 期間偏向動能的市場狀態一致，但只有一個 test 期間，無法和模型缺陷區分。
6. **H1 通過不等於能交易。** 1 日標籤是 t+1 開盤到 t+1 收盤，持有這段就是當沖。§13.4、§13.5 的回測顯示，台股前 K 名的 1 日毛優勢約每日 20–31 bps（上市），低於主要成本情境下約 52 bps 的來回成本，而且持有過夜會吐回。能否使用要由 hermes 在含成本的回測中判斷。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 模型 | `a-rank64` seed 42，run `run-20261021T064937Z-c1cf693ab51744a6`；RTX 5090 訓練，約 1.4 epoch 時 early stop |
| 選出的 checkpoint | `checkpoint-003680`（第一次 validation，約 0.2 epoch），validation 市場等權 Rank IC 0.0601；之後的 validation 沒有再改善 |
| 匯出 | 2026-10-21 在 RTX 4090 以 CUDA bf16 重新推論完整 validation 與 test（`export-20261021T123443Z-9312ab57`，約 31 分鐘）；重算的 validation 選點分數與訓練紀錄差 2.9×10⁻⁶ |
| 比較模型 | §13.10 的 baseline `baseline-5fb3473a…`：PatchTST（主要比較）、GBDT、GRU、DLinear、均線交叉（seed 42） |
| 樣本 | validation 112 個訊號日；test 台股 103 日、US 110 日 |
| 指標 | 每個（市場、horizon、訊號日）的 Spearman Rank IC，以原報酬計算 |
| 分組 | horizon：短（h = 1）、中（h = 2–5）、長（h = 6–14）。台股的每日值為 TWSE 與 TPEX 等權 |
| 區間 | 每日值序列的 circular moving-block bootstrap，1,000 次、seed 11；區塊長度短 5、中 10、長 28 個交易日 |
| 配對比較 | 同一訊號日 Kronos 減比較模型的每日值，以相同方式計算區間 |
| 設定檔 | `backtest/config/rank_pilot.json`，在計分前寫入（`97fa8a5`） |

### 3. 結果

#### 3.1 判定（test）

| 假設 | 內容 | 平均 [95% 區間] | 結果 |
| --- | --- | --- | --- |
| H1 | Kronos 台股短組大於 0 | 0.0642 [0.0271, 0.1015] | 通過 |
| H2 | Kronos 減 PatchTST，台股短組大於 0 | −0.0022 [−0.0211, 0.0182] | 未通過 |
| H3 | Kronos 台股長組大於 0 | 0.0082 [−0.0440, 0.0637] | 未通過 |

台股短組由 TWSE 0.0807 與 TPEX 0.0478 平均而成。

#### 3.2 台股各組 Rank IC（test，[95% 區間]）

| 模型 | 短（1 日） | 中（2–5 日） | 長（6–14 日） |
| --- | --- | --- | --- |
| Kronos | 0.0642 [0.0271, 0.1015] | 0.0328 [−0.0153, 0.0835] | 0.0082 [−0.0440, 0.0637] |
| PatchTST | 0.0664 [0.0399, 0.0907] | 0.0393 [0.0017, 0.0803] | 0.0119 [−0.0259, 0.0497] |
| GBDT | 0.0717 [0.0396, 0.1028] | 0.0252 [−0.0110, 0.0635] | −0.0125 [−0.0447, 0.0185] |
| GRU | 0.0711 [0.0356, 0.1053] | 0.0225 [−0.0240, 0.0705] | −0.0369 [−0.0850, 0.0100] |
| DLinear | 0.0410 [0.0303, 0.0528] | 0.0113 [−0.0161, 0.0377] | −0.0351 [−0.0684, −0.0063] |
| 均線交叉 | 0.0092 [−0.0083, 0.0276] | 0.0317 [0.0052, 0.0613] | 0.0629 [0.0136, 0.1177] |

#### 3.3 Kronos 減比較模型的配對差（台股，[95% 區間]）

| 比較模型 | 樣本 | 短 | 中 | 長 |
| --- | --- | --- | --- | --- |
| PatchTST | validation | +0.0160 [0.0040, 0.0278] | +0.0053 [−0.0062, 0.0168] | +0.0023 [−0.0063, 0.0110] |
| PatchTST | test | −0.0022 [−0.0211, 0.0182] | −0.0065 [−0.0226, 0.0104] | −0.0037 [−0.0306, 0.0281] |
| GBDT | test | −0.0075 [−0.0211, 0.0069] | +0.0076 [−0.0175, 0.0387] | +0.0207 [−0.0224, 0.0523] |
| GRU | test | −0.0069 [−0.0167, 0.0028] | +0.0103 [−0.0014, 0.0237] | +0.0451 [0.0206, 0.0692] |
| DLinear | test | +0.0233 [−0.0156, 0.0632] | +0.0215 [−0.0060, 0.0525] | +0.0433 [−0.0008, 0.0832] |
| 均線交叉 | test | +0.0550 [0.0012, 0.1093] | +0.0011 [−0.0677, 0.0710] | −0.0547 [−0.1336, 0.0237] |

只有 PatchTST 是事先指定的主要比較；其他列只作描述。Kronos 台股長組的 test 值高於 GRU（+0.0451），但沒有勝過 PatchTST。

#### 3.4 validation 與 test 對照（台股）

| 模型 | 短：validation → test | 長：validation → test |
| --- | --- | --- |
| Kronos | 0.0984 → 0.0642 | 0.0760 → 0.0082 |
| PatchTST | 0.0824 → 0.0664 | 0.0738 → 0.0119 |
| GBDT | 0.1019 → 0.0717 | 0.0452 → −0.0125 |
| GRU | 0.1082 → 0.0711 | 0.0635 → −0.0369 |
| DLinear | 0.0334 → 0.0410 | 0.0434 → −0.0351 |
| 均線交叉 | −0.0007 → 0.0092 | 0.0303 → 0.0629 |

#### 3.5 US（只作描述）

- Kronos test：短 0.0217 [−0.0101, 0.0569]、中 0.0360 [−0.0168, 0.0929]、長 0.0618 [−0.0102, 0.1223]；區間都跨 0。
- Kronos 減 PatchTST（test）：短 −0.0041、中 −0.0042、長 −0.0050，區間都跨 0。
- US 的 Rank IC 隨 horizon 上升（h = 1 為 0.022，h = 14 為 0.066），和台股相反。

#### 3.6 一致性

- Kronos 匯出與 baseline 的列順序雜湊相同（validation、test 都核對）。
- 計分程式重算的各模型、各市場、各 horizon 平均 Rank IC，與各模型自己的紀錄（baseline 的 `complete.json`、`validation-metrics.json`，Kronos 的 `split.json`）共比對 504 項，最大差距 2.8×10⁻¹⁷。

### 4. 意涵

- Kronos 在這個設定下沒有換到比 PatchTST 更好的排序，卻有高得多的訓練成本。下一步不宜直接加跑 Kronos 的 seed 或變體。
- 台股 1 日排序是所有學習型模型共有的訊號。前瞻確認時應同時比較 Kronos 與 PatchTST（§13.11 第 6 點），GBDT、GRU 可作描述。
- 台股長天期在 test 期間對所有學習型模型失效，均線交叉則變強。資料範圍（美台混合、只用台股、拉長年份）是否影響這一點，可以先用便宜的 baseline 檢驗；§13.12 已起草，尚待確認。
- 日期混洗（`a-rank64-datemix`）依 §13.11 在本結果之後另行登記，目前沒有登記。

### 5. 限制

- test 在 §13.10 已看過，分組在看過結果後決定，所以 test 的判定只是試行。
- 台股 test 只有 103 個訊號日；h 日標籤互相重疊，區塊 bootstrap 只部分處理自相關。
- 只有一個 seed；§13.10 顯示模型間 0.01–0.02 的差距常在雜訊範圍內。
- 選出的是第一次 validation 的 checkpoint（約 0.2 epoch）。之後的 checkpoint 沒有匯出，無法判斷是訓練不足還是很早就過擬合。
- 匯出在 RTX 4090、訓練在 RTX 5090；兩者的 validation 選點分數差 2.9×10⁻⁶，不影響結論。
- 均線交叉在 test 最佳是 §13.10 之後的事後觀察，相關比較只作描述。

### 6. 重現方式

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --rank-baseline-arrays
bash scripts/runpod_workflow.sh download-signal-predictions "run-20261021T064937Z-c1cf693ab51744a6"
cd backtest
uv run python -m twbacktest.rank_pilot --kronos-run "run-20261021T064937Z-c1cf693ab51744a6"
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `64df916`（§13.11） |
| 程式 commit | `f5bdf1d`（Kronos rank 訓練）、`fe8fecf`（rank 標籤讀取）、`a732e58`（baseline 檢查）、`aad2d63`（計分） |
| 設定檔 | `backtest/config/rank_pilot.json`（`97fa8a5`） |
| 匯出（不受 Git 追蹤） | `artifacts/diagnostics/signal-predictions/run-20261021T064937Z-c1cf693ab51744a6/checkpoint-003680/export-20261021T123443Z-9312ab57/` |
| baseline 陣列（不受 Git 追蹤） | `artifacts/audit/baselines/baseline-5fb3473a…/` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/rank-pilot-20261021T130741Z-937a2631/`（`report.json`、`summary.md`、`cells.csv`、`pairs.csv`、`per_horizon.csv`、`daily.csv`） |

## English

### 1. Purpose and conclusions

This report records the decisions of [experiment protocol §13.11](experiment_protocol.md): the Kronos cross-sectional rank model `a-rank64` (seed 42) against the §13.10 rank baselines on Taiwan ranking skill. Test decides; validation only selects the checkpoint. The §13.11 groups were chosen after seeing the §13.10 test results, so these decisions are a pilot; final confirmation uses data after 2026-10-20.

Conclusions:

1. **Decisions: H1 passes; H2 and H3 fail.**
   - H1 (Taiwan 1-session Rank IC above 0): 0.0642 [0.0271, 0.1015], pass.
   - H2 (Taiwan 1 session beats PatchTST): paired difference −0.0022 [−0.0211, 0.0182], fail.
   - H3 (Taiwan 6–14 session Rank IC above 0): 0.0082 [−0.0440, 0.0637], fail.
2. **Disposition under §13.11 item 6:**
   - Taiwan 1-session ranking exists, but Kronos does not beat PatchTST. The forward confirmation compares the two, with no default preference for Kronos.
   - Seeds 43 and 44 run only when H1 and H2 both pass, so they do not run.
   - Taiwan long horizons are not candidate signals.
3. **Kronos and PatchTST are nearly identical.**
   - On test, the paired differences in all six Taiwan and US cells lie between −0.002 and −0.007, and every interval crosses 0.
   - On validation, Kronos led in Taiwan 1 session by 0.0160 [0.0040, 0.0278]; the lead is gone on test.
4. **Taiwan 1-session ranking is shared by every learned model**, not specific to Kronos. On test, GBDT (0.0717) and GRU (0.0711) have slightly higher point estimates than Kronos.
5. **Taiwan long horizons reverse between validation and test.**
   - Learned models fall from 0.043–0.076 on validation to −0.037 to 0.012 on test.
   - The moving-average crossover instead rises from 0.030 to 0.063 [0.014, 0.118].
   - This fits a momentum-leaning market during test, but with a single test period it cannot be told apart from a model defect.
6. **Passing H1 does not make the signal tradeable.** The 1-session label runs from the t+1 open to the t+1 close, so holding it is a day trade. The §13.4 and §13.5 backtests found a Taiwan top-K 1-session gross edge of about 20–31 bps per day (TWSE), below the roughly 52 bps round trip of the primary cost scenario, and holding overnight gives it back. Whether it is usable is for hermes to decide in a backtest with costs.

### 2. Data and method

| Item | Content |
| --- | --- |
| Model | `a-rank64` seed 42, run `run-20261021T064937Z-c1cf693ab51744a6`; trained on an RTX 5090, early-stopped at about 1.4 epochs |
| Selected checkpoint | `checkpoint-003680` (the first validation, about 0.2 epoch), validation market-equal Rank IC 0.0601; later validations did not improve |
| Export | On 2026-10-21, full validation and test re-inferred on an RTX 4090 in CUDA bf16 (`export-20261021T123443Z-9312ab57`, about 31 minutes); the recomputed validation selection score differs from the training record by 2.9×10⁻⁶ |
| Comparators | The §13.10 baseline `baseline-5fb3473a…`: PatchTST (primary), GBDT, GRU, DLinear and the moving-average crossover (seed 42) |
| Samples | Validation 112 signal dates; test 103 dates for Taiwan and 110 for US |
| Metric | Spearman Rank IC per (market, horizon, signal date), computed on raw returns |
| Groups | Horizons: short (h = 1), medium (h = 2–5), long (h = 6–14). Taiwan's daily value weights TWSE and TPEX equally |
| Intervals | Circular moving-block bootstrap of the daily series, 1,000 replicates, seed 11; block length 5 sessions for short, 10 for medium, 28 for long |
| Paired comparisons | Kronos's daily value minus the comparator's on the same signal date, with intervals computed the same way |
| Settings | `backtest/config/rank_pilot.json`, written before scoring (`97fa8a5`) |

### 3. Results

#### 3.1 Decisions (test)

| Hypothesis | Content | Mean [95% CI] | Result |
| --- | --- | --- | --- |
| H1 | Kronos Taiwan short cell above 0 | 0.0642 [0.0271, 0.1015] | Pass |
| H2 | Kronos minus PatchTST, Taiwan short cell above 0 | −0.0022 [−0.0211, 0.0182] | Fail |
| H3 | Kronos Taiwan long cell above 0 | 0.0082 [−0.0440, 0.0637] | Fail |

The Taiwan short cell averages TWSE 0.0807 and TPEX 0.0478.

#### 3.2 Taiwan Rank IC by group (test, [95% CI])

| Model | Short (1 session) | Medium (2–5) | Long (6–14) |
| --- | --- | --- | --- |
| Kronos | 0.0642 [0.0271, 0.1015] | 0.0328 [−0.0153, 0.0835] | 0.0082 [−0.0440, 0.0637] |
| PatchTST | 0.0664 [0.0399, 0.0907] | 0.0393 [0.0017, 0.0803] | 0.0119 [−0.0259, 0.0497] |
| GBDT | 0.0717 [0.0396, 0.1028] | 0.0252 [−0.0110, 0.0635] | −0.0125 [−0.0447, 0.0185] |
| GRU | 0.0711 [0.0356, 0.1053] | 0.0225 [−0.0240, 0.0705] | −0.0369 [−0.0850, 0.0100] |
| DLinear | 0.0410 [0.0303, 0.0528] | 0.0113 [−0.0161, 0.0377] | −0.0351 [−0.0684, −0.0063] |
| Moving-average crossover | 0.0092 [−0.0083, 0.0276] | 0.0317 [0.0052, 0.0613] | 0.0629 [0.0136, 0.1177] |

#### 3.3 Kronos minus comparator, paired (Taiwan, [95% CI])

| Comparator | Sample | Short | Medium | Long |
| --- | --- | --- | --- | --- |
| PatchTST | validation | +0.0160 [0.0040, 0.0278] | +0.0053 [−0.0062, 0.0168] | +0.0023 [−0.0063, 0.0110] |
| PatchTST | test | −0.0022 [−0.0211, 0.0182] | −0.0065 [−0.0226, 0.0104] | −0.0037 [−0.0306, 0.0281] |
| GBDT | test | −0.0075 [−0.0211, 0.0069] | +0.0076 [−0.0175, 0.0387] | +0.0207 [−0.0224, 0.0523] |
| GRU | test | −0.0069 [−0.0167, 0.0028] | +0.0103 [−0.0014, 0.0237] | +0.0451 [0.0206, 0.0692] |
| DLinear | test | +0.0233 [−0.0156, 0.0632] | +0.0215 [−0.0060, 0.0525] | +0.0433 [−0.0008, 0.0832] |
| Moving-average crossover | test | +0.0550 [0.0012, 0.1093] | +0.0011 [−0.0677, 0.0710] | −0.0547 [−0.1336, 0.0237] |

Only PatchTST was fixed beforehand as the primary comparator; the other rows are descriptive. Kronos's Taiwan long cell is above GRU's on test (+0.0451), but it does not beat PatchTST.

#### 3.4 Validation against test (Taiwan)

| Model | Short: validation → test | Long: validation → test |
| --- | --- | --- |
| Kronos | 0.0984 → 0.0642 | 0.0760 → 0.0082 |
| PatchTST | 0.0824 → 0.0664 | 0.0738 → 0.0119 |
| GBDT | 0.1019 → 0.0717 | 0.0452 → −0.0125 |
| GRU | 0.1082 → 0.0711 | 0.0635 → −0.0369 |
| DLinear | 0.0334 → 0.0410 | 0.0434 → −0.0351 |
| Moving-average crossover | −0.0007 → 0.0092 | 0.0303 → 0.0629 |

#### 3.5 US (descriptive)

- Kronos on test: short 0.0217 [−0.0101, 0.0569], medium 0.0360 [−0.0168, 0.0929], long 0.0618 [−0.0102, 0.1223]; every interval crosses 0.
- Kronos minus PatchTST (test): short −0.0041, medium −0.0042, long −0.0050, every interval crossing 0.
- US Rank IC rises with the horizon (0.022 at h = 1, 0.066 at h = 14), the opposite of Taiwan.

#### 3.6 Consistency

- The Kronos export and the baselines share the same row-order digest (checked for validation and test).
- The scorer's recomputed mean Rank IC per model, market and horizon matches each model's own record (the baselines' `complete.json` and `validation-metrics.json`, Kronos's `split.json`) on all 504 values, with a largest difference of 2.8×10⁻¹⁷.

### 4. Implications

- In this setup Kronos buys no better ranking than PatchTST at a much higher training cost. More Kronos seeds or variants are not the right next step.
- Taiwan 1-session ranking is a signal shared by every learned model. The forward confirmation should compare Kronos and PatchTST (§13.11 item 6), with GBDT and GRU as descriptions.
- Taiwan long horizons failed for every learned model during test, while the moving-average crossover grew stronger. Whether the data scope (pooled, Taiwan only, longer history) affects this can be tested first with cheap baselines; §13.12 is drafted and awaits confirmation.
- Date mixing (`a-rank64-datemix`) is registered separately after these results under §13.11; it is not registered yet.

### 5. Limits

- Test was seen in §13.10 and the groups were chosen after those results, so the test decisions are only a pilot.
- Taiwan test has only 103 signal dates; h-session labels overlap, and the block bootstrap only partly handles the autocorrelation.
- One seed only; §13.10 shows that 0.01–0.02 differences between models are often within noise.
- The selected checkpoint is the first validation (about 0.2 epoch). Later checkpoints were not exported, so it is unknown whether the model is undertrained or overfits very early.
- The export ran on an RTX 4090 and training on an RTX 5090; their validation selection scores differ by 2.9×10⁻⁶, which does not affect the conclusions.
- The moving-average crossover being best on test is a post hoc observation from §13.10; comparisons with it are descriptive.

### 6. Reproduction

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --rank-baseline-arrays
bash scripts/runpod_workflow.sh download-signal-predictions "run-20261021T064937Z-c1cf693ab51744a6"
cd backtest
uv run python -m twbacktest.rank_pilot --kronos-run "run-20261021T064937Z-c1cf693ab51744a6"
```

| Item | Content |
| --- | --- |
| Pre-registration commit | `64df916` (§13.11) |
| Code commits | `f5bdf1d` (Kronos rank training), `fe8fecf` (rank-label reads), `a732e58` (baseline check), `aad2d63` (scoring) |
| Settings | `backtest/config/rank_pilot.json` (`97fa8a5`) |
| Export (not tracked by Git) | `artifacts/diagnostics/signal-predictions/run-20261021T064937Z-c1cf693ab51744a6/checkpoint-003680/export-20261021T123443Z-9312ab57/` |
| Baseline arrays (not tracked by Git) | `artifacts/audit/baselines/baseline-5fb3473a…/` |
| Output (not tracked by Git) | `artifacts/diagnostics/backtest/rank-pilot-20261021T130741Z-937a2631/` (`report.json`, `summary.md`, `cells.csv`, `pairs.csv`, `per_horizon.csv`, `daily.csv`) |
