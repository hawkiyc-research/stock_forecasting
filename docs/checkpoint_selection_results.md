# checkpoint 選點結果 / Checkpoint Selection Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.7](experiment_protocol.md) 及其修訂的 P2 選點：三個 alpha run（R21–R23）各有 5 個保留的 checkpoint，每個 run 選出 validation 上市 10 日前十分位開盤到開盤超額最高的一個。只用 validation 選點；test 只計算選出的 3 個，只作描述。

結論：

1. **15 份匯出全部通過完整性檢查。** 列順序雜湊與 baseline 一致；validation 重算的選模分數與 `metrics.jsonl` 同一步的紀錄一致（差距 ≤ 10⁻⁶）。
2. **三個 run 都選中 checkpoint-018399。**
   - R22 選中的就是 R22-IR，§13.8 主要假說的模型不變。
   - R23 選中的就是原本 validation 最佳的 checkpoint。
   - R21 由 R21-IR（022079，+61.4 bps）改為 018399（+76.0 bps），兩者的區間大幅重疊。
3. **選出者的 test（只作描述）：**
   - 上市 10 日前十分位超額：R22 +103.2 bps [+57.3, +147.8]，R23 +86.4 bps [+30.6, +127.8]，R21 −63.9 bps [−167.2, +37.7]。
   - 固定組合年化淨超額：R22 +42.8% [+10.7%, +74.6%]，R23 +11.7%，R21 −25.8%。
4. **test 期間，前、後十分位都高於等權平均。** 三者的後十分位超額都約 +154 到 +169 bps，所以前後十分位差在 test 都為負。前十分位超額有一部分可能來自兩端股票的共同表現，不全是排序能力。validation 沒有這個現象（例如 R22 的前後差為 +105.2 bps）。
5. **重新匯出可以重現。** 同一 checkpoint 在 10/18（P0）與 10/19 的兩次匯出，算出的 §13.6 指標完全相同（差距 0）。
6. **以上都不是確認。** 每個 run 都是從 5 個候選中選出，validation 數值偏樂觀；test 在 §13.6 已看過。確認只能靠 §13.8 的前瞻資料。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 推論 | RunPod RTX 5090，CUDA bf16（`export-signal-predictions`）；15 個 checkpoint 都在 2026-10-19 於同一種 GPU 匯出 |
| 匯出過程 | 第一次工作在第 4 個 checkpoint 崩潰（core dump）；修正 `e93705c` 後以 `--reuse-complete-since 20261019` 續跑，前 3 個在重新核對 SHA-256 後沿用 |
| 完整性 | 每份匯出的列順序雜湊須與 baseline 一致；validation `aggregate.selection_score` 須與 `metrics.jsonl` 同一步一致（≤ 10⁻⁶） |
| 記帳恆等式 | validation 96,166 組、test 93,402 組相鄰資料列全部成立（股票與指數） |
| 選點規則 | validation、上市、h = 10、前十分位開盤到開盤超額最高者；同分時選步數較晚者 |
| 指標與區間 | 與 §13.6 相同：前十分位減同日同市場等權平均；circular moving-block bootstrap 1,000 次、seed 11 |
| 固定組合 | s = h、K = 20、b = 2、折扣 0.6、滑價 1 倍、只做多 |

### 3. 結果

#### 3.1 validation 上市 10 日前十分位超額（bps，[95% 區間]）

| run | checkpoint | validation | 選出 |
| --- | ---: | --- | :---: |
| R21 | 003680 | −84.8 [−183.8, −3.3] | |
| R21 | 007360 | −43.3 [−111.5, +22.7] | |
| R21 | 011040 | −7.8 [−90.1, +72.5] | |
| R21 | 018399 | +76.0 [+7.8, +145.8] | ✓ |
| R21 | 022079 | +61.4 [+7.7, +123.2] | |
| R22 | 003680 | −82.5 [−163.0, −14.0] | |
| R22 | 007360 | −50.7 [−114.5, −0.7] | |
| R22 | 011040 | −37.2 [−141.9, +63.6] | |
| R22 | 014720 | −35.2 [−140.0, +62.4] | |
| R22 | 018399 | +69.6 [+8.6, +139.0] | ✓ |
| R23 | 003680 | −62.9 [−161.3, +27.8] | |
| R23 | 007360 | −68.1 [−148.7, +9.8] | |
| R23 | 018399 | +35.7 [−18.1, +108.2] | ✓ |
| R23 | 022079 | −50.7 [−125.9, +22.1] | |
| R23 | 025759 | −53.2 [−137.3, +24.3] | |

三個 run 中，較早的 checkpoint（003680、007360）的 validation 都為負，選中的都是訓練較後期的 018399。

#### 3.2 選出者：上市 10 日（bps）

| 指標 | R21-018399 validation | R21-018399 test | R22-018399 validation | R22-018399 test | R23-018399 validation | R23-018399 test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 前十分位超額 | +76.0 | −63.9 | +69.6 | +103.2 | +35.7 | +86.4 |
| 後十分位超額 | −1.2 | +165.8 | −35.6 | +168.9 | +13.1 | +154.3 |
| 前後十分位差 | +77.2 | −229.7 | +105.2 | −65.7 | +22.6 | −67.9 |
| 交易時段 / 隔夜 | +173.3 / −88.6 | +158.3 / −193.4 | +78.5 / −10.7 | +92.5 / +18.6 | +113.3 / −73.1 | +143.4 / −47.5 |
| 開盤到開盤 Rank IC（IR） | 0.066（0.52） | 0.016（0.10） | 0.036（0.32） | 0.071（0.58） | 0.040（0.32） | 0.042（0.27） |

#### 3.3 選出者：上市固定組合（年化淨超額 [95% 區間]）

| run | h | validation | test | test 每日換手 |
| --- | ---: | --- | --- | ---: |
| R21-018399 | 10 | +13.6% [−14.9%, +43.0%] | −25.8% [−59.1%, +5.8%] | 6.5% |
| R21-018399 | 5 | +4.2% [−25.0%, +29.9%] | −26.5% [−61.1%, +8.3%] | 5.3% |
| R22-018399 | 10 | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | 6.9% |
| R22-018399 | 5 | +7.2% [−18.5%, +36.5%] | +30.1% [+2.6%, +58.3%] | 6.4% |
| R23-018399 | 10 | −3.0% [−26.1%, +19.8%] | +11.7% [−25.2%, +46.0%] | 4.0% |
| R23-018399 | 5 | +3.3% [−25.4%, +33.2%] | +2.3% [−39.2%, +43.1%] | 3.8% |

#### 3.4 重新匯出的一致性

P0（2026-10-18）已匯出其中 5 個 checkpoint：R21 的 003680、022079；R22 的 011040、018399；R23 的 018399。以 10/19 的匯出重算後，與 [§13.6 報告](holding_eval_results.md)的輸出逐欄比對：

- 5 個 checkpoint 的 validation 排序指標與固定組合，差距都是 0。
- R22-018399 與 R23-018399 的 test 也是 0；其他 3 個這次不計算 test。

### 4. 意涵

- 以持有指標選點，沒有改變 §13.8 的主要假說：R22 本來就選到 R22-IR（checkpoint-018399）。
- 三個 run 都選到同一個訓練階段，而且早期 checkpoint 的 validation 都為負。這和 §13.6 的觀察一致：訓練較後期的 checkpoint 隔夜拖累較小。原因仍未檢驗。
- test 期間前、後十分位都勝過等權平均，所以只做多的前十分位超額，可能高估了排序能力。§13.8 判定時應同時列出後十分位與前後差（只作描述，不改判準）。
- 下一步是 §13.8 的前瞻檢驗，需要兩項尚未實作的工作：
  1. 在 RunPod CPU 準備含 `forward` 分割的新資料組。
  2. 匯出程式支援 `forward` 分割。

### 5. 限制

- 每個 run 從 5 個候選中選最高者，validation 數值偏樂觀；validation 約 5 個月（112 日）。
- test 在 §13.6 已看過 R22-IR 與 R23，不是獨立的確認。
- 後十分位在 test 為正，表示等權平均可能被中間股票拉低；這段期間的分布與 validation 不同。
- 未模擬成交量與流動性、每筆最低手續費、整股、股利稅；存活者偏誤同 §13.4。
- 評分在本機沙箱以單一 worker 執行。

### 6. 重現方式

```bash
cd backtest
uv run python -m twbacktest.selection --check --training-records <主 checkout>/artifacts/runpod
uv run python -m twbacktest.selection --training-records <主 checkout>/artifacts/runpod
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `e4d823c`（§13.7）、`1197b60`（§13.7 修訂） |
| 程式 commit | `2d84ff9`（選點）、`e93705c`（匯出修正） |
| 設定檔 | `backtest/config/checkpoint_selection.json` |
| 輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/selection-20261020T042715Z-2e0ff319/`（`report.json`、`ranking.csv`、`portfolio.csv`、`summary.md`） |

## English

### 1. Purpose and conclusions

This report records the P2 checkpoint selection of [experiment protocol §13.7](experiment_protocol.md) and its amendment. Each of three alpha runs (R21–R23) retains 5 checkpoints, and each run keeps the one with the highest validation TWSE 10-session top-decile open-to-open excess. Selection uses validation only; test is computed for the 3 selected checkpoints only and is descriptive.

Conclusions:

1. **All 15 exports passed the integrity checks.** Row-order digests match the baseline, and the recomputed validation selection score matches `metrics.jsonl` at the same step (within 10⁻⁶).
2. **All three runs selected checkpoint-018399.**
   - R22's selection is R22-IR, so the model of the §13.8 primary hypothesis is unchanged.
   - R23's selection is its existing validation-best checkpoint.
   - R21 moves from R21-IR (022079, +61.4 bps) to 018399 (+76.0 bps); their intervals overlap heavily.
3. **Test for the selected checkpoints (descriptive only):**
   - TWSE 10-session top-decile excess: R22 +103.2 bps [+57.3, +147.8], R23 +86.4 bps [+30.6, +127.8], R21 −63.9 bps [−167.2, +37.7].
   - Fixed-portfolio annual net excess: R22 +42.8% [+10.7%, +74.6%], R23 +11.7%, R21 −25.8%.
4. **On test, both the top and the bottom decile beat the equal-weight average.** The bottom-decile excess is about +154 to +169 bps for all three, so the top-minus-bottom spread is negative on test. Part of the top-decile excess may come from stocks at both extremes rather than from ranking skill. Validation does not show this (R22's spread is +105.2 bps, for example).
5. **Re-exports reproduce.** Checkpoints exported both on 10/18 (P0) and on 10/19 give identical §13.6 metrics (difference 0).
6. **None of this is confirmation.** Each run's winner is the best of 5, so its validation value is optimistic, and test was already seen in §13.6. Only the §13.8 forward data can confirm.

### 2. Data and method

| Item | Detail |
| --- | --- |
| Inference | RunPod RTX 5090, CUDA bf16 (`export-signal-predictions`); all 15 checkpoints exported on 2026-10-19 on the same GPU type |
| Export run | The first job crashed (core dump) on its fourth checkpoint; after fix `e93705c` it resumed with `--reuse-complete-since 20261019`, reusing the first 3 after re-verifying their SHA-256 |
| Integrity | Each export's row-order digest must match the baseline; validation `aggregate.selection_score` must match `metrics.jsonl` at the same step (≤ 10⁻⁶) |
| Accounting identity | All 96,166 validation and 93,402 test adjacent row pairs hold (stocks and index) |
| Selection rule | Validation, TWSE, h = 10, highest top-decile open-to-open excess; ties go to the later step |
| Metrics and intervals | As in §13.6: top decile minus the same-day, same-market equal-weight average; circular moving-block bootstrap, 1,000 replicates, seed 11 |
| Fixed portfolio | s = h, K = 20, b = 2, discount 0.6, slippage ×1, long only |

### 3. Results

#### 3.1 Validation TWSE 10-session top-decile excess (bps, [95% CI])

| Run | Checkpoint | Validation | Selected |
| --- | ---: | --- | :---: |
| R21 | 003680 | −84.8 [−183.8, −3.3] | |
| R21 | 007360 | −43.3 [−111.5, +22.7] | |
| R21 | 011040 | −7.8 [−90.1, +72.5] | |
| R21 | 018399 | +76.0 [+7.8, +145.8] | ✓ |
| R21 | 022079 | +61.4 [+7.7, +123.2] | |
| R22 | 003680 | −82.5 [−163.0, −14.0] | |
| R22 | 007360 | −50.7 [−114.5, −0.7] | |
| R22 | 011040 | −37.2 [−141.9, +63.6] | |
| R22 | 014720 | −35.2 [−140.0, +62.4] | |
| R22 | 018399 | +69.6 [+8.6, +139.0] | ✓ |
| R23 | 003680 | −62.9 [−161.3, +27.8] | |
| R23 | 007360 | −68.1 [−148.7, +9.8] | |
| R23 | 018399 | +35.7 [−18.1, +108.2] | ✓ |
| R23 | 022079 | −50.7 [−125.9, +22.1] | |
| R23 | 025759 | −53.2 [−137.3, +24.3] | |

In all three runs the early checkpoints (003680, 007360) are negative on validation, and the selection is the later-training 018399.

#### 3.2 Selected checkpoints: TWSE, 10 sessions (bps)

| Metric | R21-018399 validation | R21-018399 test | R22-018399 validation | R22-018399 test | R23-018399 validation | R23-018399 test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-decile excess | +76.0 | −63.9 | +69.6 | +103.2 | +35.7 | +86.4 |
| Bottom-decile excess | −1.2 | +165.8 | −35.6 | +168.9 | +13.1 | +154.3 |
| Top minus bottom | +77.2 | −229.7 | +105.2 | −65.7 | +22.6 | −67.9 |
| Session / overnight | +173.3 / −88.6 | +158.3 / −193.4 | +78.5 / −10.7 | +92.5 / +18.6 | +113.3 / −73.1 | +143.4 / −47.5 |
| Open-to-open Rank IC (IR) | 0.066 (0.52) | 0.016 (0.10) | 0.036 (0.32) | 0.071 (0.58) | 0.040 (0.32) | 0.042 (0.27) |

#### 3.3 Selected checkpoints: TWSE fixed portfolio (annual net excess [95% CI])

| Run | h | Validation | Test | Test daily turnover |
| --- | ---: | --- | --- | ---: |
| R21-018399 | 10 | +13.6% [−14.9%, +43.0%] | −25.8% [−59.1%, +5.8%] | 6.5% |
| R21-018399 | 5 | +4.2% [−25.0%, +29.9%] | −26.5% [−61.1%, +8.3%] | 5.3% |
| R22-018399 | 10 | +11.3% [−17.3%, +43.7%] | +42.8% [+10.7%, +74.6%] | 6.9% |
| R22-018399 | 5 | +7.2% [−18.5%, +36.5%] | +30.1% [+2.6%, +58.3%] | 6.4% |
| R23-018399 | 10 | −3.0% [−26.1%, +19.8%] | +11.7% [−25.2%, +46.0%] | 4.0% |
| R23-018399 | 5 | +3.3% [−25.4%, +33.2%] | +2.3% [−39.2%, +43.1%] | 3.8% |

#### 3.4 Re-export consistency

P0 (2026-10-18) had already exported 5 of the checkpoints: R21 003680 and 022079, R22 011040 and 018399, and R23 018399. Recomputed from the 10/19 exports and compared column by column with the [§13.6 report](holding_eval_results.md) outputs:

- All validation ranking metrics and fixed portfolios of the 5 checkpoints differ by 0.
- Test also differs by 0 for R22-018399 and R23-018399; test is not computed this time for the other 3.

### 4. Implications

- Selecting by a holding metric leaves the §13.8 primary hypothesis unchanged: R22 already selected R22-IR (checkpoint-018399).
- All three runs select the same training stage, and their early checkpoints are all negative on validation. This matches the §13.6 observation that later-training checkpoints drag less overnight; the cause is still untested.
- On test both deciles beat the equal-weight average, so long-only top-decile excess may overstate ranking skill. The §13.8 report should list the bottom decile and the spread next to the pre-registered criteria (descriptive only; the criteria do not change).
- The next step is the §13.8 forward test, which needs two pieces not yet implemented:
  1. Prepare a new dataset with a `forward` split on RunPod CPU.
  2. Extend the export to the `forward` split.

### 5. Limits

- Each run's winner is the best of 5 candidates, so validation is optimistic; validation covers about 5 months (112 sessions).
- Test already showed R22-IR and R23 in §13.6, so it is not an independent confirmation.
- The bottom decile is positive on test, so the equal-weight average may be pulled down by mid-ranked stocks; the distribution in this period differs from validation.
- Volume and liquidity, minimum commission per trade, board lots and dividend tax are not simulated; survivorship bias as in §13.4.
- Scoring ran in the local sandbox with one worker.

### 6. Reproduction

```bash
cd backtest
uv run python -m twbacktest.selection --check --training-records <main checkout>/artifacts/runpod
uv run python -m twbacktest.selection --training-records <main checkout>/artifacts/runpod
```

| Item | Detail |
| --- | --- |
| Pre-registration commits | `e4d823c` (§13.7), `1197b60` (§13.7 amendment) |
| Code commits | `2d84ff9` (selection), `e93705c` (export fix) |
| Configuration | `backtest/config/checkpoint_selection.json` |
| Outputs (not tracked by Git) | `artifacts/diagnostics/backtest/selection-20261020T042715Z-2e0ff319/` (`report.json`, `ranking.csv`, `portfolio.csv`, `summary.md`) |
