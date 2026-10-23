# B0 LightGBM 基線與回測結果 / B0 LightGBM Baseline and Backtest Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.13](experiment_protocol.md) 第 2 步 B0 `t-lgbm-alpha` 的 walk-forward 結果，以及第 6 步 Top-k dropout 回測（H-B）套用在 B0 上的判定。D0 `t-xs-scratch`（H-A）另行報告。

結論：

1. **B0 有穩定的排序能力。** P1 ∪ P2 共 777 個訊號日，短格（h = 1）Rank IC 0.1178 [0.1091, 0.1266]、ICIR 0.85，79.5% 的交易日為正；持有格 0.0859 [0.0650, 0.1061]；長格 0.0834 [0.0541, 0.1129]。三格的區間下界都大於 0。
2. **P2 明顯弱於 P1，長天期在 P2 接近零。** 短格由 P1 的 0.1248 降到 P2 的 0.0996；持有格由 0.1057 降到 0.0343 [−0.0004, 0.0726]；長格由 0.1112 降到 0.0107 [−0.0372, 0.0600]。這與 §13.11 看到的「台股長天期在近期失效」一致。
3. **排序能力隨 horizon 遞減。** TWSE 的平均每日 Rank IC 由 h = 1 的 0.126 降到 h = 5 的 0.091，h = 10 以後約 0.083；TPEX 由 0.109 降到 0.083–0.085。
4. **三個 seed 幾乎相同。** 短格 P1 ∪ P2 三個 seed 分別為 0.1167、0.1174、0.1172，ensemble 0.1178；seed 之間的差距小於 0.001。
5. **H-B 未通過：B0 的排序能力在登記的策略與成本下不可交易。** Top-k dropout（TWSE K = 50、每日換 5 檔）在 P1 選出 h = 10（淨超額 IR 0.014），P1 ∪ P2 主要成本情境的年化淨超額 −6.3% [−19.0%, +7.1%]、IR −0.58。P1 的毛超額 +17.5%／年，但每日換手 10.2% 的成本拖累約 17.3%／年；P2 連毛超額都是負的（三個 horizon 皆在 −5.9% 到 −21.2% 之間）。
6. **次要的持有對齊回測得到相同結論。** 依標籤方式持有（次一開盤買進、h 日後收盤賣出）時，h = 1 前 20 名每批毛超額 +0.26%，低於當沖來回成本 0.52%，年化淨超額 −66%；成本最有利的 h = 10、K = 10 在 P1 為 +5.9% [−6.5%, +19.1%]，P2 為 −10.9% [−31.5%, +11.4%]，P1 ∪ P2 為 +1.3% [−10.0%, +14.0%]。
7. **依 §13.13 第 5 點的處置：** B0 成為策略基準，但不算可交易；H-C 對 B0 不執行；問題回到策略層（換倉頻率、持有天期、成本），不是模型層。D0 仍依第 6 項第 3 步執行，以 H-A 判定 DL 是否勝過 B0；D0 若通過 H-A，再以 D0 對 B0 做 H-B。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 資料集 | §13.12 的 tw_only 2016 資料集 `53e3915b…`（TWSE、TPEX，2016-01 起，2,159 個標的，4,417,007 筆原始列） |
| 特徵與標籤 | `configs/features/alpha_tw_v1.json` 的 183 欄特徵庫；標籤為每個 (market, signal date) 的 gauss-rank alpha，h = 1–14 |
| walk-forward | `configs/walk_forward/tw_quarterly_v1.json`：14 個季度 fold（2023Q1–2026Q2），擴張視窗自 2016-01-01，15 個交易日 embargo，每折訓練視窗的最後 12 個月為 fold validation；訓練列數由 896,789（2023Q1）增至 1,589,755（2026Q2） |
| 模型 | LightGBM regression（`num_leaves` 31、`learning_rate` 0.05、`feature_fraction` 與 `bagging_fraction` 0.8、`min_data_in_leaf` 200、最多 1,000 輪），以 fold validation 的每日 Rank IC 早停（patience 50）；每個 (fold, horizon, seed) 一個模型，共 588 個；平均最佳輪數 97.6，平均 validation Rank IC 0.105 |
| seed 與 ensemble | seed 42、43、44；`t-lgbm-alpha` 為三個 seed 預測的平均名次 |
| 執行 | RunPod CPU Pod `cpu3g` × 16 vCPU（`launch-20261022T234922Z-22048038`）：4 個 worker × 4 執行緒，記憶體預算 23.5 GB；訓練牆鐘 5,766 秒 |
| 評估 | 每個 (market, horizon, signal date) 的 Spearman Rank IC（對原始 alpha）；cell 為 short（h = 1）、holding（h = 5–10）、long（h = 11–14）的平均，每日值為兩市場等權；期間 P1（2023-01 → 2025-05）、P2（2025-06 → 2026-05）與 P1 ∪ P2 |
| 區間 | 每日值序列的 circular moving-block bootstrap，1,000 次、seed 11，區塊 5／20／28 個交易日 |
| Top-k dropout（H-B） | `backtest/config/topk_dropout.json`：每日依前一收盤排序，賣出持股中排名最低的 5 檔（TPEX 3 檔）、買進未持有中排名最高者，次一開盤成交；K = 50（TPEX 30）；horizon 在 {1, 5, 10} 中依 TWSE P1 淨超額 IR 選定；成本與第一階段相同（主情境：手續費折扣 0.6、滑價 ×1）；月份 block（21 日）bootstrap 1,000 次 |
| 持有對齊回測（次要） | `backtest/config/holding_walkforward.json`：前 K 名（10／20／50）於次一開盤買進、h（1／2／3／5／10）日後收盤賣出，h 個起始 sleeve，第一階段引擎；h = 1 課當沖稅；主情境折扣 0.6、滑價 ×1、保守換手 |

主要成本情境下 TWSE 一次來回的成本：h ≥ 2 為 0.671%（手續費 0.0855% × 2、證交稅 0.3%、滑價 0.1% × 2），h = 1 當沖為 0.521%（證交稅減半）。

### 3. 結果

#### 3.1 Rank IC cell（ensemble `t-lgbm-alpha`，[95% 區間]）

| cell | P1（562 日） | P2（215 日） | P1 ∪ P2（777 日） | ICIR（P1 ∪ P2） | 正值比例（P1 ∪ P2） |
| --- | --- | --- | --- | --- | --- |
| short（h = 1） | 0.1248 [0.1154, 0.1341] | 0.0996 [0.0808, 0.1187] | 0.1178 [0.1091, 0.1266] | 0.85 | 79.5% |
| holding（h = 5–10） | 0.1057 [0.0866, 0.1259] | 0.0343 [−0.0004, 0.0726] | 0.0859 [0.0650, 0.1061] | 0.70 | 74.9% |
| long（h = 11–14） | 0.1112 [0.0849, 0.1384] | 0.0107 [−0.0372, 0.0600] | 0.0834 [0.0541, 0.1129] | 0.64 | 71.4% |

#### 3.2 各市場、各 horizon 的平均每日 Rank IC（P1 ∪ P2）

| horizon | TWSE | TPEX |
| --- | --- | --- |
| 1 | 0.126 | 0.109 |
| 2 | 0.107 | 0.099 |
| 3 | 0.100 | 0.096 |
| 5 | 0.091 | 0.091 |
| 7 | 0.085 | 0.086 |
| 10 | 0.084 | 0.083 |
| 14 | 0.083 | 0.085 |

#### 3.3 seed 一致性（short，P1 ∪ P2）

| 模型 | Rank IC | 95% 區間 | ICIR |
| --- | --- | --- | --- |
| seed 42 | 0.1167 | [0.1078, 0.1256] | 0.84 |
| seed 43 | 0.1174 | [0.1089, 0.1261] | 0.86 |
| seed 44 | 0.1172 | [0.1083, 0.1261] | 0.85 |
| ensemble | 0.1178 | [0.1091, 0.1266] | 0.85 |

#### 3.4 Top-k dropout 回測（H-B，主要成本情境，long-only）

| 市場 | 期間 | h | 年化毛超額 | 年化淨超額 | 95% 區間 | IR | 成本拖累／年 | 最大回撤 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TWSE | P1 | 1 | +9.7% | −7.6% | [−20.8%, +7.3%] | −0.68 | 17.3% | −20.1% |
| TWSE | P1 | 5 | +14.2% | −2.8% | [−16.0%, +11.3%] | −0.28 | 17.0% | −20.5% |
| TWSE | P1 | 10 | +17.5% | +0.1% | [−13.5%, +15.0%] | +0.01 | 17.3% | −18.3% |
| TWSE | P2 | 10 | −5.9% | −23.0% | [−42.5%, −3.2%] | −2.08 | 17.1% | −6.2% |
| TWSE | P1 ∪ P2 | 10 | +11.0% | −6.3% | [−19.0%, +7.1%] | −0.58 | 17.3% | −18.3% |
| TPEX | P1 | 10 | +25.3% | +1.2% | [−11.7%, +14.3%] | +0.11 | 24.1% | −31.8% |
| TPEX | P2 | 10 | −29.7% | −54.0% | [−80.0%, −28.3%] | −4.45 | 24.2% | −12.3% |
| TPEX | P1 ∪ P2 | 10 | +10.0% | −14.1% | [−27.6%, +0.0%] | −1.21 | 24.1% | −31.8% |

選定 h = 10（TWSE P1 淨超額 IR 0.014，為三個 horizon 中最高）。判定格 TWSE P1 ∪ P2 的區間下界 −19.0% < 0，H-B 未通過。TWSE 每日換手 10.2%、平均持有 9.6 日；損益兩平的單邊成本為 0.21%，低於主情境的實際單邊成本約 0.34%。P2 的 TWSE 三個 horizon 毛超額都是負的（h = 1 −21.2%、h = 5 −14.7%、h = 10 −5.9%），所以 P2 的失敗不只是成本。hedged 版本（放空等權指數）在每一格都比 long-only 差（TWSE P1 ∪ P2 h = 10 為 −21.7%）。

#### 3.5 持有對齊回測（次要分析，TWSE，主要成本情境）

| h | K | 每批毛超額（P1 ∪ P2） | 年化淨超額 P1 | 年化淨超額 P2 | 年化淨超額 P1 ∪ P2 | 95% 區間（P1 ∪ P2） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | +0.27% | −64.3% | −58.1% | −62.6% | [−76.1%, −47.8%] |
| 1 | 20 | +0.26% | −65.9% | −66.6% | −66.1% | [−77.4%, −54.1%] |
| 5 | 10 | +0.36% | −11.5% | −23.8% | −14.9% | [−24.8%, −3.8%] |
| 5 | 20 | +0.31% | −11.6% | −32.1% | −17.2% | [−25.9%, −7.2%] |
| 10 | 10 | +0.71% | +5.9% | −10.9% | +1.3% | [−10.0%, +14.0%] |
| 10 | 20 | +0.56% | +2.7% | −16.2% | −2.6% | [−12.9%, +9.1%] |
| 10 | 50 | +0.34% | −1.7% | −24.9% | −8.1% | [−17.9%, +2.0%] |

選定（TWSE P1 淨超額 IR 最高）為 h = 10、K = 10，IR 0.46。h = 1 的每批毛超額 0.26% 低於當沖來回成本 0.52%；同一格只有在零滑價加樂觀換手（+23.9%）或折扣 0.28 加零滑價（+7.3%）時淨值為正。毛超額隨 h 增加（K = 10 由 0.27% 到 0.71%），但折算成每日是遞減的，與 3.2 的 horizon 衰減一致。

### 4. 意涵

- 依 §13.13 第 5 點：H-B 未通過代表排序能力存在但不可交易，工作回到策略層而非模型層；H-C 對 B0 不執行。D0 仍要跑，因為 H-A 是「DL 是否勝過 B0」的獨立問題；D0 若通過 H-A，再以 D0 對 B0 做 H-B（D0 的 IR 不低於 B0）。
- B0 的毛優勢是每日 10–13 bps 量級的排序訊號，和第一階段與 §13.11 看到的「1 日有訊號、過夜吐回、台股成本高於毛優勢」一致。換成特徵庫與 walk-forward 之後，Rank IC 由 §13.11 的 0.06–0.08 提高到 0.12，但成本結構沒有變。
- 數字指向的策略層方向（尚未登記，只作紀錄）：降低換手（每日於 50 檔中換 5 檔即每日 10% 換手，成本拖累 17%／年，超過 P1 的毛超額）、拉長持有並集中（h = 10、K = 10 最接近兩平）、以及 P2 的市場狀態（2025-06 之後連毛超額都轉負，需確認是否為動能／反轉狀態切換）。任何新的策略規則都須在執行之前另行登記。
- 三個 seed 幾乎相同，表示 LightGBM 在這組特徵上的變異主要來自資料而非初始化；D0 的三 seed 之間若差距在 0.001 量級，不應視為差異。

### 5. 限制

- P2 已在 §13.10、§13.11 看過，P1 是第一次評估，但資料規則、特徵庫與模型設定都是在看過 P2 的彙總結果後決定的；真正的確認只有 P3（2026-06 以後）。
- 回測未模擬成交量上限、整股、最低手續費與股利稅；漲跌停以外的流動性限制也未模擬；TPEX 的滑價假設（0.2%）仍可能偏低。
- Top-k dropout 的 horizon 在 P1 選定，但 P1 三個 horizon 的淨超額 IR 都接近 0（−0.68 到 +0.01），這個選擇本身幾乎沒有辨識力。
- 持有對齊回測的 h 與 K 格子是事後分析，不是事先登記的判定；表中最好的格（h = 10、K = 10）帶有選擇偏誤。
- 訓練在 RunPod CPU Pod 上執行；本機只做計分與回測，沒有重新訓練驗證。

### 6. 重現方式

```bash
bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --cpuFlavor cpu3g --cpuNumber 16
bash scripts/runpod_workflow.sh download-lgbm
bash scripts/runpod_s3_project.sh s3 cp "s3://<volume>/datasets/<dataset>/raw/market.parquet" artifacts/datasets/<dataset>/raw/market.parquet
cd backtest
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `6146dfb`（§13.13）、`80652d7`（第 7 項實作登記） |
| 程式 commit | `8dc2a83`（B0 walk-forward）、`06dcc45`（逐折 checkpoint 與續跑）、`f6448fd`（Top-k dropout）、`0a7f122`（持有對齊回測） |
| 設定檔 | `configs/walk_forward/tw_quarterly_v1.json`、`configs/features/alpha_tw_v1.json`、`backtest/config/topk_dropout.json`、`backtest/config/holding_walkforward.json` |
| 訓練輸出（不受 Git 追蹤） | `artifacts/lgbm/launch-20261022T234922Z-22048038/`（`results.json`、`predictions.parquet`、`targets.parquet`、`daily_rank_ic.parquet`、`fits.parquet`） |
| 回測輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/topk-dropout-b0-20261023T013206Z/`、`artifacts/diagnostics/backtest/holding-wf-b0-20261023T014557Z/` |

## English

### 1. Purpose and conclusions

This report records the walk-forward result of B0 `t-lgbm-alpha`, step 2 of [experiment protocol §13.13](experiment_protocol.md), and the verdict of the step 6 top-k dropout backtest (H-B) applied to B0. D0 `t-xs-scratch` (H-A) is reported separately.

Conclusions:

1. **B0 ranks consistently.** Over the 777 signal dates of P1 ∪ P2, the short cell (h = 1) Rank IC is 0.1178 [0.1091, 0.1266] with ICIR 0.85 and 79.5% positive days; the holding cell 0.0859 [0.0650, 0.1061]; the long cell 0.0834 [0.0541, 0.1129]. All three interval lower bounds are above 0.
2. **P2 is clearly weaker than P1, and long horizons are near zero in P2.** The short cell falls from 0.1248 in P1 to 0.0996 in P2; the holding cell from 0.1057 to 0.0343 [−0.0004, 0.0726]; the long cell from 0.1112 to 0.0107 [−0.0372, 0.0600]. This matches the recent failure of Taiwan long horizons seen in §13.11.
3. **Ranking skill decays with the horizon.** The mean daily Rank IC on TWSE falls from 0.126 at h = 1 to 0.091 at h = 5 and about 0.083 from h = 10 on; TPEX falls from 0.109 to 0.083–0.085.
4. **The three seeds are nearly identical.** Short cell P1 ∪ P2: 0.1167, 0.1174 and 0.1172 per seed, 0.1178 for the ensemble; seeds differ by less than 0.001.
5. **H-B fails: B0's ranking skill is not tradeable under the registered strategy and costs.** Top-k dropout (TWSE K = 50, 5 replaced per day) selects h = 10 on P1 (net excess IR 0.014); on P1 ∪ P2 under the primary costs the annualized net excess is −6.3% [−19.0%, +7.1%] with IR −0.58. The P1 gross excess is +17.5% per year, but a daily turnover of 10.2% costs about 17.3% per year; in P2 even the gross excess is negative (−5.9% to −21.2% across the three horizons).
6. **The secondary holding-aligned backtest reaches the same conclusion.** Holding the way the label is defined (buy at the next open, sell at the close h sessions later), the h = 1 top 20 earns a gross excess of +0.26% per batch, below the 0.52% day-trade round trip, for a net excess of −66% per year; the most cost-friendly cell, h = 10 with K = 10, is +5.9% [−6.5%, +19.1%] in P1, −10.9% [−31.5%, +11.4%] in P2 and +1.3% [−10.0%, +14.0%] on P1 ∪ P2.
7. **Disposition under §13.13 item 5:** B0 becomes the strategy baseline but does not count as tradeable; H-C does not run for B0; the problem returns to the strategy layer (rotation frequency, holding period, costs), not the model layer. D0 still runs as step 3 of item 6 so that H-A decides whether DL beats B0; if D0 passes H-A, H-B is then run for D0 against B0.

### 2. Data and method

| Item | Content |
| --- | --- |
| Dataset | The §13.12 tw_only 2016 dataset `53e3915b…` (TWSE and TPEX from 2016-01; 2,159 symbols; 4,417,007 raw rows) |
| Features and labels | The 183-column feature library of `configs/features/alpha_tw_v1.json`; labels are the gauss-rank alpha per (market, signal date) for h = 1–14 |
| Walk-forward | `configs/walk_forward/tw_quarterly_v1.json`: 14 quarterly folds (2023Q1–2026Q2), expanding window from 2016-01-01, 15-session embargo, the last 12 months of each training window as fold validation; training rows grow from 896,789 (2023Q1) to 1,589,755 (2026Q2) |
| Model | LightGBM regression (`num_leaves` 31, `learning_rate` 0.05, `feature_fraction` and `bagging_fraction` 0.8, `min_data_in_leaf` 200, at most 1,000 rounds) with early stopping on the fold validation's daily Rank IC (patience 50); one model per (fold, horizon, seed), 588 in total; mean best iteration 97.6, mean validation Rank IC 0.105 |
| Seeds and ensemble | Seeds 42, 43 and 44; `t-lgbm-alpha` is the mean rank of the three seeds' predictions |
| Execution | RunPod CPU Pod `cpu3g` × 16 vCPU (`launch-20261022T234922Z-22048038`): 4 workers × 4 threads, 23.5 GB memory budget; 5,766 s of training wall time |
| Evaluation | Spearman Rank IC per (market, horizon, signal date) against the raw alpha; cells are the means over short (h = 1), holding (h = 5–10) and long (h = 11–14), with both markets weighted equally per day; periods P1 (2023-01 → 2025-05), P2 (2025-06 → 2026-05) and P1 ∪ P2 |
| Intervals | Circular moving-block bootstrap of the daily series, 1,000 replicates, seed 11, blocks of 5/20/28 sessions |
| Top-k dropout (H-B) | `backtest/config/topk_dropout.json`: every session, by the previous close's ranking, sell the 5 lowest-ranked holdings (TPEX 3) and buy the highest-ranked non-holdings at the next open; K = 50 (TPEX 30); the horizon is chosen among {1, 5, 10} by TWSE P1 net excess IR; costs as in stage 1 (primary: commission discount 0.6, slippage ×1); monthly-block (21 sessions) bootstrap with 1,000 replicates |
| Holding-aligned backtest (secondary) | `backtest/config/holding_walkforward.json`: the top K (10/20/50) bought at the next open and sold at the close h (1/2/3/5/10) sessions later, h start sleeves, the stage 1 engine; h = 1 pays the day-trade tax; primary scenario discount 0.6, slippage ×1, conservative turnover |

A TWSE round trip under the primary scenario costs 0.671% for h ≥ 2 (commission 0.0855% × 2, transaction tax 0.3%, slippage 0.1% × 2) and 0.521% for the h = 1 day trade (half the transaction tax).

### 3. Results

#### 3.1 Rank IC cells (ensemble `t-lgbm-alpha`, [95% CI])

| Cell | P1 (562 days) | P2 (215 days) | P1 ∪ P2 (777 days) | ICIR (P1 ∪ P2) | Positive share (P1 ∪ P2) |
| --- | --- | --- | --- | --- | --- |
| short (h = 1) | 0.1248 [0.1154, 0.1341] | 0.0996 [0.0808, 0.1187] | 0.1178 [0.1091, 0.1266] | 0.85 | 79.5% |
| holding (h = 5–10) | 0.1057 [0.0866, 0.1259] | 0.0343 [−0.0004, 0.0726] | 0.0859 [0.0650, 0.1061] | 0.70 | 74.9% |
| long (h = 11–14) | 0.1112 [0.0849, 0.1384] | 0.0107 [−0.0372, 0.0600] | 0.0834 [0.0541, 0.1129] | 0.64 | 71.4% |

#### 3.2 Mean daily Rank IC by market and horizon (P1 ∪ P2)

| Horizon | TWSE | TPEX |
| --- | --- | --- |
| 1 | 0.126 | 0.109 |
| 2 | 0.107 | 0.099 |
| 3 | 0.100 | 0.096 |
| 5 | 0.091 | 0.091 |
| 7 | 0.085 | 0.086 |
| 10 | 0.084 | 0.083 |
| 14 | 0.083 | 0.085 |

#### 3.3 Seed consistency (short, P1 ∪ P2)

| Model | Rank IC | 95% CI | ICIR |
| --- | --- | --- | --- |
| seed 42 | 0.1167 | [0.1078, 0.1256] | 0.84 |
| seed 43 | 0.1174 | [0.1089, 0.1261] | 0.86 |
| seed 44 | 0.1172 | [0.1083, 0.1261] | 0.85 |
| ensemble | 0.1178 | [0.1091, 0.1266] | 0.85 |

#### 3.4 Top-k dropout backtest (H-B, primary costs, long-only)

| Market | Period | h | Gross excess (annual) | Net excess (annual) | 95% CI | IR | Cost drag / year | Max drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TWSE | P1 | 1 | +9.7% | −7.6% | [−20.8%, +7.3%] | −0.68 | 17.3% | −20.1% |
| TWSE | P1 | 5 | +14.2% | −2.8% | [−16.0%, +11.3%] | −0.28 | 17.0% | −20.5% |
| TWSE | P1 | 10 | +17.5% | +0.1% | [−13.5%, +15.0%] | +0.01 | 17.3% | −18.3% |
| TWSE | P2 | 10 | −5.9% | −23.0% | [−42.5%, −3.2%] | −2.08 | 17.1% | −6.2% |
| TWSE | P1 ∪ P2 | 10 | +11.0% | −6.3% | [−19.0%, +7.1%] | −0.58 | 17.3% | −18.3% |
| TPEX | P1 | 10 | +25.3% | +1.2% | [−11.7%, +14.3%] | +0.11 | 24.1% | −31.8% |
| TPEX | P2 | 10 | −29.7% | −54.0% | [−80.0%, −28.3%] | −4.45 | 24.2% | −12.3% |
| TPEX | P1 ∪ P2 | 10 | +10.0% | −14.1% | [−27.6%, +0.0%] | −1.21 | 24.1% | −31.8% |

The selected horizon is h = 10 (TWSE P1 net excess IR 0.014, the highest of the three). The decision cell, TWSE P1 ∪ P2, has an interval lower bound of −19.0% < 0, so H-B fails. TWSE turns over 10.2% per day with a mean holding of 9.6 sessions; the break-even cost per side is 0.21%, below the primary scenario's actual cost of about 0.34% per side. In P2 the TWSE gross excess is negative at every horizon (h = 1 −21.2%, h = 5 −14.7%, h = 10 −5.9%), so the P2 failure is not only a cost effect. The hedged version (short the equal-weight index) is worse than long-only in every cell (TWSE P1 ∪ P2 h = 10: −21.7%).

#### 3.5 Holding-aligned backtest (secondary analysis, TWSE, primary costs)

| h | K | Gross excess per batch (P1 ∪ P2) | Net excess P1 | Net excess P2 | Net excess P1 ∪ P2 | 95% CI (P1 ∪ P2) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | +0.27% | −64.3% | −58.1% | −62.6% | [−76.1%, −47.8%] |
| 1 | 20 | +0.26% | −65.9% | −66.6% | −66.1% | [−77.4%, −54.1%] |
| 5 | 10 | +0.36% | −11.5% | −23.8% | −14.9% | [−24.8%, −3.8%] |
| 5 | 20 | +0.31% | −11.6% | −32.1% | −17.2% | [−25.9%, −7.2%] |
| 10 | 10 | +0.71% | +5.9% | −10.9% | +1.3% | [−10.0%, +14.0%] |
| 10 | 20 | +0.56% | +2.7% | −16.2% | −2.6% | [−12.9%, +9.1%] |
| 10 | 50 | +0.34% | −1.7% | −24.9% | −8.1% | [−17.9%, +2.0%] |

The selection (highest TWSE P1 net excess IR) is h = 10 with K = 10, IR 0.46. The h = 1 gross excess of 0.26% per batch is below the 0.52% day-trade round trip; that cell is net positive only with zero slippage plus optimistic turnover (+23.9%) or a 0.28 discount plus zero slippage (+7.3%). The gross excess grows with h (K = 10: 0.27% to 0.71%) but decays per session, consistent with the horizon decay in 3.2.

### 4. Implications

- Under §13.13 item 5, a failed H-B means ranking skill exists but is not tradeable; the work returns to the strategy layer, not the model layer, and H-C does not run for B0. D0 still runs because H-A, whether DL beats B0, is a separate question; if D0 passes H-A, H-B is run for D0 against B0 (D0's IR not below B0's).
- B0's gross edge is a ranking signal of about 10–13 bps per day, consistent with stage 1 and §13.11: a one-session signal that reverses overnight and is smaller than Taiwan's costs. The feature library and walk-forward raise the Rank IC from 0.06–0.08 in §13.11 to 0.12, but the cost structure is unchanged.
- The numbers point to strategy-layer directions (recorded, not registered): lower turnover (5 of 50 per day is 10% daily turnover, a 17% per year cost drag that exceeds the P1 gross excess), longer and more concentrated holding (h = 10 with K = 10 is closest to break-even), and the P2 regime (after 2025-06 even the gross excess turns negative; whether this is a momentum/reversal regime switch needs checking). Any new strategy rule must be registered before it runs.
- The three seeds are nearly identical, so LightGBM's variance on these features comes from the data rather than from initialization; differences of about 0.001 between D0's seeds should not be read as differences.

### 5. Limits

- P2 was seen in §13.10 and §13.11, and P1 is evaluated for the first time here, but the data rules, feature library and model settings were decided after seeing P2's aggregate results; only P3 (from 2026-06) is a true confirmation.
- The backtests do not model volume caps, round lots, minimum commissions or dividend taxes, nor liquidity limits beyond price limits; the TPEX slippage assumption (0.2%) may still be low.
- The top-k dropout horizon is chosen on P1, but all three P1 net excess IRs are near 0 (−0.68 to +0.01), so the choice itself has almost no discriminating power.
- The holding-aligned h and K grid is a post-hoc analysis, not a pre-registered decision; the best cell in the table (h = 10, K = 10) carries selection bias.
- Training ran on a RunPod CPU Pod; the local machine only scored and backtested, with no re-training check.

### 6. Reproduction

```bash
bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --cpuFlavor cpu3g --cpuNumber 16
bash scripts/runpod_workflow.sh download-lgbm
bash scripts/runpod_s3_project.sh s3 cp "s3://<volume>/datasets/<dataset>/raw/market.parquet" artifacts/datasets/<dataset>/raw/market.parquet
cd backtest
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Item | Content |
| --- | --- |
| Pre-registration commits | `6146dfb` (§13.13), `80652d7` (item 7 implementation registration) |
| Code commits | `8dc2a83` (B0 walk-forward), `06dcc45` (per-fold checkpoints and resume), `f6448fd` (top-k dropout), `0a7f122` (holding-aligned backtest) |
| Configuration | `configs/walk_forward/tw_quarterly_v1.json`, `configs/features/alpha_tw_v1.json`, `backtest/config/topk_dropout.json`, `backtest/config/holding_walkforward.json` |
| Training output (not tracked by Git) | `artifacts/lgbm/launch-20261022T234922Z-22048038/` (`results.json`, `predictions.parquet`, `targets.parquet`, `daily_rank_ic.parquet`, `fits.parquet`) |
| Backtest output (not tracked by Git) | `artifacts/diagnostics/backtest/topk-dropout-b0-20261023T013206Z/`, `artifacts/diagnostics/backtest/holding-wf-b0-20261023T014557Z/` |
