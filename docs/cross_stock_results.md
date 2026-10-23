# D0 跨股票 transformer 與消融 A1 結果 / D0 Cross-Stock Transformer and Ablation A1 Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.13](experiment_protocol.md) 第 3 步 D0 `t-xs-scratch`（MASTER 式跨股票 transformer，三 seed 集成）對 B0 `t-lgbm-alpha` 的 H-A 判定、D0 的 Top-k dropout（H-B）與持有對齊回測，以及第 4 步消融 A1 `t-xs-a1-no-cross`（關閉股票間 attention）。A2–A4 與 D1 依使用者 2026-10-23 的決定延後。

結論：

1. **H-A 一格通過。** D0 − B0 的配對 Rank IC 差在 P1 ∪ P2：短格 +0.0043 [+0.0015, +0.0071]（通過）、持有格 +0.0062 [−0.0012, +0.0136]（未通過）。依第 5 點 D0 為候選模型。
2. **差距統計上成立、經濟上很小。** D0 短格 0.1221 對 B0 0.1178；每個 horizon 的差距在 0.002–0.007 之間；P2 短格差 +0.0013 [−0.0029, +0.0059] 不顯著。D0 在 TPEX 的多日 horizon 略好（h = 10：0.098 對 0.083），在 TWSE 幾乎相同。
3. **D0 的多日 horizon 在 P2 一樣崩掉。** 持有格由 P1 的 0.109 降到 P2 的 0.047，長格 0.110 降到 0.036（區間跨 0）；B0 分別為 0.034 與 0.011。市場狀態門控沒有擋住這個失效，原因在標籤含市值暴露（見 §13.16 的登記）。
4. **D0 的 H-B 未通過。** Top-k dropout 選 h = 10，TWSE P1 ∪ P2 年化淨超額 −7.3% [−17.9%, +3.5%]、IR −0.79，低於 B0 的 −0.58；絕對年化淨報酬 +17.7%、Sharpe 1.05、最大回撤 −24.7%。TPEX P1 一度 +13.4% [+0.4%, +26.9%]，P2 −31%。持有對齊回測同樣為負（h = 10、K = 10：P1 ∪ P2 −6.9% [−15.5%, +2.3%]）。
5. **消融 A1：股票間 attention 沒有可量測的貢獻。** 關閉後 seed 42 的短格 0.1195 [0.1098, 0.1284]，與 D0 自己的 seed 42（0.1185）相當；對 B0 的配對差 +0.0017 [−0.0014, +0.0047] 不顯著。D0 集成對 B0 的 +0.0043 來自三個 seed 的集成，不是跨股票結構。
6. **三個 seed 幾乎相同**（短格 0.1185／0.1181／0.1209），與 B0 的 seed 結果一樣；D0 的訓練在 fold validation 上平均 9.2 個 epoch 早停。
7. **處置**：D0 保留為候選模型但不投入 D1 與其餘消融，等 §13.16 殘差化標籤的結果決定要不要在新標籤上重訓；GPU 使用率只有 13–17%，重訓前應先處理同日分組取數的瓶頸。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 資料、標籤、特徵、walk-forward | 與 B0 完全相同（§13.12、§13.13；資料集 `53e3915b…`，183 欄特徵庫，14 季度 fold，15 日 embargo） |
| D0 架構 | 128 日 OHLCV 視窗的 patch 編碼器（patch 8、d 128、2 層）+ 特徵側輸入（batch 內橫斷面 z-score，裁剪 ±5）→ 同日同市場全部股票一個 batch → 股票間 attention 2 層（4 heads、ff 256、dropout 0.1）→ 市場狀態門控 → 14 個 horizon 分數；參數 645,518 |
| 損失與訓練 | MSE + 1 ×（1 − 當日橫斷面 Pearson）；AdamW lr 5e-4、weight decay 1e-4、最多 20 epoch、以 fold validation 的每日 Rank IC 早停（patience 4）、gradient clip 1.0、bf16 autocast；seed 42、43、44；每折所有 seed 寫入 `partial/` 可續跑 |
| A1 | 同 D0，`inter_stock_attention=false`，seed 42 |
| 硬體 | RunPod GPU Pod，RTX PRO 4500 Blackwell（32 GB，compute capability 12.0，torch 2.9.1+cu128）；資料表常駐 GPU 6.1 GB；D0 42 個 fit 共 17,335 秒（4.8 小時），A1 14 個 fit 4,202 秒；GPU 使用率 13–17%（取數為瓶頸）；時脈監測從 A1 之後才有（SM 2,400–2,800 MHz，無 throttle） |
| 評估 | 同 §13.13：Spearman Rank IC、cell short／holding／long、兩市場等權、P1／P2／P1 ∪ P2；配對差為同一訊號日相減後的 circular block bootstrap（1,000 次、seed 11、區塊 5／20／28） |
| 回測 | §13.13 第 3 項 Top-k dropout（`--compare-predictions` B0）與持有對齊回測（次要），設定同 B0 報告 |

### 3. 結果

#### 3.1 Rank IC cell（三 seed 集成，[95% 區間]）

| cell | 模型 | P1（562 日） | P2（215 日） | P1 ∪ P2（777 日） | ICIR（P1 ∪ P2） |
| --- | --- | --- | --- | --- | --- |
| short | D0 | 0.1303 [0.1207, 0.1401] | 0.1009 [0.0817, 0.1214] | 0.1221 [0.1129, 0.1311] | 0.88 |
| short | B0 | 0.1248 [0.1154, 0.1341] | 0.0996 [0.0808, 0.1187] | 0.1178 [0.1091, 0.1266] | 0.85 |
| holding | D0 | 0.1093 [0.0930, 0.1274] | 0.0472 [0.0157, 0.0820] | 0.0921 [0.0739, 0.1096] | 0.84 |
| holding | B0 | 0.1057 [0.0866, 0.1259] | 0.0343 [−0.0004, 0.0726] | 0.0859 [0.0650, 0.1061] | 0.70 |
| long | D0 | 0.1097 [0.0870, 0.1318] | 0.0360 [−0.0027, 0.0723] | 0.0893 [0.0664, 0.1121] | 0.80 |
| long | B0 | 0.1112 [0.0849, 0.1384] | 0.0107 [−0.0372, 0.0600] | 0.0834 [0.0541, 0.1129] | 0.64 |

#### 3.2 配對差 D0 − B0（同一訊號日，[95% 區間]）

| cell | P1 | P2 | P1 ∪ P2 | 判定 |
| --- | --- | --- | --- | --- |
| short | +0.0055 [+0.0022, +0.0091] | +0.0013 [−0.0029, +0.0059] | +0.0043 [+0.0015, +0.0071] | 通過 |
| holding | +0.0036 [−0.0039, +0.0107] | +0.0129 [−0.0016, +0.0330] | +0.0062 [−0.0012, +0.0136] | 未通過 |
| long | −0.0015 [−0.0117, +0.0081] | +0.0253 [+0.0008, +0.0598] | +0.0059 [−0.0060, +0.0186] | （不在判定內） |

#### 3.3 各市場、各 horizon 的平均每日 Rank IC（P1 ∪ P2）

| horizon | TWSE D0 | TWSE B0 | TPEX D0 | TPEX B0 |
| --- | --- | --- | --- | --- |
| 1 | 0.128 | 0.126 | 0.116 | 0.109 |
| 5 | 0.091 | 0.091 | 0.102 | 0.091 |
| 10 | 0.082 | 0.084 | 0.098 | 0.083 |
| 14 | 0.082 | 0.083 | 0.097 | 0.085 |

#### 3.4 seed 與消融（short 與 holding，P1 ∪ P2）

| 模型 | short | holding | 對 B0 配對差（short） |
| --- | --- | --- | --- |
| D0 seed 42 | 0.1185 | 0.0882 | — |
| D0 seed 43 | 0.1181 | 0.0880 | — |
| D0 seed 44 | 0.1209 | 0.0906 | — |
| D0 集成 | 0.1221 | 0.0921 | +0.0043 [+0.0015, +0.0071] |
| A1 關閉股票間 attention（seed 42） | 0.1195 [0.1098, 0.1284] | 0.0879 [0.0700, 0.1054] | +0.0017 [−0.0014, +0.0047] |
| B0 集成 | 0.1178 | 0.0859 | — |

#### 3.5 D0 的回測（主要成本情境）

| 回測 | 市場 | 期間 | 年化淨報酬 | 年化淨超額 | 95% 區間 | IR | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top-k dropout（h = 10，50／5） | TWSE | P1 | +17.8% | +1.0% | [−10.0%, +12.2%] | 0.12 | 選定 horizon |
| Top-k dropout | TWSE | P2 | +17.4% | −29.1% | [−44.7%, −12.6%] | −2.69 | |
| Top-k dropout | TWSE | P1 ∪ P2 | +17.7% | −7.3% | [−17.9%, +3.5%] | −0.79 | H-B 未通過；B0 IR −0.58 |
| Top-k dropout | TPEX | P1 | +31.2% | +13.4% | [+0.4%, +26.9%] | 1.30 | 唯一下界 > 0 的格 |
| Top-k dropout | TPEX | P1 ∪ P2 | +31.2% | +1.1% | [−11.6%, +14.1%] | 0.10 | |
| 持有對齊（h = 10，K = 10） | TWSE | P1 ∪ P2 | +8.7% | −6.9% | [−15.5%, +2.3%] | −0.59 | 次要分析，每批毛超額 +0.37% |

### 4. 意涵

- 跨股票 transformer 在這組資料上比 LightGBM 多出的排序能力是 0.004 個 Rank IC，且來自 seed 集成；A1 顯示把股票間 attention 拿掉也一樣。MASTER 類架構的增益在這裡沒有重現，與第三方重現 MASTER 只得到原文不到一半 Rank IC 的經驗一致。
- 兩個模型在 P2 的多日失效一模一樣，證明問題不在模型容量而在目標：這是 §13.16 把殘差化標籤往前提的依據。
- 以可交易性而言，D0 與 B0 無差：Top-k dropout 與持有回測都未通過，策略層報告的結論不因模型而變。
- GPU 使用率 13–17% 表示 `cross_stock_training` 的同日分組取數是瓶頸；若要在新標籤上重訓 D0，先把分組索引預先建成張量或改用更大的 batch 合併，否則 4.8 小時中大半是等待。

### 5. 限制

- A1 只有 seed 42，而 D0 的 seed 間差距約 0.002；消融結論依賴「A1 ≈ D0 seed 42」這個單點比較。
- P2 已在 §13.10、§13.11 看過；配對差的區間在 P2 都跨 0。
- D0 的 Pod 沒有時脈監測紀錄，4.8 小時是否受降頻影響無法回溯。
- D1 與 A2–A4 未執行；預訓練編碼器是否有增益沒有答案。

### 6. 重現方式

```bash
bash scripts/runpod_workflow.sh xs-train --experiment t-xs-scratch --maxRuntime 24h --gpuId "NVIDIA RTX PRO 4500 Blackwell"
bash scripts/runpod_workflow.sh xs-train --experiment t-xs-a1-no-cross --seed 42 --maxRuntime 24h --gpuId "NVIDIA RTX PRO 4500 Blackwell"
bash scripts/runpod_workflow.sh download-xs
cd backtest
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet --compare-predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --compare-model t-lgbm-alpha
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 項目 | 內容 |
| --- | --- |
| 事先登記 commit | `6146dfb`（§13.13）、`80652d7`（第 7 項 D0 參數） |
| 程式 commit | `80652d7`（D0、消融、編碼器、xs-train 工作流程）、`06dcc45`／`66aabab`（逐折 checkpoint 與續跑）、`74b9bfc`（xs 准入） |
| 訓練輸出（不受 Git 追蹤） | `artifacts/xs/launch-20261023T015746Z-224647797/`（D0）、`artifacts/xs/launch-20261023T075624Z-2596926448/`（A1）：`results.json`、`predictions.parquet`、`targets.parquet`、`training_history.json` |
| 回測輸出（不受 Git 追蹤） | `artifacts/diagnostics/backtest/topk-dropout-d0-20261023T071315Z/`、`artifacts/diagnostics/backtest/holding-wf-d0-20261023T071315Z/` |

## English

### 1. Purpose and conclusions

This report records the H-A decision of [experiment protocol §13.13](experiment_protocol.md) step 3, D0 `t-xs-scratch` (a MASTER-style cross-stock transformer, three-seed ensemble) against B0 `t-lgbm-alpha`, D0's top-k dropout (H-B) and holding-aligned backtests, and step 4's ablation A1 `t-xs-a1-no-cross` (inter-stock attention off). A2–A4 and D1 are deferred by the user's decision of 2026-10-23.

Conclusions:

1. **H-A passes in one cell.** The paired Rank IC difference D0 − B0 on P1 ∪ P2 is +0.0043 [+0.0015, +0.0071] in the short cell (passes) and +0.0062 [−0.0012, +0.0136] in the holding cell (fails). Under item 5, D0 is the candidate model.
2. **The edge is statistically real and economically small.** D0's short cell is 0.1221 against B0's 0.1178; per-horizon differences are 0.002–0.007; the P2 short difference +0.0013 [−0.0029, +0.0059] is not significant. D0 is slightly better on TPEX multi-session horizons (h = 10: 0.098 vs 0.083) and almost identical on TWSE.
3. **D0's multi-session horizons collapse in P2 as well.** The holding cell falls from 0.109 in P1 to 0.047 in P2 and the long cell from 0.110 to 0.036 (interval spanning 0); B0 shows 0.034 and 0.011. The market-state gating did not prevent the failure, whose cause is label exposure to size (see the §13.16 registration).
4. **D0's H-B fails.** Top-k dropout selects h = 10; the TWSE P1 ∪ P2 annualized net excess is −7.3% [−17.9%, +3.5%] with IR −0.79, below B0's −0.58; the absolute annualized net return is +17.7%, Sharpe 1.05, max drawdown −24.7%. TPEX P1 reaches +13.4% [+0.4%, +26.9%] but P2 is −31%. The holding-aligned backtest is negative too (h = 10, K = 10: P1 ∪ P2 −6.9% [−15.5%, +2.3%]).
5. **Ablation A1: inter-stock attention has no measurable contribution.** Without it, seed 42's short cell is 0.1195 [0.1098, 0.1284], on a par with D0's own seed 42 (0.1185); the paired difference against B0 is +0.0017 [−0.0014, +0.0047], not significant. The D0 ensemble's +0.0043 over B0 comes from the three-seed ensemble, not from the cross-stock structure.
6. **The three seeds are nearly identical** (short 0.1185 / 0.1181 / 0.1209), as for B0; training stopped early after 9.2 epochs on average on the fold validation.
7. **Disposition**: D0 stays the candidate model but gets no D1 or further ablations until the §13.16 residualized labels decide whether it is retrained on new labels; GPU utilization was only 13–17%, so the per-date grouping bottleneck should be fixed before any retraining.

### 2. Data and method

| Item | Content |
| --- | --- |
| Data, labels, features, walk-forward | Identical to B0 (§13.12, §13.13; dataset `53e3915b…`, 183-column feature library, 14 quarterly folds, 15-session embargo) |
| D0 architecture | Patch encoder on the 128-session OHLCV window (patch 8, width 128, 2 layers) + feature side input (within-batch cross-sectional z-score clipped to ±5) → every stock of one signal date and market as one batch → 2 inter-stock attention layers (4 heads, ff 256, dropout 0.1) → market-state gating → 14 horizon scores; 645,518 parameters |
| Loss and training | MSE + 1 × (1 − daily cross-sectional Pearson); AdamW lr 5e-4, weight decay 1e-4, at most 20 epochs, early stopping on the fold validation's daily Rank IC (patience 4), gradient clip 1.0, bf16 autocast; seeds 42, 43, 44; every finished fold and seed written to `partial/` for resumption |
| A1 | As D0 with `inter_stock_attention=false`, seed 42 |
| Hardware | RunPod GPU Pod, RTX PRO 4500 Blackwell (32 GB, compute capability 12.0, torch 2.9.1+cu128); the sample table resident on the GPU (6.1 GB); D0's 42 fits took 17,335 s (4.8 hours), A1's 14 fits 4,202 s; GPU utilization 13–17% (data access is the bottleneck); clock monitoring exists only from A1 on (SM 2,400–2,800 MHz, no throttling) |
| Evaluation | As in §13.13: Spearman Rank IC, cells short / holding / long, both markets weighted equally, P1 / P2 / P1 ∪ P2; paired differences on common signal dates with a circular block bootstrap (1,000 replicates, seed 11, blocks 5 / 20 / 28) |
| Backtests | §13.13 item 3 top-k dropout (`--compare-predictions` B0) and the secondary holding-aligned backtest, settings as in the B0 report |

### 3. Results

#### 3.1 Rank IC cells (three-seed ensembles, [95% CI])

| Cell | Model | P1 (562 days) | P2 (215 days) | P1 ∪ P2 (777 days) | ICIR (P1 ∪ P2) |
| --- | --- | --- | --- | --- | --- |
| short | D0 | 0.1303 [0.1207, 0.1401] | 0.1009 [0.0817, 0.1214] | 0.1221 [0.1129, 0.1311] | 0.88 |
| short | B0 | 0.1248 [0.1154, 0.1341] | 0.0996 [0.0808, 0.1187] | 0.1178 [0.1091, 0.1266] | 0.85 |
| holding | D0 | 0.1093 [0.0930, 0.1274] | 0.0472 [0.0157, 0.0820] | 0.0921 [0.0739, 0.1096] | 0.84 |
| holding | B0 | 0.1057 [0.0866, 0.1259] | 0.0343 [−0.0004, 0.0726] | 0.0859 [0.0650, 0.1061] | 0.70 |
| long | D0 | 0.1097 [0.0870, 0.1318] | 0.0360 [−0.0027, 0.0723] | 0.0893 [0.0664, 0.1121] | 0.80 |
| long | B0 | 0.1112 [0.0849, 0.1384] | 0.0107 [−0.0372, 0.0600] | 0.0834 [0.0541, 0.1129] | 0.64 |

#### 3.2 Paired difference D0 − B0 (common signal dates, [95% CI])

| Cell | P1 | P2 | P1 ∪ P2 | Decision |
| --- | --- | --- | --- | --- |
| short | +0.0055 [+0.0022, +0.0091] | +0.0013 [−0.0029, +0.0059] | +0.0043 [+0.0015, +0.0071] | passes |
| holding | +0.0036 [−0.0039, +0.0107] | +0.0129 [−0.0016, +0.0330] | +0.0062 [−0.0012, +0.0136] | fails |
| long | −0.0015 [−0.0117, +0.0081] | +0.0253 [+0.0008, +0.0598] | +0.0059 [−0.0060, +0.0186] | (not a decision cell) |

#### 3.3 Mean daily Rank IC by market and horizon (P1 ∪ P2)

| Horizon | TWSE D0 | TWSE B0 | TPEX D0 | TPEX B0 |
| --- | --- | --- | --- | --- |
| 1 | 0.128 | 0.126 | 0.116 | 0.109 |
| 5 | 0.091 | 0.091 | 0.102 | 0.091 |
| 10 | 0.082 | 0.084 | 0.098 | 0.083 |
| 14 | 0.082 | 0.083 | 0.097 | 0.085 |

#### 3.4 Seeds and the ablation (short and holding, P1 ∪ P2)

| Model | short | holding | Paired vs B0 (short) |
| --- | --- | --- | --- |
| D0 seed 42 | 0.1185 | 0.0882 | — |
| D0 seed 43 | 0.1181 | 0.0880 | — |
| D0 seed 44 | 0.1209 | 0.0906 | — |
| D0 ensemble | 0.1221 | 0.0921 | +0.0043 [+0.0015, +0.0071] |
| A1 inter-stock attention off (seed 42) | 0.1195 [0.1098, 0.1284] | 0.0879 [0.0700, 0.1054] | +0.0017 [−0.0014, +0.0047] |
| B0 ensemble | 0.1178 | 0.0859 | — |

#### 3.5 D0 backtests (primary costs)

| Backtest | Market | Period | Net return (annual) | Net excess (annual) | 95% CI | IR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top-k dropout (h = 10, 50 / 5) | TWSE | P1 | +17.8% | +1.0% | [−10.0%, +12.2%] | 0.12 | selected horizon |
| Top-k dropout | TWSE | P2 | +17.4% | −29.1% | [−44.7%, −12.6%] | −2.69 | |
| Top-k dropout | TWSE | P1 ∪ P2 | +17.7% | −7.3% | [−17.9%, +3.5%] | −0.79 | H-B fails; B0 IR −0.58 |
| Top-k dropout | TPEX | P1 | +31.2% | +13.4% | [+0.4%, +26.9%] | 1.30 | the only cell with a lower bound > 0 |
| Top-k dropout | TPEX | P1 ∪ P2 | +31.2% | +1.1% | [−11.6%, +14.1%] | 0.10 | |
| Holding-aligned (h = 10, K = 10) | TWSE | P1 ∪ P2 | +8.7% | −6.9% | [−15.5%, +2.3%] | −0.59 | secondary; gross +0.37% per batch |

### 4. Implications

- On this data the cross-stock transformer's extra ranking skill over LightGBM is 0.004 Rank IC and comes from the seed ensemble; A1 shows removing inter-stock attention changes nothing. The MASTER-style gain does not reproduce here, consistent with the third-party reproduction that reached less than half of the paper's Rank IC.
- Both models fail identically on multi-session horizons in P2, which shows the problem is the target, not model capacity: that is the basis for pulling the §13.16 residualized labels forward.
- For tradeability D0 and B0 are indistinguishable: top-k dropout and the holding backtest both fail, and the strategy-layer conclusions do not depend on the model.
- GPU utilization of 13–17% means the per-date grouping in `cross_stock_training` is the bottleneck; before retraining D0 on new labels the group indices should be prebuilt as tensors or several dates merged per step, otherwise most of the 4.8 hours is waiting.

### 5. Limits

- A1 has seed 42 only while D0's seeds differ by about 0.002; the ablation conclusion rests on the single comparison A1 ≈ D0 seed 42.
- P2 was seen in §13.10 and §13.11; every paired difference in P2 spans 0.
- D0's Pod has no clock-monitor record, so whether the 4.8 hours were affected by down-clocking cannot be checked.
- D1 and A2–A4 did not run; whether pretrained encoders add value is unanswered.

### 6. Reproduction

```bash
bash scripts/runpod_workflow.sh xs-train --experiment t-xs-scratch --maxRuntime 24h --gpuId "NVIDIA RTX PRO 4500 Blackwell"
bash scripts/runpod_workflow.sh xs-train --experiment t-xs-a1-no-cross --seed 42 --maxRuntime 24h --gpuId "NVIDIA RTX PRO 4500 Blackwell"
bash scripts/runpod_workflow.sh download-xs
cd backtest
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet --compare-predictions ../artifacts/lgbm/launch-20261022T234922Z-22048038/predictions.parquet --compare-model t-lgbm-alpha
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/xs/launch-20261023T015746Z-224647797/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Item | Content |
| --- | --- |
| Pre-registration commits | `6146dfb` (§13.13), `80652d7` (item 7 D0 parameters) |
| Code commits | `80652d7` (D0, ablations, encoders, xs-train workflow), `06dcc45` / `66aabab` (per-fold checkpoints and resume), `74b9bfc` (xs admission) |
| Training output (not tracked by Git) | `artifacts/xs/launch-20261023T015746Z-224647797/` (D0), `artifacts/xs/launch-20261023T075624Z-2596926448/` (A1): `results.json`, `predictions.parquet`, `targets.parquet`, `training_history.json` |
| Backtest output (not tracked by Git) | `artifacts/diagnostics/backtest/topk-dropout-d0-20261023T071315Z/`, `artifacts/diagnostics/backtest/holding-wf-d0-20261023T071315Z/` |
