# §13.18 籌碼增量 G1／H-H 結果 / Chip Increment Results: G1 and H-H

## 中文

### 1. 目的與結論

[實驗協定 §13.18](experiment_protocol.md) 問的是：免費官方籌碼與事件資料做成的特徵庫，加到價量模型上，排序能力有沒有增量。三個 B0 LightGBM walk-forward 在同一排程（`tw_quarterly_v1`，14 折）、同一標籤（`residual_v1a`）、同一 seed（42／43／44 集成）下只換特徵規格：F0 = `alpha_tw_v1`（183 欄價量）、F1 = `chip_tw_v1`（67 欄籌碼與事件）、F2 = 兩者合一（250 欄）。本報告記錄第 4、5 項的配對計分、H-H 與 G1 的判定、G1 子分析，以及依第 10 項的處置。

結論：

1. **H-H 通過。** F2 − F0 在全宇宙持有格（h 5–10、市場等權）對殘差目標的配對 Rank IC 差為 **+0.017 [+0.013, +0.021]**（P1 ∪ P2，777 日），P1 與 P2 各自的下界也都為正；流動性宇宙 h = 10 的前後四分位殘差價差由 F0 的每 10 日 0.65% 升到 F2 的 **1.04%，為 1.59 倍**（門檻 1.5 倍）。兩個條件都成立。第 (ii) 項的倍數接近門檻：同一價差在全宇宙是 1.44 倍、中小型股 1.48 倍，宇宙的選擇是計分前寫死的（實作註記 (b)），但倍數本身沒有多少餘裕。
2. **G1 通過，而且兩個宇宙都過。** 流動性宇宙 ΔIC **+0.020 [+0.015, +0.025]**、價差 1.04% 對 0.65%；中小型股 ΔIC **+0.019 [+0.014, +0.024]**、價差 1.08% 對 0.73%。依第 10 項，分點資料的第一道門已開，購買 FinMind Sponsor Pro 一個月（NT$3,330）由使用者決定；G2（事件反轉）仍要做，它決定的是事件觸發策略 S-B 要不要實作，不再是購買條件。
3. **籌碼單獨（F1）與價量打平，P2 反而更好。** F1 對 F0 的 P1 ∪ P2 配對差 +0.004 [−0.004, +0.012]，沒有通過；但分期看，P1 −0.003 [−0.012, +0.005]、**P2 +0.025 [+0.014, +0.034]**。67 欄籌碼特徵在 2025-06 之後的排序能力（持有格殘差 IC 0.090）超過 183 欄價量特徵（0.065），全宇宙的 h = 1 短格則一直略輸（P1 −0.005）。籌碼不能取代價量，但它在價量模型最弱的時段最有用。
4. **增量在每一段、每一組都有，但在近期強勢股裡更大。** F2 − F0 的持有格殘差 ΔIC，流動性宇宙裡強勢股組 **+0.029 [+0.021, +0.037]**、其餘 +0.018 [+0.012, +0.024]（P1 ∪ P2）；P2 強勢股 +0.029 [+0.013, +0.046]、其餘 +0.017 [+0.005, +0.027]，符合事前寫下的預期方向，但 P1 的強勢股組也有同樣大小的改善，所以增量不是只在動能延續期出現的條件化反轉，而是「獨立訊號加上對強勢股更敏感」。被 F0 排到後四分位的強勢股在 P2 事後仍比宇宙多漲每 10 日 +0.86%（區間跨 0），F2 把它壓到 +0.30%；P1 ∪ P2 由 −0.17% 變 **−0.60% [−1.09%, −0.08%]**，配對差 −0.43% [−0.64%, −0.23%]。籌碼特徵確實讓「被排到後段卻繼續漲的強勢股」變少，但沒有消失。
5. **幅度仍小。** F2 在流動性宇宙 h = 10 的產業中性、金額中性多空每批 1.02%，換算毛年化約 +12.9%（F0 +8.4%）；P2 為 +6.6%（F0 0.0%）。這個量級低於 §13.17 估的每日換 1/10 部位的成本拖累，H-H 通過不表示可交易；可不可交易由第 7 項的策略層（低換手的 S-A／S-C、雙成本檔）決定。股票期貨流動宇宙裡 F2 的 P1 ∪ P2 殘差 IC 0.075（F0 0.054），P2 由 0.010 升到 0.049，但 P2 的四分位價差仍為負（−0.19%，F0 −1.17%）——可放空的大型股宇宙裡，籌碼讓排序不再失效，還不夠讓它賺錢。
6. **處置**：H-H 與 G1 通過寫入協定；下一步依序是 (a) G2 事件研究（第 6 項，決定 S-B）；(b) 第 7 項策略層以 F2 分數實作 S-A／S-C（與 §13.19 的 band 建構、V1 正交化並列）；(c) 第 3 項「F2 通過 H-H 且殘差幅度明顯時另案決定是否重訓 D0」——殘差價差由 0.65% 升到 1.04%，是否算「明顯」交由使用者決定，本專案的建議是先做 (b)，因為 D0 的增量集中在 2023–2024 且 GPU 成本高；(d) 分點購買由使用者決定。P2 已經看過、設計是在 §13.16–§13.17 之後才定的，所有通過都要在 P3 確認。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 三個 run | F0 `launch-20261023T103805Z-2637116736`、F1 `launch-20261027T082635Z-105393503`、F2 `launch-20261027T104815Z-1559225699`；B0 LightGBM、`tw_quarterly_v1` 14 折、`residual_v1a` 標籤、seed 42／43／44 集成，588 fits；F1 的 `best_iteration` 中位數 135、F2 182，三個 split 的特徵庫 `chip_coverage` 皆 2,444 檔融資、2,680 檔法人（作廢的第一次 F1 結果見協定 §13.18 實作註記 (h)） |
| 配對 | 每個指標先算成每日序列（市場等權），再在三個 run 的共同日期逐日相減；circular block bootstrap 1,000 次、seed 11；h = 1 區塊 5，持有格與 h = 10 價差區塊 20 |
| 目標 | 殘差 = 各 run `targets.parquet` 的 `residual_hNN`（§13.16 的殘差目標，三個 run 相同）；原始 = 同檔的 `alpha_hNN`（次一開盤到 h 日後收盤） |
| 宇宙 | all = 每市場全部有分數的股票（§13.13 的格，每市場平均約 460 檔、強勢股組約 90 檔）；liquidity = §13.14 市場內流動性宇宙（20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬、收盤 ≥ 10 元、無 ETF，每市場平均約 360 檔、強勢股組約 70 檔）；small_mid = liquidity 扣除 §13.15 股票期貨流動宇宙（每市場平均約 300 檔、強勢股組約 60 檔）；futures_liquid = 股票期貨流動宇宙（兩市場合併，平均約 130 檔、強勢股組約 26 檔；期交所 `taifex-all`，862 個交易日、314 個契約） |
| 指標 | Spearman Rank IC（持有格 = h 5–10 平均；短格 = h 1）；前後四分位價差（h 1、5、10，每批）；後四分位超額 = 模型排在後四分位者的原始目標平均減同日宇宙組平均；產業中性、金額中性多空（§13.17 步驟 0 的 `quantile_long_short`，h = 10、全部樣本） |
| 判定（事前） | H-H：(i) all 持有格殘差 ΔIC 下界 > 0 且 (ii) liquidity h = 10 殘差價差 F2 ≥ 1.5 × F0（F0 > 0）；G1：liquidity 或 small_mid 任一的 ΔIC 下界 > 0 且 F2 價差 ≥ F0 |
| 子分析 | 強勢股 = 訊號日前 20 個交易日（還原收盤）報酬在宇宙組內前五分之一；其餘 = 其他；每（日期, 組）至少 20 檔 |
| 程式與產物 | `backtest/twbacktest/chip_increment.py`（合成資料測試 `backtest/tests/test_chip_increment.py`）；輸出 `artifacts/diagnostics/backtest/chip-increment-f0-f1-f2-20261027T210000Z/`（`grid.csv` 1,440 列水準、`paired.csv` 960 列配對差、`daily.csv`、`report.json`、`summary.md`） |

### 3. 結果

#### 3.1 判定（P1 ∪ P2，全部樣本）

| 判定 | 條件 | 數值 | 結果 |
| --- | --- | --- | --- |
| H-H (i) | all 持有格殘差 ΔIC（F2 − F0）下界 > 0 | +0.0173 [+0.0128, +0.0214] | 成立 |
| H-H (ii) | liquidity h = 10 殘差四分位價差 F2 ≥ 1.5 × F0 | 1.04% 對 0.65%（1.59 倍） | 成立 |
| **H-H** | 兩者皆成立 | | **通過** |
| G1 (a) | liquidity ΔIC 下界 > 0 且價差不低於 F0 | +0.0202 [+0.0149, +0.0250]；1.04% ≥ 0.65% | 成立 |
| G1 (b) | small_mid 同上 | +0.0191 [+0.0137, +0.0241]；1.08% ≥ 0.73% | 成立 |
| **G1** | 任一成立 | | **通過** |
| F1 − F0 | 同一組條件 | ΔIC +0.0045 [−0.0035, +0.0120]；0.83% 對 0.65%（1.28 倍） | H-H、G1 皆未通過（只作描述） |

#### 3.2 水準：持有格殘差 Rank IC 與 h = 10 四分位殘差價差（全部樣本）

| 宇宙 | 期間 | F0 IC [95%] | F1 IC | F2 IC | F0 價差／批 | F1 價差 | F2 價差 | F0 產業中性毛年化 | F2 產業中性毛年化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | P1 | 0.095 [0.082, 0.108] | 0.091 | 0.111 | +0.93% | +0.85% | +1.15% | +12.4% | +15.2% |
| all | P2 | 0.065 [0.046, 0.086] | 0.090 | 0.084 | +0.17% | +0.82% | +0.73% | −0.1% | +6.4% |
| all | P1 ∪ P2 | 0.086 [0.074, 0.100] | 0.091 | 0.104 | +0.72% | +0.84% | +1.04% | +8.8% | +12.7% |
| liquidity | P1 | 0.091 [0.077, 0.106] | 0.091 | 0.111 | +0.88% | +0.87% | +1.18% | +11.6% | +15.2% |
| liquidity | P2 | 0.058 [0.036, 0.082] | 0.083 | 0.079 | +0.06% | +0.73% | +0.65% | +0.1% | +6.6% |
| liquidity | P1 ∪ P2 | 0.082 [0.067, 0.096] | 0.089 | 0.102 | +0.65% | +0.83% | +1.04% | +8.4% | +12.9% |
| small_mid | P1 | 0.094 [0.079, 0.109] | 0.095 | 0.114 | +0.91% | +0.92% | +1.24% | +12.1% | +16.3% |
| small_mid | P2 | 0.067 [0.044, 0.093] | 0.085 | 0.083 | +0.28% | +0.72% | +0.68% | +1.6% | +6.6% |
| small_mid | P1 ∪ P2 | 0.086 [0.072, 0.100] | 0.092 | 0.105 | +0.73% | +0.87% | +1.08% | +9.2% | +13.6% |
| futures_liquid | P1 | 0.071 [0.048, 0.095] | 0.063 | 0.084 | +0.52% | +0.50% | +0.73% | +6.4% | +10.3% |
| futures_liquid | P2 | 0.010 [−0.024, +0.046] | 0.060 | 0.049 | −1.17% | +0.04% | −0.19% | −9.7% | −0.3% |
| futures_liquid | P1 ∪ P2 | 0.054 [0.032, 0.076] | 0.063 | 0.075 | +0.05% | +0.37% | +0.48% | +2.0% | +7.4% |

產業中性多空毛年化 = 每批價差 × 0.5 × 252／10（總名目 = 資本，每邊一半），未扣成本。

#### 3.3 配對差 F2 − F0 與 F1 − F0（全部樣本）

| 對照 | 宇宙 | 期間 | ΔIC 殘差 持有 [95%] | ΔIC 原始 持有 | ΔIC 殘差 短（h 1） | Δ殘差價差 h 10 [95%] | Δ後四分位超額 h 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 − F0 | all | P1 | +0.016 [+0.011, +0.021] | +0.016 | +0.013 | +0.23% [+0.11%, +0.33%] | −0.12% |
| F2 − F0 | all | P2 | +0.020 [+0.013, +0.026] | +0.026 | +0.012 | +0.56% [+0.32%, +0.83%] | −0.39% |
| F2 − F0 | all | P1 ∪ P2 | **+0.017 [+0.013, +0.021]** | +0.018 | +0.013 | **+0.32% [+0.20%, +0.44%]** | −0.20% |
| F2 − F0 | liquidity | P1 | +0.020 [+0.014, +0.026] | +0.019 | +0.015 | +0.31% [+0.19%, +0.42%] | −0.17% |
| F2 − F0 | liquidity | P2 | +0.021 [+0.013, +0.029] | +0.025 | +0.014 | +0.58% [+0.32%, +0.87%] | −0.37% |
| F2 − F0 | liquidity | P1 ∪ P2 | **+0.020 [+0.015, +0.025]** | +0.021 | +0.015 | **+0.38% [+0.26%, +0.51%]** | −0.23% |
| F2 − F0 | small_mid | P1 | +0.021 [+0.015, +0.026] | +0.021 | +0.016 | +0.33% [+0.20%, +0.45%] | −0.20% |
| F2 − F0 | small_mid | P2 | +0.016 [+0.006, +0.024] | +0.017 | +0.013 | +0.40% [+0.16%, +0.65%] | −0.22% |
| F2 − F0 | small_mid | P1 ∪ P2 | **+0.019 [+0.014, +0.024]** | +0.020 | +0.015 | **+0.35% [+0.23%, +0.47%]** | −0.21% |
| F2 − F0 | futures_liquid | P1 | +0.014 [−0.005, +0.032] | +0.002 | +0.013 | +0.22% [−0.08%, +0.50%] | −0.00% |
| F2 − F0 | futures_liquid | P2 | +0.039 [+0.016, +0.060] | +0.038 | +0.015 | +0.98% [+0.37%, +1.62%] | −0.40% |
| F2 − F0 | futures_liquid | P1 ∪ P2 | +0.021 [+0.006, +0.035] | +0.012 | +0.013 | +0.43% [+0.13%, +0.74%] | −0.11% |
| F1 − F0 | all | P1 | −0.003 [−0.012, +0.005] | −0.009 | −0.005 | −0.08% [−0.25%, +0.09%] | +0.06% |
| F1 − F0 | all | P2 | **+0.025 [+0.014, +0.034]** | +0.020 | +0.002 | +0.64% [+0.37%, +0.94%] | −0.31% |
| F1 − F0 | all | P1 ∪ P2 | +0.004 [−0.004, +0.012] | −0.001 | −0.003 | +0.12% [−0.06%, +0.30%] | −0.04% |
| F1 − F0 | liquidity | P1 | +0.000 [−0.009, +0.010] | −0.004 | −0.001 | −0.00% [−0.19%, +0.18%] | −0.00% |
| F1 − F0 | liquidity | P2 | **+0.025 [+0.013, +0.036]** | +0.021 | +0.005 | +0.66% [+0.34%, +1.01%] | −0.31% |
| F1 − F0 | liquidity | P1 ∪ P2 | +0.007 [−0.002, +0.016] | +0.003 | +0.000 | +0.18% [−0.01%, +0.39%] | −0.09% |
| F1 − F0 | small_mid | P1 ∪ P2 | +0.006 [−0.003, +0.015] | +0.003 | +0.001 | +0.13% [−0.05%, +0.32%] | −0.04% |
| F1 − F0 | small_mid | P2 | +0.018 [+0.006, +0.028] | +0.013 | +0.003 | +0.45% [+0.20%, +0.67%] | −0.13% |
| F1 − F0 | futures_liquid | P1 ∪ P2 | +0.009 [−0.012, +0.028] | −0.012 | +0.002 | +0.32% [−0.07%, +0.76%] | +0.01% |
| F1 − F0 | futures_liquid | P2 | +0.050 [+0.021, +0.080] | +0.038 | +0.012 | +1.21% [+0.35%, +2.21%] | −0.38% |

#### 3.4 G1 子分析：近期強勢股對其餘（F2 − F0，持有格殘差 ΔIC 與 h = 10 指標）

| 宇宙 | 組 | 期間 | ΔIC 殘差 持有 [95%] | Δ殘差價差 h 10 [95%] | 後四分位超額 F0 | 後四分位超額 F2 [95%] | Δ後四分位超額 [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| liquidity | 強勢股 | P1 | +0.029 [+0.021, +0.039] | +0.55% [+0.26%, +0.85%] | −0.56% | −0.94% [−1.42%, −0.49%] | −0.38% [−0.62%, −0.16%] |
| liquidity | 強勢股 | P2 | +0.029 [+0.013, +0.046] | +1.05% [+0.26%, +1.92%] | +0.86% | +0.30% [−0.81%, +1.43%] | −0.56% [−1.06%, −0.10%] |
| liquidity | 強勢股 | P1 ∪ P2 | **+0.029 [+0.021, +0.037]** | **+0.69% [+0.37%, +1.01%]** | −0.17% | **−0.60% [−1.09%, −0.08%]** | −0.43% [−0.64%, −0.23%] |
| liquidity | 其餘 | P1 | +0.019 [+0.011, +0.025] | +0.25% [+0.10%, +0.39%] | −0.58% | −0.69% | −0.11% [−0.22%, +0.01%] |
| liquidity | 其餘 | P2 | +0.017 [+0.005, +0.027] | +0.42% [+0.14%, +0.68%] | −0.11% | −0.31% | −0.19% [−0.40%, +0.01%] |
| liquidity | 其餘 | P1 ∪ P2 | +0.018 [+0.012, +0.024] | +0.30% [+0.16%, +0.43%] | −0.45% | −0.58% | −0.13% [−0.23%, −0.03%] |
| small_mid | 強勢股 | P2 | +0.019 [+0.003, +0.033] | +0.54% [−0.09%, +1.17%] | +0.41% | +0.26% | −0.15% [−0.51%, +0.22%] |
| small_mid | 強勢股 | P1 ∪ P2 | +0.025 [+0.017, +0.034] | +0.52% [+0.20%, +0.84%] | −0.35% | −0.64% | −0.29% [−0.50%, −0.08%] |
| small_mid | 其餘 | P2 | +0.011 [+0.001, +0.021] | +0.32% [+0.07%, +0.57%] | −0.07% | −0.19% | −0.12% [−0.31%, +0.06%] |
| small_mid | 其餘 | P1 ∪ P2 | +0.017 [+0.011, +0.023] | +0.31% [+0.16%, +0.44%] | −0.44% | −0.59% | −0.15% [−0.25%, −0.04%] |
| futures_liquid | 強勢股 | P2 | +0.034 [−0.016, +0.091] | +0.84% [−1.01%, +3.07%] | | | −0.11% [−1.23%, +0.83%] |
| futures_liquid | 強勢股 | P1 ∪ P2 | +0.029 [+0.001, +0.058] | +0.83% [−0.03%, +1.70%] | | | −0.47% [−0.98%, +0.02%] |
| futures_liquid | 其餘 | P2 | +0.042 [+0.019, +0.065] | +0.81% [+0.37%, +1.25%] | | | −0.20% [−0.57%, +0.15%] |
| futures_liquid | 其餘 | P1 ∪ P2 | +0.019 [+0.003, +0.035] | +0.27% [−0.03%, +0.55%] | | | +0.06% [−0.11%, +0.23%] |

F1 − F0 的子分析（流動性宇宙，P1 ∪ P2）：強勢股組 ΔIC +0.020 [+0.008, +0.032]、其餘 +0.004 [−0.007, +0.015]——籌碼單獨的增量幾乎全部集中在強勢股；P2 兩組都是 +0.023。強勢股組在股票期貨流動宇宙每日約 25–30 檔，區間很寬。

### 4. 判讀

- **事前預期的檢驗**：預期「若籌碼機制成立，F2 在 P2 強勢股組的改善應大於其他組」。P2 流動性宇宙強勢股 +0.029 對其餘 +0.017、中小型股 +0.019 對 +0.011，方向符合；但 P1 強勢股組的改善（+0.029）與 P2 一樣大，而其餘組在兩期也都顯著改善。結論是籌碼特徵同時帶來一個與期間無關的獨立訊號，以及對近期強勢股更強的分辨力；它不是只在 P2 才出現的條件化反轉。
- **F1 的形態**：籌碼單獨在 P1 與價量打平、在 P2 勝出，短格（h 1）一直略輸。P2（2025-06 之後）是價量模型殘差排序衰減最多的時段，籌碼特徵補的正是這一段；這也意味著 F2 的增量有多少能延續到 P3，要看 P2 的形態是否持續。
- **H-H (ii) 的餘裕**：1.59 倍只比門檻高 0.09；全宇宙 1.44 倍、中小型股 1.48 倍會不過。本報告照事前寫死的宇宙判定，但把這個敏感度列為限制。
- **可放空宇宙**：F2 把股票期貨流動宇宙的 P2 殘差 IC 從 0.01 拉到 0.05，四分位價差從 −1.17% 拉到 −0.19%，仍為負；§13.17 步驟 0「alpha 不在可放空宇宙」的結論在 F2 下變成「可放空宇宙的排序不再失效，但沒有可交易的幅度」。
- **限制**：P2 已經看過，§13.18 本身是看到 §13.16–§13.17 結果後設計的，通過只在資訊層面，正式判定在 P3；三 seed 集成單一 walk-forward；籌碼特徵的 `FDLR_Z60` 欄恆為缺值（已知）；公布時刻用保守規則（T+1 開盤前，實作註記 (b)），若實測某品項更晚公布須延後一天重跑。

### 5. 處置

| 項目 | 協定依據 | 處置 |
| --- | --- | --- |
| H-H | §13.18 第 5 項 | 通過；第 7 項策略層（S-A、S-C 以 F2 分數，雙成本檔）登記為下一步，S-B 待 G2 |
| G1 | §13.18 第 5 項、第 10 項 | 通過；分點購買（FinMind Sponsor Pro 一個月）的條件成立，由使用者決定 |
| F1 | §13.18 第 5 項 | 只作描述：籌碼不能取代價量（P1 ∪ P2 未過），但 P2 勝過價量 |
| D0 於 F2 | §13.18 第 3 項 | 另案決定；本專案建議先做策略層 |
| G2 | §13.18 第 6 項 | 照原計畫執行，決定 S-B |

## English

### 1. Purpose and conclusions

[Experiment protocol §13.18](experiment_protocol.md) asks whether a feature library built from free official chip (投資人籌碼) and event data adds ranking skill to the price / volume model. Three B0 LightGBM walk-forwards share one schedule (`tw_quarterly_v1`, 14 folds), one label (`residual_v1a`) and one seed set (42 / 43 / 44 ensemble) and differ only in the feature spec: F0 = `alpha_tw_v1` (183 price / volume columns), F1 = `chip_tw_v1` (67 chip and event columns), F2 = both (250 columns). This report records the paired scoring of items 4–5, the H-H and G1 verdicts, the G1 sub-analysis and the dispositions under item 10.

Conclusions:

1. **H-H passes.** The F2 − F0 paired Rank IC difference against the residual target in the full-universe holding cell (h 5–10, equal market weights) is **+0.017 [+0.013, +0.021]** (P1 ∪ P2, 777 days), with positive lower bounds in P1 and P2 separately; the h = 10 top-minus-bottom quartile residual spread in the liquidity universe rises from F0's 0.65% per 10 sessions to F2's **1.04%, a factor of 1.59** (bar 1.5). Both conditions hold. The margin on (ii) is thin: the same ratio is 1.44 in the full universe and 1.48 in the small / mid caps; the universe was fixed before scoring (implementation note (b)), but the multiple itself has little slack.
2. **G1 passes, in both universes.** Liquidity universe ΔIC **+0.020 [+0.015, +0.025]** with spreads 1.04% vs 0.65%; small / mid caps ΔIC **+0.019 [+0.014, +0.024]** with 1.08% vs 0.73%. Under item 10 the first gate for broker-branch (分點) data is open; buying FinMind Sponsor Pro for one month (NT$3,330) is the user's decision. G2 (event reversals) is still to be run; it now decides only whether the event-triggered strategy S-B is implemented, not the purchase.
3. **Chip data alone (F1) ties price / volume and beats it in P2.** F1 − F0 on P1 ∪ P2 is +0.004 [−0.004, +0.012], not a pass; by period, P1 −0.003 [−0.012, +0.005] and **P2 +0.025 [+0.014, +0.034]**. After 2025-06 the 67 chip columns rank better (holding-cell residual IC 0.090) than the 183 price / volume columns (0.065), while the h = 1 short cell on the full universe stays slightly behind (P1 −0.005). Chip data cannot replace price / volume, but it is most useful exactly where the price / volume model is weakest.
4. **The increment is present in every period and segment, and larger among recent winners.** The F2 − F0 holding-cell residual ΔIC in the liquidity universe is **+0.029 [+0.021, +0.037]** for the winners and +0.018 [+0.012, +0.024] for the rest (P1 ∪ P2); in P2, +0.029 [+0.013, +0.046] vs +0.017 [+0.005, +0.027], the direction written down beforehand — but the P1 winners improve by the same amount, and the rest improves significantly in both periods, so the increment is an independent signal plus sharper discrimination among winners, not a conditional reversal confined to momentum regimes. Winners that F0 ranks in the bottom quartile still beat the universe by +0.86% per 10 sessions in P2 (interval spans 0); F2 cuts that to +0.30%; on P1 ∪ P2 the figure moves from −0.17% to **−0.60% [−1.09%, −0.08%]**, a paired change of −0.43% [−0.64%, −0.23%]. Chip features make "winners ranked at the bottom that keep rising" rarer, not extinct.
5. **The magnitude stays small.** F2's industry-neutral, dollar-neutral long-short in the liquidity universe at h = 10 earns 1.02% per cohort, about +12.9% gross a year (F0 +8.4%); P2 +6.6% (F0 0.0%). That is below the cost drag §13.17 estimated for replacing a tenth of the book daily, so passing H-H does not mean tradeable; tradeability is decided by the item 7 strategy layer (low-turnover S-A / S-C, both cost profiles). Inside the liquid stock-futures universe F2's P1 ∪ P2 residual IC is 0.075 (F0 0.054) and P2 rises from 0.010 to 0.049, but the P2 quartile spread stays negative (−0.19%, F0 −1.17%): in the shortable large caps the chip data stops the ranking from failing without making it pay.
6. **Dispositions**: H-H and G1 passes go into the protocol; next, in order, (a) the G2 event study (item 6, decides S-B); (b) the item 7 strategy layer with F2 scores for S-A / S-C (alongside §13.19's band construction and V1 orthogonalization); (c) item 3's "decide separately whether to retrain D0 once F2 passes H-H with a clear residual spread" — the spread rose from 0.65% to 1.04%; whether that is "clear" is the user's call, and this project recommends (b) first because D0's gains sat in 2023–2024 and GPU runs are costly; (d) the broker-branch purchase is the user's decision. P2 has been seen and the design was fixed after §13.16–§13.17, so every pass is confirmed in P3.

### 2. Data and method

| Item | Content |
| --- | --- |
| The three runs | F0 `launch-20261023T103805Z-2637116736`, F1 `launch-20261027T082635Z-105393503`, F2 `launch-20261027T104815Z-1559225699`; B0 LightGBM, `tw_quarterly_v1` 14 folds, `residual_v1a` labels, seeds 42 / 43 / 44 ensembled, 588 fits; median `best_iteration` 135 (F1) and 182 (F2); every split store's `chip_coverage` is 2,444 margin and 2,680 institutional symbols (the voided first F1 result is in protocol §13.18 implementation note (h)) |
| Pairing | Each metric becomes a per-date series (equal market weights), subtracted date by date on the three runs' common dates; circular block bootstrap, 1,000 replicates, seed 11; blocks 5 for h = 1, 20 for the holding cell and h = 10 spreads |
| Targets | Residual = `residual_hNN` of each run's `targets.parquet` (the §13.16 residual target, identical across runs); raw = `alpha_hNN` of the same file (next open to the close h sessions later) |
| Universes | all = every scored stock per market (the §13.13 cell, about 460 names per market on average, about 90 in the winners segment); liquidity = the §13.14 per-market liquidity universe (20-session median dollar volume ≥ NT$30M TWSE / 10M TPEX, close ≥ NT$10, no ETFs, about 360 names per market, about 70 winners); small_mid = liquidity minus the §13.15 liquid stock-futures universe (about 300 names per market, about 60 winners); futures_liquid = that universe, both markets together (about 130 names on average, about 26 winners; TAIFEX `taifex-all`, 862 sessions, 314 contracts) |
| Metrics | Spearman Rank IC (holding cell = mean over h 5–10; short cell = h 1); top-minus-bottom quartile spreads (h 1, 5, 10, per cohort); bottom-quartile excess = raw target of the names ranked in the bottom quartile minus the universe group's mean on the date; industry-neutral, dollar-neutral long-short (§13.17 step 0's `quantile_long_short`, h = 10, segment all) |
| Decisions (pre-registered) | H-H: (i) full-universe holding-cell residual ΔIC lower bound > 0 and (ii) liquidity h = 10 residual spread F2 ≥ 1.5 × F0 (F0 > 0); G1: in liquidity or small_mid, ΔIC lower bound > 0 and F2's spread ≥ F0's |
| Sub-analysis | Winners = top fifth of the universe group by the trailing 20-session (adjusted close) return on the signal date; rest = everyone else; at least 20 names per (date, group) |
| Code and outputs | `backtest/twbacktest/chip_increment.py` (synthetic test `backtest/tests/test_chip_increment.py`); outputs in `artifacts/diagnostics/backtest/chip-increment-f0-f1-f2-20261027T210000Z/` (`grid.csv` 1,440 level rows, `paired.csv` 960 paired rows, `daily.csv`, `report.json`, `summary.md`) |

### 3. Results

#### 3.1 Verdicts (P1 ∪ P2, segment all)

| Decision | Condition | Value | Outcome |
| --- | --- | --- | --- |
| H-H (i) | full-universe holding-cell residual ΔIC (F2 − F0) lower bound > 0 | +0.0173 [+0.0128, +0.0214] | holds |
| H-H (ii) | liquidity h = 10 residual quartile spread F2 ≥ 1.5 × F0 | 1.04% vs 0.65% (1.59×) | holds |
| **H-H** | both | | **pass** |
| G1 (a) | liquidity ΔIC lower bound > 0 and spread not below F0 | +0.0202 [+0.0149, +0.0250]; 1.04% ≥ 0.65% | holds |
| G1 (b) | small_mid, same | +0.0191 [+0.0137, +0.0241]; 1.08% ≥ 0.73% | holds |
| **G1** | either | | **pass** |
| F1 − F0 | same conditions | ΔIC +0.0045 [−0.0035, +0.0120]; 0.83% vs 0.65% (1.28×) | fails H-H and G1 (descriptive only) |

#### 3.2 Levels: holding-cell residual Rank IC and h = 10 residual quartile spread (segment all)

| Universe | Period | F0 IC [95%] | F1 IC | F2 IC | F0 spread / cohort | F1 spread | F2 spread | F0 industry-neutral gross annual | F2 industry-neutral gross annual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | P1 | 0.095 [0.082, 0.108] | 0.091 | 0.111 | +0.93% | +0.85% | +1.15% | +12.4% | +15.2% |
| all | P2 | 0.065 [0.046, 0.086] | 0.090 | 0.084 | +0.17% | +0.82% | +0.73% | −0.1% | +6.4% |
| all | P1 ∪ P2 | 0.086 [0.074, 0.100] | 0.091 | 0.104 | +0.72% | +0.84% | +1.04% | +8.8% | +12.7% |
| liquidity | P1 | 0.091 [0.077, 0.106] | 0.091 | 0.111 | +0.88% | +0.87% | +1.18% | +11.6% | +15.2% |
| liquidity | P2 | 0.058 [0.036, 0.082] | 0.083 | 0.079 | +0.06% | +0.73% | +0.65% | +0.1% | +6.6% |
| liquidity | P1 ∪ P2 | 0.082 [0.067, 0.096] | 0.089 | 0.102 | +0.65% | +0.83% | +1.04% | +8.4% | +12.9% |
| small_mid | P1 | 0.094 [0.079, 0.109] | 0.095 | 0.114 | +0.91% | +0.92% | +1.24% | +12.1% | +16.3% |
| small_mid | P2 | 0.067 [0.044, 0.093] | 0.085 | 0.083 | +0.28% | +0.72% | +0.68% | +1.6% | +6.6% |
| small_mid | P1 ∪ P2 | 0.086 [0.072, 0.100] | 0.092 | 0.105 | +0.73% | +0.87% | +1.08% | +9.2% | +13.6% |
| futures_liquid | P1 | 0.071 [0.048, 0.095] | 0.063 | 0.084 | +0.52% | +0.50% | +0.73% | +6.4% | +10.3% |
| futures_liquid | P2 | 0.010 [−0.024, +0.046] | 0.060 | 0.049 | −1.17% | +0.04% | −0.19% | −9.7% | −0.3% |
| futures_liquid | P1 ∪ P2 | 0.054 [0.032, 0.076] | 0.063 | 0.075 | +0.05% | +0.37% | +0.48% | +2.0% | +7.4% |

Industry-neutral long-short gross annual = spread per cohort × 0.5 × 252 / 10 (gross notional = capital, half per side), before costs.

#### 3.3 Paired differences F2 − F0 and F1 − F0 (segment all)

| Pair | Universe | Period | ΔIC residual holding [95%] | ΔIC raw holding | ΔIC residual short (h 1) | Δ residual spread h 10 [95%] | Δ bottom-quartile excess h 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 − F0 | all | P1 | +0.016 [+0.011, +0.021] | +0.016 | +0.013 | +0.23% [+0.11%, +0.33%] | −0.12% |
| F2 − F0 | all | P2 | +0.020 [+0.013, +0.026] | +0.026 | +0.012 | +0.56% [+0.32%, +0.83%] | −0.39% |
| F2 − F0 | all | P1 ∪ P2 | **+0.017 [+0.013, +0.021]** | +0.018 | +0.013 | **+0.32% [+0.20%, +0.44%]** | −0.20% |
| F2 − F0 | liquidity | P1 | +0.020 [+0.014, +0.026] | +0.019 | +0.015 | +0.31% [+0.19%, +0.42%] | −0.17% |
| F2 − F0 | liquidity | P2 | +0.021 [+0.013, +0.029] | +0.025 | +0.014 | +0.58% [+0.32%, +0.87%] | −0.37% |
| F2 − F0 | liquidity | P1 ∪ P2 | **+0.020 [+0.015, +0.025]** | +0.021 | +0.015 | **+0.38% [+0.26%, +0.51%]** | −0.23% |
| F2 − F0 | small_mid | P1 | +0.021 [+0.015, +0.026] | +0.021 | +0.016 | +0.33% [+0.20%, +0.45%] | −0.20% |
| F2 − F0 | small_mid | P2 | +0.016 [+0.006, +0.024] | +0.017 | +0.013 | +0.40% [+0.16%, +0.65%] | −0.22% |
| F2 − F0 | small_mid | P1 ∪ P2 | **+0.019 [+0.014, +0.024]** | +0.020 | +0.015 | **+0.35% [+0.23%, +0.47%]** | −0.21% |
| F2 − F0 | futures_liquid | P1 | +0.014 [−0.005, +0.032] | +0.002 | +0.013 | +0.22% [−0.08%, +0.50%] | −0.00% |
| F2 − F0 | futures_liquid | P2 | +0.039 [+0.016, +0.060] | +0.038 | +0.015 | +0.98% [+0.37%, +1.62%] | −0.40% |
| F2 − F0 | futures_liquid | P1 ∪ P2 | +0.021 [+0.006, +0.035] | +0.012 | +0.013 | +0.43% [+0.13%, +0.74%] | −0.11% |
| F1 − F0 | all | P1 | −0.003 [−0.012, +0.005] | −0.009 | −0.005 | −0.08% [−0.25%, +0.09%] | +0.06% |
| F1 − F0 | all | P2 | **+0.025 [+0.014, +0.034]** | +0.020 | +0.002 | +0.64% [+0.37%, +0.94%] | −0.31% |
| F1 − F0 | all | P1 ∪ P2 | +0.004 [−0.004, +0.012] | −0.001 | −0.003 | +0.12% [−0.06%, +0.30%] | −0.04% |
| F1 − F0 | liquidity | P1 | +0.000 [−0.009, +0.010] | −0.004 | −0.001 | −0.00% [−0.19%, +0.18%] | −0.00% |
| F1 − F0 | liquidity | P2 | **+0.025 [+0.013, +0.036]** | +0.021 | +0.005 | +0.66% [+0.34%, +1.01%] | −0.31% |
| F1 − F0 | liquidity | P1 ∪ P2 | +0.007 [−0.002, +0.016] | +0.003 | +0.000 | +0.18% [−0.01%, +0.39%] | −0.09% |
| F1 − F0 | small_mid | P1 ∪ P2 | +0.006 [−0.003, +0.015] | +0.003 | +0.001 | +0.13% [−0.05%, +0.32%] | −0.04% |
| F1 − F0 | small_mid | P2 | +0.018 [+0.006, +0.028] | +0.013 | +0.003 | +0.45% [+0.20%, +0.67%] | −0.13% |
| F1 − F0 | futures_liquid | P1 ∪ P2 | +0.009 [−0.012, +0.028] | −0.012 | +0.002 | +0.32% [−0.07%, +0.76%] | +0.01% |
| F1 − F0 | futures_liquid | P2 | +0.050 [+0.021, +0.080] | +0.038 | +0.012 | +1.21% [+0.35%, +2.21%] | −0.38% |

#### 3.4 G1 sub-analysis: recent winners versus the rest (F2 − F0, holding-cell residual ΔIC and h = 10 metrics)

| Universe | Segment | Period | ΔIC residual holding [95%] | Δ residual spread h 10 [95%] | Bottom-quartile excess F0 | Bottom-quartile excess F2 [95%] | Δ bottom-quartile excess [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| liquidity | winners | P1 | +0.029 [+0.021, +0.039] | +0.55% [+0.26%, +0.85%] | −0.56% | −0.94% [−1.42%, −0.49%] | −0.38% [−0.62%, −0.16%] |
| liquidity | winners | P2 | +0.029 [+0.013, +0.046] | +1.05% [+0.26%, +1.92%] | +0.86% | +0.30% [−0.81%, +1.43%] | −0.56% [−1.06%, −0.10%] |
| liquidity | winners | P1 ∪ P2 | **+0.029 [+0.021, +0.037]** | **+0.69% [+0.37%, +1.01%]** | −0.17% | **−0.60% [−1.09%, −0.08%]** | −0.43% [−0.64%, −0.23%] |
| liquidity | rest | P1 | +0.019 [+0.011, +0.025] | +0.25% [+0.10%, +0.39%] | −0.58% | −0.69% | −0.11% [−0.22%, +0.01%] |
| liquidity | rest | P2 | +0.017 [+0.005, +0.027] | +0.42% [+0.14%, +0.68%] | −0.11% | −0.31% | −0.19% [−0.40%, +0.01%] |
| liquidity | rest | P1 ∪ P2 | +0.018 [+0.012, +0.024] | +0.30% [+0.16%, +0.43%] | −0.45% | −0.58% | −0.13% [−0.23%, −0.03%] |
| small_mid | winners | P2 | +0.019 [+0.003, +0.033] | +0.54% [−0.09%, +1.17%] | +0.41% | +0.26% | −0.15% [−0.51%, +0.22%] |
| small_mid | winners | P1 ∪ P2 | +0.025 [+0.017, +0.034] | +0.52% [+0.20%, +0.84%] | −0.35% | −0.64% | −0.29% [−0.50%, −0.08%] |
| small_mid | rest | P2 | +0.011 [+0.001, +0.021] | +0.32% [+0.07%, +0.57%] | −0.07% | −0.19% | −0.12% [−0.31%, +0.06%] |
| small_mid | rest | P1 ∪ P2 | +0.017 [+0.011, +0.023] | +0.31% [+0.16%, +0.44%] | −0.44% | −0.59% | −0.15% [−0.25%, −0.04%] |
| futures_liquid | winners | P2 | +0.034 [−0.016, +0.091] | +0.84% [−1.01%, +3.07%] | | | −0.11% [−1.23%, +0.83%] |
| futures_liquid | winners | P1 ∪ P2 | +0.029 [+0.001, +0.058] | +0.83% [−0.03%, +1.70%] | | | −0.47% [−0.98%, +0.02%] |
| futures_liquid | rest | P2 | +0.042 [+0.019, +0.065] | +0.81% [+0.37%, +1.25%] | | | −0.20% [−0.57%, +0.15%] |
| futures_liquid | rest | P1 ∪ P2 | +0.019 [+0.003, +0.035] | +0.27% [−0.03%, +0.55%] | | | +0.06% [−0.11%, +0.23%] |

F1 − F0 sub-analysis (liquidity universe, P1 ∪ P2): winners ΔIC +0.020 [+0.008, +0.032], rest +0.004 [−0.007, +0.015] — chip data alone adds almost only among the winners; in P2 both segments are +0.023. The winners segment of the liquid stock-futures universe holds about 25–30 names a day, hence the wide intervals.

### 4. Reading

- **The pre-registered expectation**: "if the chip mechanism holds, F2's improvement in the P2 winners should exceed the other groups". In P2 the liquidity universe shows +0.029 for winners vs +0.017 for the rest, small / mid caps +0.019 vs +0.011 — the right direction — but the P1 winners improve by as much (+0.029) and the rest improves significantly in both periods. The chip features therefore carry a period-independent independent signal together with stronger discrimination among recent winners; they are not a conditional reversal that appears only in P2.
- **F1's shape**: chip data alone ties price / volume in P1, wins in P2 and stays slightly behind in the short cell (h 1). P2 (after 2025-06) is where the price / volume model's residual ranking decays most, and the chip features fill exactly that gap; how much of F2's increment carries into P3 therefore depends on whether P2's pattern persists.
- **The margin on H-H (ii)**: 1.59× is 0.09 above the bar; the full universe (1.44×) and the small / mid caps (1.48×) would fail it. The verdict follows the universe fixed beforehand; the sensitivity is listed as a limitation.
- **The shortable universe**: F2 lifts the liquid stock-futures universe's P2 residual IC from 0.01 to 0.05 and its quartile spread from −1.17% to −0.19%, still negative; §13.17 step 0's "the alpha is not in the shortable universe" becomes, under F2, "the shortable universe's ranking no longer fails, but has no tradeable magnitude".
- **Limitations**: P2 has been seen and §13.18 itself was designed after the §13.16–§13.17 results, so the passes are informational and the formal decision is P3; three-seed ensemble on one walk-forward; the chip column `FDLR_Z60` is always missing (known); publication times use the conservative rule (before the T+1 open, implementation note (b)) and any dataset found to publish later must be shifted a day and rerun.

### 5. Dispositions

| Item | Protocol basis | Disposition |
| --- | --- | --- |
| H-H | §13.18 item 5 | Pass; the item 7 strategy layer (S-A, S-C on F2 scores, both cost profiles) is registered as the next step, S-B waits for G2 |
| G1 | §13.18 items 5 and 10 | Pass; the condition for buying broker-branch data (FinMind Sponsor Pro, one month) is met, the purchase is the user's decision |
| F1 | §13.18 item 5 | Descriptive only: chip data does not replace price / volume (P1 ∪ P2 fails) but beats it in P2 |
| D0 on F2 | §13.18 item 3 | Decided separately; this project recommends the strategy layer first |
| G2 | §13.18 item 6 | Runs as planned and decides S-B |
