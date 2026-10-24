# 殘差化標籤與可抽換編碼器結果 / Residualized Labels and Pluggable Encoder Results

## 中文

### 1. 目的與結論

本報告記錄[實驗協定 §13.16](experiment_protocol.md) 的全部結果：把 gauss-rank 標籤改為「對市值代理與產業取殘差後再排名」的標籤變體，先以 B0 `t-lgbm-alpha` 驗證（H-F），再把 D0 `t-xs-scratch` 與兩個凍結編碼器 D1（Kronos-small、Chronos-2-small）在新標籤上重訓，並記錄事前登記的 seed 穩定度規則與判定。所有數字來自 RunPod 上的 walk-forward 輸出（14 個季度 fold，P1 2023-01 至 2025-05、P2 2025-06 至 2026-05），本機只做評分與配對。

結論：

1. **H-F 通過（變體 A）。** B0 在殘差目標上的 P2 持有格 Rank IC 0.0646 [0.0457, 0.0856]，區間不跨 0；h = 1 對原始 alpha 的 Rank IC 與舊 B0 的配對差 −0.0002 [−0.0013, +0.0009]，沒有受損。殘差化把 P2 多日排序從「區間跨 0」救回「下界明顯大於 0」。
2. **崩掉的是標籤裡的暴露，不是模型。** 舊標籤下 B0／D0 的 P2 持有格對原始 alpha 只有 0.034／0.047，長格 0.011／0.036；對殘差目標，新標籤下 B0 為 0.065／0.055、D0 為 0.068／0.062，P1→P2 保留率從一到四成回到六到七成。對原始 alpha 的 P2 多日 Rank IC 仍只有 P1 的四到五成，因為 P2 的原始報酬被市值與產業因子主導，而模型現在刻意不預測這兩個因子。
3. **變體 B 形式上通過但每格都劣於變體 A**，持有格 P2 對舊 B0 的配對差 −0.0066 [−0.0102, −0.0029]；以快照股數乘當日收盤價當市值代理，把價格水準當成規模因子殘差化掉，反而移走可預測成分。依第 5 點，只用變體 A。
4. **D0 在新標籤上重訓：H-A 仍只有短格通過。** 對同標籤 B0 的配對差短格 +0.0037 [+0.0011, +0.0065]、持有格 +0.0037 [−0.0033, +0.0106]。殘差目標上 B0 與 D0 幾乎相同（P2 持有格 0.0646 對 0.0676）；跨股票 DL 對特徵 GBDT 的增量仍只在 h = 1、幅度 0.003–0.004。
5. **三 seed 穩定，D1 改跑單 seed。** D0 三格的 seed 極差 0.0024–0.0083、集成增益 0.0034–0.0036，都小於集成 95% 區間半寬的一半（事前規則，第 9 點）。
6. **兩個凍結編碼器都沒有增量。** Kronos-small（單 seed）與 D0 單 seed 完全相當（短格 0.1172 對 0.1171），對 D0 三 seed 集成的短格差 −0.0040 [−0.0069, −0.0014] 等於集成增益；Chronos-2-small 每一格都略劣於 D0 集成（−0.005，區間不跨 0）、持有格劣於 Kronos（−0.0076 [−0.0128, −0.0020]），對 B0 的殘差目標配對差也為負。這與 §13.11（Kronos ≈ PatchTST）、§13.13 A1（跨股票 attention 無增量）一致：特徵 GBDT 已取走絕大部分可預測性。
7. **模型線結論**：B0 為策略基線，D0 三 seed 集成並列候選訊號；D1 不進入策略層；`t-xs-kronos-base-lora` 以「前者有效」為前提，不執行。策略層的後續見 [§13.17 步驟 0 報告](exposure_diagnostics_results.md)。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 資料、特徵、walk-forward | 與 §13.12、§13.13 相同（資料集 `53e3915b…`，183 欄特徵庫，14 季度 fold，15 日 embargo）；原標籤庫不變 |
| 公司參考表 | `configs/reference/tw_company_reference_20261022.csv`：證交所 OpenAPI `t187ap03_L` 與櫃買 OpenAPI `mopsfin_t187ap03_O` 2026-10-22 快照（1,097 上市 + 894 上櫃）：產業別兩碼、已發行股數；覆蓋 2026-05 仍交易標的 99.2%／99.8%，全部樣本 90%／93% |
| 殘差化標籤（schema `cross-sectional-gauss-rank-residual-v1`） | 每個（訊號日, 市場）群組內，各 horizon 的原始 alpha 目標對截距、市值代理、產業虛擬變數做 OLS，殘差再 gauss-rank；群組 < 30 檔或某產業只有 1 檔時併入「其他」。變體 A：市值代理 = 訊號日前 20 日成交金額中位數的對數，產業缺值為 UNK；變體 B：log（原始收盤價 × 快照股數），缺值回退變體 A。identity 含規格與參考表 digest，store 位於既有標籤庫旁 |
| B0 | 設定、超參數、seed（42、43、44）同 §13.13；RunPod CPU；588 個模型／變體；模型名 `t-lgbm-alpha-residual_v1a`／`…_v1b` |
| D0 | 同 §13.13（patch encoder + 特徵側輸入 + 股票間 attention + 市場門控，645,518 參數，MSE + 1 × (1 − Pearson)），三 seed；殘差變體的 rank 與殘差目標陣列寫在跨股票快取旁 `xs-cache/<identity>/label-variants/<digest>/`，快取與 embedding 快取不重建；配對基線為同標籤 B0 |
| D1 | 凍結編碼器取代 patch encoder，embedding 以 (symbol, cutoff) 快取一次：`t-xs-kronos-small`（`NeoQuasar/Kronos-small` 901c26c、Tokenizer-base 0e01173，取最後位置 hidden state，維度 512）、`t-xs-chronos2-small`（`autogluon/chronos-2-small` ddec013，五個 OHLCV 通道為多變量序列，對 variates 與 patches 平均池化，維度 512）；跨股票模組 438,670 參數；seed 42 |
| 指標 | Spearman Rank IC 同時對殘差目標與原始 alpha 目標計算；cell short（h = 1）／holding（5–10）／long（11–14），兩市場等權，P1／P2／P1 ∪ P2；circular block bootstrap 1,000 次、seed 11、區塊 5／20／28；配對差在同一訊號日相減 |
| 判定 | H-F（第 5 點）：變體 A 的 B0，(i) P2 持有格對殘差目標的下界 > 0 且 (ii) P1 ∪ P2 短格對原始 alpha 的配對差下界 > −0.01。H-A（§13.13）：對同標籤 B0 的原始 alpha 配對差，短格與持有格下界 > 0 |
| 硬體 | B0：RunPod CPU Pod；D0：RTX PRO 4500 Blackwell（42 個 fit 32,676 秒；GPU 利用率 7–9%、時脈正常，主機 CPU 開銷為瓶頸）；D1：RTX 5090（Kronos 14 個 fit 1,500 秒；Chronos-2 1,151 秒，trainer 移除每步主機同步後每 epoch 由 13.4 秒降至 9.2 秒，訓練階段 GPU 利用率 25–33% → 44–48%，embedding 階段 92–100%／56–60%） |

### 3. 結果

#### 3.1 B0：標籤變體對舊 B0（三 seed 集成，[95% 區間]）

| cell | 期間 | 舊 B0（原始 alpha） | 變體 A 原始 alpha | 變體 A 殘差目標 | 變體 A − 舊 B0 | 變體 B 原始 alpha | 變體 B 殘差目標 | 變體 B − 舊 B0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | P1 | 0.1248 | 0.1242 | 0.1049 | −0.0005 [−0.0018, +0.0007] | 0.1203 | 0.1029 | −0.0045 [−0.0063, −0.0028] |
| short | P2 | 0.0996 | 0.1001 | 0.0902 [0.0805, 0.1010] | +0.0005 [−0.0018, +0.0027] | 0.0963 | 0.0860 | −0.0033 [−0.0057, −0.0009] |
| short | P1 ∪ P2 | 0.1178 | 0.1175 | 0.1008 [0.0954, 0.1061] | **−0.0002 [−0.0013, +0.0009]** | 0.1136 | 0.0982 | −0.0042 [−0.0056, −0.0026] |
| holding | P1 | 0.1057 | 0.1061 | 0.0947 | +0.0005 [−0.0022, +0.0031] | 0.1048 | 0.0942 | −0.0009 [−0.0042, +0.0024] |
| holding | P2 | 0.0343 [−0.0004, 0.0726] | 0.0434 [0.0103, 0.0794] | **0.0646 [0.0457, 0.0856]** | +0.0090 [+0.0055, +0.0125] | 0.0277 [−0.0061, 0.0637] | 0.0483 [0.0290, 0.0710] | −0.0066 [−0.0102, −0.0029] |
| holding | P1 ∪ P2 | 0.0859 | 0.0888 | 0.0864 [0.0740, 0.0996] | +0.0028 [+0.0006, +0.0053] | 0.0834 | 0.0815 | −0.0025 [−0.0055, +0.0001] |
| long | P1 | 0.1112 | 0.1119 | 0.0990 | +0.0007 [−0.0031, +0.0049] | 0.1098 | 0.0972 | −0.0013 [−0.0063, +0.0031] |
| long | P2 | 0.0107 [−0.0372, 0.0600] | 0.0239 [−0.0216, 0.0692] | 0.0549 [0.0324, 0.0763] | +0.0132 [+0.0084, +0.0179] | 0.0023 [−0.0469, 0.0500] | 0.0346 [0.0111, 0.0580] | −0.0084 [−0.0129, −0.0035] |
| long | P1 ∪ P2 | 0.0834 | 0.0876 | 0.0868 [0.0700, 0.1035] | +0.0042 [+0.0005, +0.0080] | 0.0801 | 0.0798 | −0.0033 [−0.0073, +0.0005] |

H-F：變體 A 條件 (i) 0.0457 > 0、條件 (ii) −0.0013 > −0.01，**通過**；變體 B (i) 0.0290、(ii) −0.0056，形式上通過，但每格劣於變體 A，依第 5 點不採用。

#### 3.2 D0 與 D1 在變體 A 標籤上（對原始 alpha／對殘差目標）

| 模型 | seeds | cell | P1 | P2 | P1 ∪ P2 |
| --- | --- | --- | --- | --- | --- |
| B0 變體 A | 3 | short | 0.1242 ／ 0.1049 | 0.1001 ／ 0.0902 | 0.1175 ／ 0.1008 |
| | | holding | 0.1061 ／ 0.0947 | 0.0434 ／ 0.0646 | 0.0888 ／ 0.0864 |
| | | long | 0.1119 ／ 0.0990 | 0.0239 ／ 0.0549 | 0.0876 ／ 0.0868 |
| D0 `t-xs-scratch` | 3 | short | 0.1277 ／ 0.1069 | 0.1044 ／ 0.0946 | 0.1212 [0.1119, 0.1299] ／ 0.1035 |
| | | holding | 0.1076 ／ 0.0938 | 0.0528 [0.0218, 0.0860] ／ 0.0676 [0.0465, 0.0899] | 0.0925 ／ 0.0866 |
| | | long | 0.1112 ／ 0.0964 | 0.0396 [0.0037, 0.0737] ／ 0.0620 [0.0377, 0.0858] | 0.0914 ／ 0.0869 |
| D1 `t-xs-kronos-small` | 1 | short | 0.1241 ／ 0.1038 | 0.0993 ／ 0.0908 | 0.1172 ／ 0.1002 |
| | | holding | 0.1100 ／ 0.0962 | 0.0549 ／ 0.0710 [0.0480, 0.0950] | 0.0948 ／ 0.0893 |
| | | long | 0.1109 ／ 0.0965 | 0.0444 ／ 0.0683 | 0.0925 ／ 0.0887 |
| D1 `t-xs-chronos2-small` | 1 | short | 0.1222 ／ 0.1018 | 0.0995 ／ 0.0891 | 0.1159 ／ 0.0983 |
| | | holding | 0.1001 ／ 0.0882 | 0.0535 ／ 0.0648 | 0.0872 ／ 0.0817 |
| | | long | 0.1032 ／ 0.0907 | 0.0400 ／ 0.0583 | 0.0857 ／ 0.0817 |

#### 3.3 配對差（同一訊號日相減，[95% 區間]）

| 比較 | 目標 | short P1 ∪ P2 | holding P1 ∪ P2 | long P1 ∪ P2 | holding P2 |
| --- | --- | --- | --- | --- | --- |
| D0 − B0（同標籤） | 原始 alpha | **+0.0037 [+0.0011, +0.0065]** | +0.0037 [−0.0033, +0.0106] | +0.0038 [−0.0078, +0.0162] | +0.0094 [−0.0029, +0.0269] |
| D0 − B0 | 殘差 | +0.0027 [+0.0004, +0.0049] | +0.0002 [−0.0043, +0.0046] | +0.0001 [−0.0070, +0.0073] | +0.0030 [−0.0029, +0.0111] |
| Kronos D1 − B0 | 原始 alpha | −0.0003 [−0.0037, +0.0030] | +0.0060 [−0.0017, +0.0137] | +0.0049 [−0.0068, +0.0170] | +0.0116 [−0.0034, +0.0305] |
| Kronos D1 − D0 集成 | 原始 alpha | **−0.0040 [−0.0069, −0.0014]** | +0.0023 [−0.0018, +0.0065] | +0.0011 [−0.0044, +0.0074] | +0.0022 [−0.0045, +0.0097] |
| Kronos D1 − D0 集成 | 殘差 | −0.0033 [−0.0055, −0.0013] | +0.0027 [−0.0003, +0.0057] | +0.0019 [−0.0026, +0.0064] | +0.0034 [−0.0015, +0.0083] |
| Chronos-2 − B0 | 原始 alpha | −0.0016 [−0.0044, +0.0012] | −0.0016 [−0.0089, +0.0065] | −0.0019 [−0.0137, +0.0116] | +0.0102 [−0.0049, +0.0281] |
| Chronos-2 − B0 | 殘差 | **−0.0026 [−0.0048, −0.0003]** | **−0.0047 [−0.0096, −0.0002]** | −0.0050 [−0.0114, +0.0009] | +0.0002 [−0.0059, +0.0065] |
| Chronos-2 − D0 集成 | 原始 alpha | −0.0053 [−0.0073, −0.0034] | −0.0053 [−0.0091, −0.0013] | −0.0057 [−0.0105, −0.0006] | +0.0008 [−0.0071, +0.0089] |
| Chronos-2 − Kronos D1 | 原始 alpha | −0.0013 [−0.0047, +0.0021] | **−0.0076 [−0.0128, −0.0020]** | −0.0068 [−0.0135, +0.0001] | −0.0014 [−0.0124, +0.0096] |

舊標籤 D0 對舊 B0（§13.13）：短格 +0.0043 [+0.0015, +0.0071]、持有格 +0.0062 [−0.0012, +0.0136]；新舊標籤下 DL 的增量相同。

#### 3.4 Seed 穩定度（D0，P1 ∪ P2，原始 alpha）

| cell | seed 42／43／44 | 極差 | 集成 | 集成增益 | 集成 95% 半寬 | 單 seed 足夠 |
| --- | --- | --- | --- | --- | --- | --- |
| short | 0.1171／0.1168／0.1191 | 0.0024 | 0.1212 | 0.0036 | 0.0090 | 是 |
| holding | 0.0906／0.0846／0.0921 | 0.0075 | 0.0925 | 0.0034 | 0.0170 | 是 |
| long | 0.0894／0.0830／0.0913 | 0.0083 | 0.0914 | 0.0035 | 0.0222 | 是 |

規則（第 9 點，事前）：極差與集成增益都小於半寬的一半 → D1 只跑 seed 42。D1 單 seed 與 D0 單 seed 42 並列：短格 0.1172 對 0.1171、持有格 0.0948 對 0.0906、長格 0.0925 對 0.0894；D1 對集成的短格差就是集成增益。

#### 3.5 P1 → P2 的保留率（持有格 Rank IC，P2／P1）

| 模型 | 原始 alpha | 殘差目標 |
| --- | --- | --- |
| 舊 B0 | 0.106 → 0.034（32%） | — |
| 舊 D0 | 0.109 → 0.047（43%） | — |
| B0 變體 A | 0.106 → 0.043（41%） | 0.095 → 0.065（68%） |
| D0 變體 A | 0.108 → 0.053（49%） | 0.094 → 0.068（72%） |
| Kronos D1 | 0.110 → 0.055（50%） | 0.096 → 0.071（74%） |
| Chronos-2 | 0.100 → 0.054（53%） | 0.088 → 0.065（73%） |

### 4. 判讀限制

- 產業與股數是 2026-10-22 的快照；已下市的 7–10% 樣本沒有產業（UNK）與股數，有存活者偏誤。
- 殘差化只移除線性的市值與產業暴露；動能、波動等其他風格仍在目標裡。
- 殘差目標上的 Rank IC 不等於可交易：多頭組合仍承受因子報酬，而殘差 alpha 的幅度要看 §13.17 步驟 0。
- P2 已經看過；殘差化是看到失效之後設計的修補，確認需要 P3。
- D1 兩個編碼器都只有 seed 42；對 D0 集成的配對差包含集成增益（約 0.0036），以單 seed 對單 seed 並列解讀。
- 殘差目標下 D1 長格 P2 對 B0 的 +0.0133 [+0.0022, +0.0287] 是九個格期組合中唯一不跨 0 的，不作為證據。

### 5. 處置

依 §13.16 第 7 點，H-F 通過後 D0 已在新標籤重訓，D1 兩個編碼器已跑；模型線收斂為 B0 基線加 D0 集成候選。策略層依使用者 2026-10-24 設定的門檻（勝過 0050 買進持有）登記為 §13.17，其步驟 0 的診斷結果與處置見[§13.17 步驟 0 報告](exposure_diagnostics_results.md)。

產物：B0 變體 A `artifacts/lgbm/launch-20261023T103805Z-2637116736/`、變體 B `launch-20261023T130541Z-2860518787/`（各含 `label-variant-score.json`）；D0 `artifacts/xs/launch-20261023T162203Z-877225215/`；Kronos `launch-20261024T074202Z-2401818372/`、Chronos-2 `launch-20261024T104417Z-980120111/`（各含 `paired-vs-xs.json`／`paired-vs-d0.json`、`paired-vs-d1.json`）。

## English

### 1. Purpose and conclusions

This report records every result of [experiment protocol §13.16](experiment_protocol.md): replacing the gauss-rank labels with a variant that ranks the *residual* of the alpha target after a cross-sectional regression on a size proxy and industry dummies, validated first with B0 `t-lgbm-alpha` (H-F), then retraining D0 `t-xs-scratch` and the two frozen D1 encoders (Kronos-small, Chronos-2-small) on the new labels, with the pre-registered seed-stability rule. All numbers come from walk-forward outputs produced on RunPod (14 quarterly folds, P1 2023-01 to 2025-05, P2 2025-06 to 2026-05); the local machine only scores and pairs them.

Conclusions:

1. **H-F passes (variant A).** B0's P2 holding-cell Rank IC against the residual target is 0.0646 [0.0457, 0.0856], an interval clear of 0; the h = 1 Rank IC against the raw alpha moves by −0.0002 [−0.0013, +0.0009] versus the old B0, so nothing is lost. Residualization turns the P2 multi-day ranking from "interval spans 0" into "lower bound well above 0".
2. **What collapsed was the exposure inside the labels, not the model.** Under the old labels the P2 holding cells of B0 / D0 against raw alpha were 0.034 / 0.047 and the long cells 0.011 / 0.036; against the residual target under the new labels they are 0.065 / 0.055 for B0 and 0.068 / 0.062 for D0, and the P1 → P2 retention rises from 10–40% to 60–70%. Against raw alpha the P2 multi-day Rank IC is still only 40–50% of P1, because P2's raw returns are driven by the size and industry factors the model now deliberately ignores.
3. **Variant B passes formally but trails variant A in every cell**; its P2 holding cell is −0.0066 [−0.0102, −0.0029] against the old B0. A snapshot share count times the day's close uses the price level as the size factor and removes predictable variation. Variant A is the only one carried forward (item 5).
4. **D0 retrained on the new labels: H-A still passes only the short cell.** Paired against the same-label B0: short +0.0037 [+0.0011, +0.0065], holding +0.0037 [−0.0033, +0.0106]. On the residual target B0 and D0 are nearly identical (P2 holding 0.0646 vs 0.0676); the cross-stock DL's increment over the feature GBDT remains h = 1 only and 0.003–0.004 in size.
5. **Seeds are stable, so D1 runs one seed.** D0's seed range is 0.0024–0.0083 and the ensemble gain 0.0034–0.0036 in all three cells, both below half the ensemble's 95% half-width (pre-registered rule, item 9).
6. **Neither frozen encoder adds anything.** Kronos-small (one seed) equals a single D0 seed (short 0.1172 vs 0.1171); its −0.0040 [−0.0069, −0.0014] against the three-seed D0 ensemble is the ensemble gain. Chronos-2-small is slightly worse than the D0 ensemble in every cell (−0.005, intervals clear of 0), worse than Kronos in the holding cell (−0.0076 [−0.0128, −0.0020]) and worse than B0 on the residual target. This repeats §13.11 (Kronos ≈ PatchTST) and §13.13 A1 (inter-stock attention adds nothing): the feature GBDT already captures almost all of the predictability.
7. **Model line conclusion**: B0 is the strategy baseline, the three-seed D0 ensemble is the co-candidate signal; D1 does not enter the strategy layer; `t-xs-kronos-base-lora` is not run because its precondition (the frozen encoder helping) failed. The strategy layer continues in the [§13.17 step 0 report](exposure_diagnostics_results.md).

### 2. Data and method

| Item | Content |
| --- | --- |
| Data, features, walk-forward | Identical to §13.12 and §13.13 (dataset `53e3915b…`, 183-column feature library, 14 quarterly folds, 15-session embargo); the standard label store is untouched |
| Company reference | `configs/reference/tw_company_reference_20261022.csv`: TWSE OpenAPI `t187ap03_L` and TPEX OpenAPI `mopsfin_t187ap03_O` snapshot of 2026-10-22 (1,097 TWSE + 894 TPEX): two-digit industry code and issued shares; covers 99.2% / 99.8% of names trading in 2026-05 and 90% / 93% of all samples |
| Residualized labels (schema `cross-sectional-gauss-rank-residual-v1`) | Within every (signal date, market) group, each horizon's raw alpha target is regressed on an intercept, the size proxy and industry dummies; the residual is gauss-ranked. Groups below 30 names or industries with one name fold into "other". Variant A: size proxy = log of the 20-session median dollar volume, missing industry = UNK; variant B: log(raw close × snapshot shares), falling back to A. The identity includes the spec and reference digests; the store sits beside the standard one |
| B0 | Settings, hyper-parameters and seeds (42, 43, 44) as in §13.13; RunPod CPU; 588 models per variant; model names `t-lgbm-alpha-residual_v1a` / `…_v1b` |
| D0 | As in §13.13 (patch encoder + feature side input + inter-stock attention + market gating, 645,518 parameters, MSE + 1 × (1 − Pearson)), three seeds; the variant's ranks and residual targets live beside the cross-stock cache under `xs-cache/<identity>/label-variants/<digest>/`, so neither the cache nor the embedding caches are rebuilt; paired against the same-label B0 |
| D1 | A frozen encoder replaces the patch encoder; embeddings are cached once per (symbol, cutoff): `t-xs-kronos-small` (`NeoQuasar/Kronos-small` 901c26c, Tokenizer-base 0e01173, last-position hidden state, dim 512) and `t-xs-chronos2-small` (`autogluon/chronos-2-small` ddec013, the five OHLCV channels as one multivariate series, mean-pooled over variates and patches, dim 512); cross-stock module 438,670 parameters; seed 42 |
| Metrics | Spearman Rank IC against both the residual and the raw alpha target; cells short (h = 1) / holding (5–10) / long (11–14), markets equally weighted, P1 / P2 / P1 ∪ P2; circular block bootstrap, 1,000 replicates, seed 11, blocks 5 / 20 / 28; paired differences on common signal dates |
| Decisions | H-F (item 5): variant A's B0, (i) P2 holding cell against the residual target has a lower bound > 0 and (ii) the P1 ∪ P2 short cell's paired difference against the old B0 has a lower bound > −0.01. H-A (§13.13): paired differences against the same-label B0 on raw alpha, lower bounds > 0 in the short and holding cells |
| Hardware | B0: RunPod CPU Pod; D0: RTX PRO 4500 Blackwell (42 fits, 32,676 s; GPU utilization 7–9% at normal clocks, host-CPU bound); D1: RTX 5090 (Kronos 14 fits 1,500 s; Chronos-2 1,151 s after the trainer's per-step host syncs were removed, 13.4 → 9.2 s per epoch, training-phase GPU utilization 25–33% → 44–48%, embedding phase 92–100% / 56–60%) |

### 3. Results

#### 3.1 B0: label variants against the old B0 (three-seed ensembles, [95% intervals])

| Cell | Period | Old B0 (raw alpha) | Variant A raw | Variant A residual | Variant A − old B0 | Variant B raw | Variant B residual | Variant B − old B0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | P1 | 0.1248 | 0.1242 | 0.1049 | −0.0005 [−0.0018, +0.0007] | 0.1203 | 0.1029 | −0.0045 [−0.0063, −0.0028] |
| short | P2 | 0.0996 | 0.1001 | 0.0902 [0.0805, 0.1010] | +0.0005 [−0.0018, +0.0027] | 0.0963 | 0.0860 | −0.0033 [−0.0057, −0.0009] |
| short | P1 ∪ P2 | 0.1178 | 0.1175 | 0.1008 [0.0954, 0.1061] | **−0.0002 [−0.0013, +0.0009]** | 0.1136 | 0.0982 | −0.0042 [−0.0056, −0.0026] |
| holding | P1 | 0.1057 | 0.1061 | 0.0947 | +0.0005 [−0.0022, +0.0031] | 0.1048 | 0.0942 | −0.0009 [−0.0042, +0.0024] |
| holding | P2 | 0.0343 [−0.0004, 0.0726] | 0.0434 [0.0103, 0.0794] | **0.0646 [0.0457, 0.0856]** | +0.0090 [+0.0055, +0.0125] | 0.0277 [−0.0061, 0.0637] | 0.0483 [0.0290, 0.0710] | −0.0066 [−0.0102, −0.0029] |
| holding | P1 ∪ P2 | 0.0859 | 0.0888 | 0.0864 [0.0740, 0.0996] | +0.0028 [+0.0006, +0.0053] | 0.0834 | 0.0815 | −0.0025 [−0.0055, +0.0001] |
| long | P1 | 0.1112 | 0.1119 | 0.0990 | +0.0007 [−0.0031, +0.0049] | 0.1098 | 0.0972 | −0.0013 [−0.0063, +0.0031] |
| long | P2 | 0.0107 [−0.0372, 0.0600] | 0.0239 [−0.0216, 0.0692] | 0.0549 [0.0324, 0.0763] | +0.0132 [+0.0084, +0.0179] | 0.0023 [−0.0469, 0.0500] | 0.0346 [0.0111, 0.0580] | −0.0084 [−0.0129, −0.0035] |
| long | P1 ∪ P2 | 0.0834 | 0.0876 | 0.0868 [0.0700, 0.1035] | +0.0042 [+0.0005, +0.0080] | 0.0801 | 0.0798 | −0.0033 [−0.0073, +0.0005] |

H-F: variant A passes, condition (i) 0.0457 > 0 and (ii) −0.0013 > −0.01; variant B passes formally ((i) 0.0290, (ii) −0.0056) but trails variant A everywhere and is not used (item 5).

#### 3.2 D0 and D1 on the variant A labels (raw alpha / residual target)

| Model | Seeds | Cell | P1 | P2 | P1 ∪ P2 |
| --- | --- | --- | --- | --- | --- |
| B0 variant A | 3 | short | 0.1242 / 0.1049 | 0.1001 / 0.0902 | 0.1175 / 0.1008 |
| | | holding | 0.1061 / 0.0947 | 0.0434 / 0.0646 | 0.0888 / 0.0864 |
| | | long | 0.1119 / 0.0990 | 0.0239 / 0.0549 | 0.0876 / 0.0868 |
| D0 `t-xs-scratch` | 3 | short | 0.1277 / 0.1069 | 0.1044 / 0.0946 | 0.1212 [0.1119, 0.1299] / 0.1035 |
| | | holding | 0.1076 / 0.0938 | 0.0528 [0.0218, 0.0860] / 0.0676 [0.0465, 0.0899] | 0.0925 / 0.0866 |
| | | long | 0.1112 / 0.0964 | 0.0396 [0.0037, 0.0737] / 0.0620 [0.0377, 0.0858] | 0.0914 / 0.0869 |
| D1 `t-xs-kronos-small` | 1 | short | 0.1241 / 0.1038 | 0.0993 / 0.0908 | 0.1172 / 0.1002 |
| | | holding | 0.1100 / 0.0962 | 0.0549 / 0.0710 [0.0480, 0.0950] | 0.0948 / 0.0893 |
| | | long | 0.1109 / 0.0965 | 0.0444 / 0.0683 | 0.0925 / 0.0887 |
| D1 `t-xs-chronos2-small` | 1 | short | 0.1222 / 0.1018 | 0.0995 / 0.0891 | 0.1159 / 0.0983 |
| | | holding | 0.1001 / 0.0882 | 0.0535 / 0.0648 | 0.0872 / 0.0817 |
| | | long | 0.1032 / 0.0907 | 0.0400 / 0.0583 | 0.0857 / 0.0817 |

#### 3.3 Paired differences (common signal dates, [95% intervals])

| Comparison | Target | short P1 ∪ P2 | holding P1 ∪ P2 | long P1 ∪ P2 | holding P2 |
| --- | --- | --- | --- | --- | --- |
| D0 − B0 (same labels) | raw alpha | **+0.0037 [+0.0011, +0.0065]** | +0.0037 [−0.0033, +0.0106] | +0.0038 [−0.0078, +0.0162] | +0.0094 [−0.0029, +0.0269] |
| D0 − B0 | residual | +0.0027 [+0.0004, +0.0049] | +0.0002 [−0.0043, +0.0046] | +0.0001 [−0.0070, +0.0073] | +0.0030 [−0.0029, +0.0111] |
| Kronos D1 − B0 | raw alpha | −0.0003 [−0.0037, +0.0030] | +0.0060 [−0.0017, +0.0137] | +0.0049 [−0.0068, +0.0170] | +0.0116 [−0.0034, +0.0305] |
| Kronos D1 − D0 ensemble | raw alpha | **−0.0040 [−0.0069, −0.0014]** | +0.0023 [−0.0018, +0.0065] | +0.0011 [−0.0044, +0.0074] | +0.0022 [−0.0045, +0.0097] |
| Kronos D1 − D0 ensemble | residual | −0.0033 [−0.0055, −0.0013] | +0.0027 [−0.0003, +0.0057] | +0.0019 [−0.0026, +0.0064] | +0.0034 [−0.0015, +0.0083] |
| Chronos-2 − B0 | raw alpha | −0.0016 [−0.0044, +0.0012] | −0.0016 [−0.0089, +0.0065] | −0.0019 [−0.0137, +0.0116] | +0.0102 [−0.0049, +0.0281] |
| Chronos-2 − B0 | residual | **−0.0026 [−0.0048, −0.0003]** | **−0.0047 [−0.0096, −0.0002]** | −0.0050 [−0.0114, +0.0009] | +0.0002 [−0.0059, +0.0065] |
| Chronos-2 − D0 ensemble | raw alpha | −0.0053 [−0.0073, −0.0034] | −0.0053 [−0.0091, −0.0013] | −0.0057 [−0.0105, −0.0006] | +0.0008 [−0.0071, +0.0089] |
| Chronos-2 − Kronos D1 | raw alpha | −0.0013 [−0.0047, +0.0021] | **−0.0076 [−0.0128, −0.0020]** | −0.0068 [−0.0135, +0.0001] | −0.0014 [−0.0124, +0.0096] |

Old-label D0 against old B0 (§13.13): short +0.0043 [+0.0015, +0.0071], holding +0.0062 [−0.0012, +0.0136]; the DL increment is the same under both label sets.

#### 3.4 Seed stability (D0, P1 ∪ P2, raw alpha)

| Cell | Seeds 42 / 43 / 44 | Range | Ensemble | Ensemble gain | Ensemble 95% half-width | Single seed adequate |
| --- | --- | --- | --- | --- | --- | --- |
| short | 0.1171 / 0.1168 / 0.1191 | 0.0024 | 0.1212 | 0.0036 | 0.0090 | yes |
| holding | 0.0906 / 0.0846 / 0.0921 | 0.0075 | 0.0925 | 0.0034 | 0.0170 | yes |
| long | 0.0894 / 0.0830 / 0.0913 | 0.0083 | 0.0914 | 0.0035 | 0.0222 | yes |

Rule (item 9, pre-registered): both the range and the ensemble gain below half the half-width → D1 runs seed 42 only. Single-seed D1 beside single-seed D0 (seed 42): short 0.1172 vs 0.1171, holding 0.0948 vs 0.0906, long 0.0925 vs 0.0894; D1's short-cell deficit against the ensemble is exactly the ensemble gain.

#### 3.5 P1 → P2 retention (holding-cell Rank IC, P2 / P1)

| Model | Raw alpha | Residual target |
| --- | --- | --- |
| Old B0 | 0.106 → 0.034 (32%) | — |
| Old D0 | 0.109 → 0.047 (43%) | — |
| B0 variant A | 0.106 → 0.043 (41%) | 0.095 → 0.065 (68%) |
| D0 variant A | 0.108 → 0.053 (49%) | 0.094 → 0.068 (72%) |
| Kronos D1 | 0.110 → 0.055 (50%) | 0.096 → 0.071 (74%) |
| Chronos-2 | 0.100 → 0.054 (53%) | 0.088 → 0.065 (73%) |

### 4. Reading limits

- Industry and share counts are a 2026-10-22 snapshot; the 7–10% of samples that were delisted have no industry (UNK) or share count, a survivorship bias.
- Residualization removes only linear size and industry exposure; momentum, volatility and other styles remain in the target.
- Rank IC against the residual target is not tradability: a long portfolio still carries the factor returns, and the magnitude of the residual alpha is the subject of §13.17 step 0.
- P2 had already been seen; residualization was designed after observing the failure, so confirmation needs P3.
- Both D1 encoders have seed 42 only; their paired differences against the D0 ensemble include the ensemble gain (about 0.0036) and are read seed for seed.
- The residual-target long-cell P2 difference of Kronos D1 against B0, +0.0133 [+0.0022, +0.0287], is the only one of nine cell-period combinations clear of 0 and is not treated as evidence.

### 5. Disposition

Under §13.16 item 7, D0 was retrained on the new labels after H-F passed and both D1 encoders were run; the model line converges on the B0 baseline with the D0 ensemble as co-candidate. The strategy layer is registered as §13.17 under the user's 2026-10-24 bar (beat 0050 buy-and-hold); its step 0 diagnostic and disposition are in the [§13.17 step 0 report](exposure_diagnostics_results.md).

Artifacts: B0 variant A `artifacts/lgbm/launch-20261023T103805Z-2637116736/`, variant B `launch-20261023T130541Z-2860518787/` (each with `label-variant-score.json`); D0 `artifacts/xs/launch-20261023T162203Z-877225215/`; Kronos `launch-20261024T074202Z-2401818372/`, Chronos-2 `launch-20261024T104417Z-980120111/` (with `paired-vs-xs.json` / `paired-vs-d0.json`, `paired-vs-d1.json`).
